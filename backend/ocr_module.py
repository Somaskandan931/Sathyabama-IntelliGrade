"""
IntelliGrade-H — OCR Module (v9 — Hybrid Pipeline)
====================================================
Pipeline priority (per document type):

  TYPED PDFs (question papers, answer keys with selectable text):
    1. pdfplumber          — fast, 100% accurate for typed text
    2. PyMuPDF             — fallback typed text extractor

  SCANNED / HANDWRITTEN PDFs (student booklets, scanned answer keys):
    1. Google Vision API   — best overall accuracy (GOOGLE_VISION_API_KEY)
    2. Mistral OCR         — built for documents + handwriting, 1000 pages/month free (MISTRAL_API_KEY)
    3. Azure AI Vision     — 5000 pages/month free, no expiry (AZURE_VISION_KEY + AZURE_VISION_ENDPOINT)
    4. PaddleOCR           — best local engine for mixed print + handwriting
    5. Tesseract (psm 11)  — solid layout-aware fallback
    6. TrOCR line-by-line  — last resort, handwriting-trained transformer

Decision:
  • If pdfplumber extracts > 50 chars  → return immediately (typed PDF)
  • Else → run image OCR pipeline (scanned/handwritten)
  • Cloud APIs (1-3) return immediately if result > 10 chars (authoritative)
  • Local engines (4-6) compete — best real-word count × 0.7 + confidence × 0.3 wins

Install:
  pip install paddleocr paddlepaddle
  pip install pytesseract
  pip install pdf2image
  pip install pdfplumber pymupdf
  pip install mistralai
  apt-get install -y tesseract-ocr poppler-utils

Environment variables:
  GOOGLE_VISION_API_KEY=...                            (optional)
  MISTRAL_API_KEY=...                                  (optional, 1000 pages/month free)
  AZURE_VISION_KEY=...                                 (optional, 5000/month free)
  AZURE_VISION_ENDPOINT=https://<n>.cognitiveservices.azure.com
  TROCR_MODEL_PATH=microsoft/trocr-small-handwritten  (default — HuggingFace model or local path)
  TROCR_FINETUNED_PATH=models/trocr-finetuned         (default — auto-used when dir + config.json exist)
  OCR_DPI=400                                          (default)
  PADDLEOCR_LANG=en                                    (default)
"""
import logging
import os
import re
import threading

# Load .env immediately — must happen before any os.getenv() call
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(override=False)
except ImportError:
    pass
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

logger = logging.getLogger(__name__)

_OCR_DPI         = int(os.getenv("OCR_DPI", "400"))
_PADDLEOCR_LANG  = os.getenv("PADDLEOCR_LANG", "en")
_TROCR_MODEL_ENV     = os.getenv("TROCR_MODEL_PATH", "microsoft/trocr-small-handwritten")
_FINETUNED_MODEL_PATH = os.getenv("TROCR_FINETUNED_PATH", "models/trocr-finetuned")

def _resolve_trocr_model() -> str:
    """
    Return the best available TrOCR model path:
      1. Fine-tuned local model  (TROCR_FINETUNED_PATH, default 'models/trocr-finetuned')
         — used only when the directory exists AND contains model files.
      2. TROCR_MODEL_PATH env var / default  (microsoft/trocr-small-handwritten)
    """
    finetuned = Path(_FINETUNED_MODEL_PATH)
    # A valid HuggingFace saved model directory contains config.json
    if finetuned.is_dir() and (finetuned / "config.json").exists():
        logger.info("Fine-tuned TrOCR model found at '%s' — using it.", finetuned)
        return str(finetuned)
    return _TROCR_MODEL_ENV

_TROCR_MODEL = _resolve_trocr_model()

# Per-session circuit breakers — set True on first permanent failure
_GOOGLE_VISION_DISABLED = False  # trips on HTTP 403 (billing not enabled)
_MISTRAL_DISABLED       = False  # trips on HTTP 401 (bad/missing key)
_AZURE_DISABLED         = False  # trips on DNS failure (not configured)
_PADDLE_DISABLED        = False  # trips on OneDNN crash

ACADEMIC_VOCAB_CORRECTIONS = {
    "backpropogation": "backpropagation",
    "alogrithm": "algorithm",
    "databse": "database",
    "recieve": "receive",
    "seperate": "separate",
    "occured": "occurred",
    "defenition": "definition",
    "dependant": "dependent",
    "existance": "existence",
    "grammer": "grammar",
    "neccessary": "necessary",
    "occurance": "occurrence",
    "persistance": "persistence",
    "priviledge": "privilege",
    "recomend": "recommend",
    "sucess": "success",
    "temparature": "temperature",
    "transfering": "transferring",
}


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OCRResult:
    text: str
    confidence: float   # 0.0 – 1.0
    engine: str         # e.g. "pdfplumber", "paddle", "tesseract", "trocr"


# ─────────────────────────────────────────────────────────────────────────────
# Image preprocessing (shared by all engines)
# ─────────────────────────────────────────────────────────────────────────────

def _preprocess_image(img: Image.Image, dpi_hint: int = 300) -> np.ndarray:
    """
    Full preprocessing pipeline for scanned documents:
      RGB → grayscale → bilateral filter → CLAHE → adaptive threshold
    Returns a binary numpy array ready for OCR.
    """
    import cv2
    arr = np.array(img.convert("RGB"))

    # 1. Grayscale
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    # 2. Upscale if resolution is too low
    h, w = gray.shape
    if w < 1200:
        scale = 1200 / w
        gray  = cv2.resize(gray, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_CUBIC)

    # 3. Bilateral filter — smooths noise while preserving edges
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    # 4. CLAHE — improve local contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray  = clahe.apply(gray)

    # 5. Adaptive threshold — larger block size suits handwriting better
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10,
    )

    # 6. Deskew using Hough line detection
    try:
        coords = np.column_stack(np.where(binary < 128))
        if len(coords) > 100:
            angle = cv2.minAreaRect(coords.astype(np.float32))[-1]
            if angle < -45:
                angle += 90
            if abs(angle) > 0.5:
                (hh, ww) = binary.shape
                M = cv2.getRotationMatrix2D((ww / 2, hh / 2), angle, 1.0)
                binary = cv2.warpAffine(
                    binary, M, (ww, hh),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE,
                )
    except Exception:
        pass  # deskew is optional — never block OCR

    return binary


def _pil_from_preprocessed(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr)


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 0: pdfplumber — for typed PDFs
# ─────────────────────────────────────────────────────────────────────────────

def _extract_typed_text(pdf_path: str) -> str:
    """
    Extract selectable text from a PDF.  Returns "" for scanned PDFs.
    """
    # Try pdfplumber first
    try:
        import pdfplumber
        parts = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                t = page.extract_text(x_tolerance=3, y_tolerance=3)
                if t:
                    parts.append(t)
        text = "\n\n".join(parts).strip()
        if len(text) > 50:
            logger.info("pdfplumber: %d chars extracted", len(text))
            return text
    except ImportError:
        pass
    except Exception as e:
        logger.debug("pdfplumber failed: %s", e)

    # Try PyMuPDF
    try:
        import fitz
        doc  = fitz.open(pdf_path)
        text = "\n\n".join(page.get_text("text") for page in doc).strip()
        if len(text) > 50:
            logger.info("PyMuPDF: %d chars extracted", len(text))
            return text
    except ImportError:
        pass
    except Exception as e:
        logger.debug("PyMuPDF failed: %s", e)

    return ""


# ─────────────────────────────────────────────────────────────────────────────
# PDF → PIL page images
# ─────────────────────────────────────────────────────────────────────────────

def _pdf_to_images(pdf_path: str, dpi: int = _OCR_DPI) -> List[Image.Image]:
    """Render each page of a PDF as a PIL Image at the requested DPI."""
    try:
        from pdf2image import convert_from_path
        return convert_from_path(pdf_path, dpi=dpi)
    except ImportError:
        logger.warning("pdf2image not installed — trying PyMuPDF render")
    except Exception as e:
        logger.warning("pdf2image failed: %s — trying PyMuPDF", e)

    # Fallback: PyMuPDF rendering
    try:
        import fitz
        doc    = fitz.open(pdf_path)
        images = []
        zoom   = dpi / 72.0
        mat    = fitz.Matrix(zoom, zoom)
        for page in doc:
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
        return images
    except Exception as e:
        logger.error("PDF render failed: %s", e)
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 1: PaddleOCR
# ─────────────────────────────────────────────────────────────────────────────

_paddle_lock      = threading.Lock()
_paddle_singleton = None


def _get_paddle():
    global _paddle_singleton
    if _paddle_singleton is not None:
        return _paddle_singleton
    with _paddle_lock:
        if _paddle_singleton is not None:
            return _paddle_singleton
        from paddleocr import PaddleOCR
        import logging as _logging
        # Suppress OneDNN/MKL-DNN which crashes on some Windows builds
        os.environ["FLAGS_use_mkldnn"] = "0"
        os.environ["FLAGS_use_dnnl_install_path"] = "0"
        logger.info("Loading PaddleOCR (first use only)…")
        # Suppress verbose PaddleOCR/paddlex internal logging
        _loggers = ["ppocr", "paddle", "paddleocr", "paddlex", "PaddleOCR"]
        _saved   = {n: _logging.getLogger(n).level for n in _loggers}
        for n in _loggers:
            _logging.getLogger(n).setLevel(_logging.ERROR)
        try:
            # v3 (paddlex-based): no constructor args required
            _paddle_singleton = PaddleOCR()
        except TypeError:
            try:
                # v2 with show_log
                _paddle_singleton = PaddleOCR(use_angle_cls=True, lang=_PADDLEOCR_LANG, show_log=False)
            except TypeError:
                # v2 without show_log
                _paddle_singleton = PaddleOCR(use_angle_cls=True, lang=_PADDLEOCR_LANG)
        finally:
            for n in _loggers:
                _logging.getLogger(n).setLevel(_saved[n])
        logger.info("✅ PaddleOCR loaded")
        return _paddle_singleton


def _ocr_page_paddle(img: Image.Image) -> OCRResult:
    """Run PaddleOCR on a single PIL Image. Compatible with v2 and v3 API."""
    global _PADDLE_DISABLED
    if _PADDLE_DISABLED:
        return OCRResult(text="", confidence=0.0, engine="paddleocr-disabled")
    import numpy as np
    import logging as _logging
    ocr = _get_paddle()
    arr = np.array(img.convert("RGB"))

    # Suppress ppocr's per-inference warnings (angle classifier, etc.)
    _paddle_loggers = ["ppocr", "paddle", "paddleocr", "paddlex", "PaddleOCR"]
    _saved_levels   = {n: _logging.getLogger(n).level for n in _paddle_loggers}
    for n in _paddle_loggers:
        _logging.getLogger(n).setLevel(_logging.ERROR)

    lines: list[str] = []
    confs: list[float] = []

    # ── v3: predict() returns generator of result dicts ──────────
    if hasattr(ocr, "predict"):
        try:
            for res in ocr.predict(arr):
                if isinstance(res, dict):
                    texts = res.get("rec_texts", res.get("rec_text", []))
                    scores = res.get("rec_scores", res.get("rec_score", []))
                    for i, txt in enumerate(texts):
                        t = str(txt).strip()
                        if t:
                            lines.append(t)
                            c = scores[i] if i < len(scores) else 0.9
                            confs.append(float(c))
                elif hasattr(res, "rec_texts"):
                    for i, txt in enumerate(res.rec_texts):
                        t = str(txt).strip()
                        if t:
                            lines.append(t)
                            scores = getattr(res, "rec_scores", [])
                            confs.append(float(scores[i]) if i < len(scores) else 0.9)
                elif hasattr(res, "__iter__"):
                    for item in res:
                        if isinstance(item, (list, tuple)) and len(item) >= 2:
                            ti = item[1]
                            if isinstance(ti, (list, tuple)) and len(ti) >= 2:
                                t = str(ti[0]).strip()
                                if t:
                                    lines.append(t)
                                    confs.append(float(ti[1]))
            if lines:
                full_text = "\n".join(lines)
                avg_conf  = float(np.mean(confs)) if confs else 0.9
                return OCRResult(
                    text=_post_correct(full_text),
                    confidence=round(avg_conf, 4),
                    engine="paddle-v3",
                )
        except Exception as e:
            logger.debug("PaddleOCR v3 predict failed, trying v2: %s", e)

    # ── v2: ocr() returns list of [bbox, (text, conf)] ───────────
    if hasattr(ocr, "ocr"):
        try:
            result = ocr.ocr(arr, cls=True)
            if result and result[0]:
                for line in result[0]:
                    if line and len(line) >= 2:
                        text_info = line[1]
                        if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                            text, conf = str(text_info[0]), float(text_info[1])
                            if text.strip():
                                lines.append(text.strip())
                                confs.append(conf)
        except Exception as e:
            if "OneDnn" in str(e) or "onednn" in str(e).lower() or "fused_conv2d" in str(e):
                _PADDLE_DISABLED = True
                logger.warning("PaddleOCR permanently disabled for session (OneDNN crash). Fix: pip install paddleocr==2.7.3")
            else:
                logger.warning("PaddleOCR v2 ocr() failed: %s", e)

    # Restore paddle logger levels
    for n in _paddle_loggers:
        _logging.getLogger(n).setLevel(_saved_levels[n])

    if not lines:
        return OCRResult(text="", confidence=0.0, engine="paddle-failed")

    full_text = "\n".join(lines)
    avg_conf  = float(np.mean(confs)) if confs else 0.0
    return OCRResult(
        text=_post_correct(full_text),
        confidence=round(avg_conf, 4),
        engine="paddle",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 2: Tesseract
# ─────────────────────────────────────────────────────────────────────────────

def _ocr_page_tesseract(img: Image.Image) -> OCRResult:
    """
    Run Tesseract on a single PIL Image.
    Tries multiple PSM modes and picks the one with the most readable text.
    PSM 6  = assume uniform block of text (good for printed)
    PSM 11 = sparse text, find as much as possible (better for handwriting)
    PSM 4  = assume single column of text
    """
    try:
        import pytesseract
        # Re-apply path in case module was imported before dotenv loaded
        _tess_cmd = os.getenv("TESSERACT_CMD")
        if _tess_cmd:
            pytesseract.pytesseract.tesseract_cmd = _tess_cmd
    except ImportError:
        return OCRResult(text="", confidence=0.0, engine="tesseract-unavailable")

    try:
        # Preprocess — try both standard binary and original (handwriting sometimes
        # does better without aggressive thresholding)
        binary  = _preprocess_image(img)
        pil_bin = Image.fromarray(binary)
        pil_orig = img.convert("L")  # grayscale original, no harsh threshold

        best_text = ""
        best_conf = 0.0

        # Try multiple configs — pick the one producing most high-confidence words
        configs = [
            ("--oem 1 --psm 11 -l eng", pil_orig),   # LSTM + sparse, original image
            ("--oem 1 --psm 6  -l eng", pil_orig),   # LSTM + block,  original image
            ("--oem 3 --psm 11 -l eng", pil_bin),    # best+legacy + sparse, binary
            ("--oem 3 --psm 6  -l eng", pil_bin),    # best+legacy + block,  binary
        ]

        for cfg, src in configs:
            try:
                data = pytesseract.image_to_data(
                    src, config=cfg,
                    output_type=pytesseract.Output.DICT,
                )
                words = [w for w, c in zip(data["text"], data["conf"])
                         if w.strip() and str(c).lstrip("-").isdigit() and int(c) >= 0]
                confs = [int(c) for w, c in zip(data["text"], data["conf"])
                         if w.strip() and str(c).lstrip("-").isdigit() and int(c) >= 0]

                if not words:
                    continue

                text = " ".join(words).strip()
                conf = sum(confs) / len(confs) / 100.0

                # Prefer result with more real words (letters, not just symbols)
                real_words = [w for w in words if any(c.isalpha() for c in w)]
                score = len(real_words) + conf  # weighted: real words + confidence

                if score > (len([w for w in best_text.split() if any(c.isalpha() for c in w)]) + best_conf):
                    best_text = text
                    best_conf = conf
            except Exception:
                continue

        if not best_text:
            return OCRResult(text="", confidence=0.0, engine="tesseract-failed")

        logger.info("Tesseract: %d chars | conf=%.2f", len(best_text), best_conf)
        return OCRResult(
            text=_post_correct(best_text),
            confidence=round(best_conf, 4),
            engine="tesseract",
        )
    except Exception as e:
        logger.warning("Tesseract failed: %s", e)
        return OCRResult(text="", confidence=0.0, engine="tesseract-failed")


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 3: TrOCR (line-by-line — last resort)
# ─────────────────────────────────────────────────────────────────────────────

_trocr_lock      = threading.Lock()
_trocr_singleton = None


def _get_trocr(model_path: str) -> dict:
    global _trocr_singleton
    if _trocr_singleton and _trocr_singleton.get("model_path") == model_path:
        return _trocr_singleton
    with _trocr_lock:
        if _trocr_singleton and _trocr_singleton.get("model_path") == model_path:
            return _trocr_singleton
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        logger.info("Loading TrOCR: %s", model_path)
        processor = TrOCRProcessor.from_pretrained(model_path)
        model     = VisionEncoderDecoderModel.from_pretrained(model_path)
        device    = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        _trocr_singleton = {"processor": processor, "model": model,
                            "device": device, "model_path": model_path}
        logger.info("✅ TrOCR loaded on %s", device)
        return _trocr_singleton


def _ocr_line_trocr(line_img: Image.Image, ctx: dict) -> tuple:
    import torch
    import torch.nn.functional as F
    processor, model, device = ctx["processor"], ctx["model"], ctx["device"]
    if line_img.mode != "RGB":
        line_img = line_img.convert("RGB")
    px = processor(line_img, return_tensors="pt").pixel_values.to(device)
    with torch.inference_mode():
        out = model.generate(px, max_new_tokens=256)
    text = processor.batch_decode(out, skip_special_tokens=True)[0].strip()
    conf = min(0.85, 0.55 + len(text.split()) * 0.015)
    return text, conf


def _segment_lines(img: Image.Image) -> List[Image.Image]:
    """Segment a PIL Image into horizontal line crops."""
    import cv2
    arr  = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    h_proj = np.sum(binary, axis=1)
    in_line, start, lines = False, 0, []
    for y, val in enumerate(h_proj):
        if val > 0 and not in_line:
            in_line, start = True, y
        elif val == 0 and in_line:
            in_line = False
            if y - start > 5:
                lines.append((max(0, start - 4), min(img.height, y + 4)))

    if not lines:
        return [img]

    pil_lines = []
    for y1, y2 in lines:
        crop = img.crop((0, y1, img.width, y2))
        if crop.width > 20 and crop.height > 5:
            pil_lines.append(crop)
    return pil_lines or [img]


def _ocr_page_trocr(img: Image.Image) -> OCRResult:
    """TrOCR line-by-line — last resort for pure handwriting."""
    try:
        ctx   = _get_trocr(_TROCR_MODEL)
        lines = _segment_lines(img)
        texts, confs = [], []
        for line in lines:
            try:
                t, c = _ocr_line_trocr(line, ctx)
                if t.strip():
                    texts.append(t)
                    confs.append(c)
            except Exception as e:
                logger.debug("TrOCR line failed: %s", e)

        text = "\n".join(texts)
        conf = float(np.mean(confs)) if confs else 0.0

        return OCRResult(
            text=_post_correct(text),
            confidence=round(conf, 4),
            engine="trocr",
        )
    except ImportError:
        logger.warning("TrOCR not available (transformers/torch missing)")
        return OCRResult(text="", confidence=0.0, engine="trocr-unavailable")
    except Exception as e:
        logger.error("TrOCR failed: %s", e)
        return OCRResult(text="", confidence=0.0, engine="trocr-failed")



# ─────────────────────────────────────────────────────────────────────────────
# Strategy 0: Google Cloud Vision API (best for handwriting)
# ─────────────────────────────────────────────────────────────────────────────

def _get_google_vision_key() -> str:
    """Read at call time — ensures .env is already loaded by dotenv."""
    return os.getenv("GOOGLE_VISION_API_KEY", "")


def _ocr_page_google_vision(img: Image.Image) -> OCRResult:
    """
    Use Google Cloud Vision API for handwriting-optimised OCR.
    Requires GOOGLE_VISION_API_KEY in .env.
    Falls back gracefully if key is missing or request fails.
    """
    global _GOOGLE_VISION_DISABLED
    if _GOOGLE_VISION_DISABLED:
        return OCRResult(text="", confidence=0.0, engine="google-vision-disabled")
    _GOOGLE_VISION_KEY = _get_google_vision_key()
    if not _GOOGLE_VISION_KEY:
        return OCRResult(text="", confidence=0.0, engine="google-vision-disabled")

    try:
        import base64, json as _json
        import urllib.request, urllib.error

        # Convert PIL Image to base64 JPEG
        import io as _io
        buf = _io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=95)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        # Build request payload — DOCUMENT_TEXT_DETECTION is best for dense text
        payload = _json.dumps({
            "requests": [{
                "image": {"content": b64},
                "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
                "imageContext": {"languageHints": ["en"]}
            }]
        }).encode("utf-8")

        url = f"https://vision.googleapis.com/v1/images:annotate?key={_GOOGLE_VISION_KEY}"
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        with urllib.request.urlopen(req, timeout=30) as resp:
            result = _json.loads(resp.read().decode("utf-8"))

        # Parse response
        responses = result.get("responses", [{}])
        full_annotation = responses[0].get("fullTextAnnotation", {})
        text = full_annotation.get("text", "").strip()

        if not text:
            return OCRResult(text="", confidence=0.0, engine="google-vision-empty")

        # Estimate confidence from page-level confidence if available
        pages = full_annotation.get("pages", [])
        conf  = 0.95  # Google Vision is generally very accurate
        if pages:
            block_confs = []
            for page in pages:
                for block in page.get("blocks", []):
                    c = block.get("confidence", 0.95)
                    block_confs.append(c)
            if block_confs:
                conf = sum(block_confs) / len(block_confs)

        logger.info("Google Vision: %d chars | conf=%.2f", len(text), conf)
        return OCRResult(
            text=_post_correct(text),
            confidence=round(conf, 4),
            engine="google-vision",
        )

    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        if e.code in (401, 403):
            _GOOGLE_VISION_DISABLED = True
            logger.warning("Google Vision disabled for session (HTTP %d — billing required)", e.code)
        else:
            logger.warning("Google Vision HTTP error %d: %s", e.code, body[:300])
        return OCRResult(text="", confidence=0.0, engine="google-vision-failed")
    except Exception as e:
        logger.warning("Google Vision failed: %s", e)
        return OCRResult(text="", confidence=0.0, engine="google-vision-failed")


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 1b: Mistral OCR (mistral-ocr-latest) — 1000 pages/month free, purpose-built for documents
# ─────────────────────────────────────────────────────────────────────────────

def _get_mistral_api_key() -> str:
    """Read at call time — ensures .env is already loaded by dotenv."""
    return os.getenv("MISTRAL_API_KEY", "")


def _ocr_page_mistral(img: Image.Image) -> OCRResult:
    """
    Use Mistral's free vision model (pixtral-12b-2409) for OCR.
    Requires MISTRAL_API_KEY in .env.

    Setup (free tier):
      1. Go to console.mistral.ai -> API Keys -> Create new key
      2. Add to .env: MISTRAL_API_KEY=your_key_here
      3. pip install mistralai

    Note: mistral-ocr-latest requires a paid plan.
          pixtral-12b-2409 is free and works well for exam answer sheets.
    Falls back gracefully if key is missing or mistralai is not installed.
    """
    global _MISTRAL_DISABLED
    if _MISTRAL_DISABLED:
        return OCRResult(text="", confidence=0.0, engine="mistral-ocr-disabled")
    _MISTRAL_API_KEY = _get_mistral_api_key().strip()
    if not _MISTRAL_API_KEY:
        logger.debug("Mistral: MISTRAL_API_KEY not set — skipping")
        _MISTRAL_DISABLED = True
        return OCRResult(text="", confidence=0.0, engine="mistral-ocr-disabled")

    try:
        import base64
        import io as _io
        from mistralai import Mistral

        # Encode image as base64 JPEG
        buf = _io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=95)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        client = Mistral(api_key=_MISTRAL_API_KEY)

        # pixtral-12b-2409 — free vision model, strong on documents
        response = client.chat.complete(
            model="pixtral-12b-2409",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{b64}",
                        },
                        {
                            "type": "text",
                            "text": (
                                "Transcribe ALL handwritten and printed text from this exam answer sheet page. "
                                "Preserve the original structure: question numbers, bullet points, headings, "
                                "diagram labels, and tables. Output only the transcribed text, nothing else."
                            ),
                        },
                    ],
                }
            ],
            max_tokens=2000,
        )

        text = response.choices[0].message.content.strip()
        if not text:
            return OCRResult(text="", confidence=0.0, engine="mistral-ocr-empty")

        logger.info("Mistral OCR: %d chars", len(text))
        return OCRResult(
            text=_post_correct(text),
            confidence=0.88,
            engine="mistral-ocr",
        )

    except ImportError:
        logger.warning("mistralai not installed -- run: pip install mistralai")
        _MISTRAL_DISABLED = True
        return OCRResult(text="", confidence=0.0, engine="mistral-ocr-missing")
    except Exception as e:
        err_str = str(e)
        if "401" in err_str or "Unauthorized" in err_str:
            _MISTRAL_DISABLED = True
            logger.warning("Mistral disabled for session (401) — fix MISTRAL_API_KEY in .env (no spaces!)")
        elif "429" in err_str or "rate_limit" in err_str.lower() or "Rate limit" in err_str:
            _MISTRAL_DISABLED = True
            logger.warning("Mistral rate-limited (429) — disabled for this session. Free tier: ~1 req/sec.")
        else:
            logger.warning("Mistral OCR failed: %s", e)
        return OCRResult(text="", confidence=0.0, engine="mistral-ocr-failed")


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 1: Azure AI Vision (Read API) — 5000 pages/month free
# ─────────────────────────────────────────────────────────────────────────────

def _get_azure_vision_key() -> str:
    """Read at call time — ensures .env is already loaded by dotenv."""
    return os.getenv("AZURE_VISION_KEY", "")

def _get_azure_vision_endpoint() -> str:
    """Read at call time — ensures .env is already loaded by dotenv."""
    return os.getenv("AZURE_VISION_ENDPOINT", "").rstrip("/")


def _ocr_page_azure_vision(img: Image.Image) -> OCRResult:
    """
    Use Azure AI Vision Read API for handwriting-optimised OCR.
    Requires AZURE_VISION_KEY and AZURE_VISION_ENDPOINT in .env.

    Setup (free tier — 5000 pages/month, no expiry):
      1. Go to portal.azure.com → Create resource → "Computer Vision"
      2. Pricing tier: Free (F0)
      3. Copy Key 1 → AZURE_VISION_KEY
      4. Copy Endpoint → AZURE_VISION_ENDPOINT
         e.g. https://my-resource.cognitiveservices.azure.com

    Falls back gracefully if credentials are missing or the request fails.
    """
    global _AZURE_DISABLED
    if _AZURE_DISABLED:
        return OCRResult(text="", confidence=0.0, engine="azure-vision-disabled")
    _AZURE_VISION_KEY = _get_azure_vision_key()
    _AZURE_VISION_ENDPOINT = _get_azure_vision_endpoint()
    if not _AZURE_VISION_KEY or not _AZURE_VISION_ENDPOINT:
        return OCRResult(text="", confidence=0.0, engine="azure-vision-disabled")

    try:
        import base64, json as _json, time
        import urllib.request, urllib.error
        import io as _io

        # ── Encode image as JPEG bytes ────────────────────────────────────────
        buf = _io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=95)
        img_bytes = buf.getvalue()

        # ── Step 1: Submit image to Read API (async) ──────────────────────────
        analyze_url = (
            f"{_AZURE_VISION_ENDPOINT}"
            "/computervision/imageanalysis:analyze"
            "?api-version=2024-02-01"
            "&features=read"
        )

        req = urllib.request.Request(
            analyze_url,
            data=img_bytes,
            headers={
                "Ocp-Apim-Subscription-Key": _AZURE_VISION_KEY,
                "Content-Type": "application/octet-stream",
            },
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=30) as resp:
            result = _json.loads(resp.read().decode("utf-8"))

        # ── Step 2: Parse read result (synchronous in 2024-02-01 API) ─────────
        read_result = result.get("readResult", {})
        blocks = read_result.get("blocks", [])

        if not blocks:
            return OCRResult(text="", confidence=0.0, engine="azure-vision-empty")

        lines_text, word_confs = [], []
        for block in blocks:
            for line in block.get("lines", []):
                line_words = []
                for word in line.get("words", []):
                    content = word.get("text", "").strip()
                    conf = word.get("confidence", 0.9)
                    if content:
                        line_words.append(content)
                        word_confs.append(conf)
                if line_words:
                    lines_text.append(" ".join(line_words))

        text = "\n".join(lines_text).strip()
        if not text:
            return OCRResult(text="", confidence=0.0, engine="azure-vision-empty")

        avg_conf = round(sum(word_confs) / len(word_confs), 4) if word_confs else 0.9
        logger.info("Azure Vision: %d chars | %d lines | conf=%.2f",
                    len(text), len(lines_text), avg_conf)

        return OCRResult(
            text=_post_correct(text),
            confidence=avg_conf,
            engine="azure-vision",
        )

    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        if e.code in (401, 403):
            _AZURE_DISABLED = True
            logger.warning("Azure disabled for session (HTTP %d)", e.code)
        else:
            logger.warning("Azure Vision HTTP %d: %s", e.code, body[:300])
        return OCRResult(text="", confidence=0.0, engine="azure-vision-failed")
    except Exception as e:
        if "getaddrinfo failed" in str(e) or "Name or service not known" in str(e):
            _AZURE_DISABLED = True
            logger.warning("Azure disabled for session (DNS failure — AZURE_VISION_ENDPOINT not set)")
        else:
            logger.warning("Azure Vision failed: %s", e)
        return OCRResult(text="", confidence=0.0, engine="azure-vision-failed")


# ─────────────────────────────────────────────────────────────────────────────
# Core per-page hybrid OCR
# ─────────────────────────────────────────────────────────────────────────────

def _ocr_image(img: Image.Image) -> OCRResult:
    """
    Run hybrid OCR on a single PIL Image.
    Order: Google Vision → Mistral OCR → Azure Vision → PaddleOCR → Tesseract → TrOCR

    Cloud APIs (1-3) return immediately when they produce a good result.
    Local engines (4-6) all run and the best score wins.
    """
    # 1. Google Vision (best overall — used if GOOGLE_VISION_API_KEY is set)
    if not _GOOGLE_VISION_DISABLED and _get_google_vision_key():
        try:
            r = _ocr_page_google_vision(img)
            if r.text.strip() and len(r.text.strip()) > 10:
                logger.info("Using Google Vision result (%d chars)", len(r.text))
                return r
        except Exception as e:
            logger.warning("Google Vision error: %s — trying Mistral next", e)

    # 2. Mistral OCR (1000 pages/month free — used if MISTRAL_API_KEY is set)
    if not _MISTRAL_DISABLED and _get_mistral_api_key():
        try:
            r = _ocr_page_mistral(img)
            if r.text.strip() and len(r.text.strip()) > 10:
                logger.info("Using Mistral OCR result (%d chars)", len(r.text))
                return r
        except Exception as e:
            logger.warning("Mistral OCR error: %s — trying Azure next", e)

    # 3. Azure AI Vision (5000 pages/month free — used if AZURE_VISION_KEY is set)
    if not _AZURE_DISABLED and _get_azure_vision_key() and _get_azure_vision_endpoint():
        try:
            r = _ocr_page_azure_vision(img)
            if r.text.strip() and len(r.text.strip()) > 10:
                logger.info("Using Azure Vision result (%d chars)", len(r.text))
                return r
        except Exception as e:
            logger.warning("Azure Vision error: %s — falling back to local OCR", e)

    results = []

    # 4. PaddleOCR (best local engine for mixed print + handwriting)
    try:
        r = _ocr_page_paddle(img)
        if r.text.strip() and len(r.text.strip()) > 10:
            logger.info("PaddleOCR: %d chars | conf=%.2f", len(r.text), r.confidence)
            results.append(r)
    except Exception as e:
        logger.debug("PaddleOCR unavailable: %s", e)

    # 5. Tesseract (solid layout-aware fallback)
    try:
        r = _ocr_page_tesseract(img)
        if r.text.strip() and len(r.text.strip()) > 10:
            results.append(r)
    except Exception as e:
        logger.debug("Tesseract unavailable: %s", e)

    # 6. TrOCR (last resort — slowest but handwriting-trained transformer)
    if not results or (results and max(r.confidence for r in results) < 0.4):
        logger.info("Falling back to TrOCR line-by-line…")
        r = _ocr_page_trocr(img)
        if r.text.strip():
            results.append(r)

    if not results:
        return OCRResult(text="", confidence=0.0, engine="none")

    # Return result with most real words (letters) weighted by confidence
    def _score(r):
        real_words = [w for w in r.text.split() if any(c.isalpha() for c in w)]
        return len(real_words) * 0.7 + r.confidence * 0.3

    best = max(results, key=_score)
    return best


# ─────────────────────────────────────────────────────────────────────────────
# TrOCREngine — kept for backward compat (evaluator.py uses it)
# ─────────────────────────────────────────────────────────────────────────────

class TrOCREngine:
    """
    Backward-compatible wrapper. Internally uses the hybrid pipeline for
    full pages, and pure TrOCR only for pre-segmented lines.
    """

    MODEL_NAME = "microsoft/trocr-small-handwritten"

    def __init__(self, model_path: Optional[str] = None):
        self._model_path = model_path or self.MODEL_NAME

    def warmup(self):
        try:
            _get_trocr(self._model_path)
        except Exception as e:
            logger.warning("TrOCR warmup skipped: %s", e)

    def recognize(self, image: Image.Image) -> OCRResult:
        """Recognize a single image — uses full hybrid pipeline."""
        return _ocr_image(image)

    def recognize_lines(self, line_images: List[Image.Image]) -> OCRResult:
        """Recognize pre-segmented line images with TrOCR."""
        try:
            ctx = _get_trocr(self._model_path)
        except Exception as e:
            logger.warning("TrOCR not available for line recognition: %s", e)
            # Fall back to tesseract for each line
            texts = []
            for li in line_images:
                r = _ocr_page_tesseract(li)
                if r.text.strip():
                    texts.append(r.text)
            return OCRResult(
                text="\n".join(texts),
                confidence=0.6,
                engine="tesseract-lines",
            )

        texts, confs = [], []
        for i, li in enumerate(line_images):
            try:
                t, c = _ocr_line_trocr(li, ctx)
                if t.strip():
                    texts.append(t)
                    confs.append(c)
            except Exception as e:
                logger.debug("Line %d TrOCR failed: %s", i, e)

        return OCRResult(
            text="\n".join(texts),
            confidence=round(float(np.mean(confs)) if confs else 0.0, 4),
            engine="trocr",
        )


# ─────────────────────────────────────────────────────────────────────────────
# FastPreprocessor — kept for backward compat (preprocessor.py uses it)
# ─────────────────────────────────────────────────────────────────────────────

class FastPreprocessor:
    @staticmethod
    def enhance_for_ocr(image: Image.Image, scale: float = 1.0) -> Image.Image:
        if image.mode != "RGB":
            image = image.convert("RGB")
        if scale != 1.0:
            w, h  = image.size
            image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        image = image.filter(ImageFilter.SHARPEN)
        image = ImageEnhance.Contrast(image).enhance(1.6)
        image = ImageEnhance.Brightness(image).enhance(1.05)
        return image


# ─────────────────────────────────────────────────────────────────────────────
# OCRModule — main public interface (unchanged API)
# ─────────────────────────────────────────────────────────────────────────────

class OCRModule:
    """
    High-level OCR interface with hybrid pipeline.

    For TYPED PDFs  → pdfplumber/PyMuPDF (instant, perfect)
    For SCANNED PDFs → PaddleOCR → Tesseract → TrOCR (image OCR)

    All existing call sites work unchanged:
      ocr_module.extract_text(path)
      ocr_module.extract_from_pdf(pdf_path)
      ocr_module.extract_text_from_image(image)
    """

    def __init__(self, trocr_model_path: Optional[str] = None):
        # Re-resolve each time: fine-tuned model may have been saved after startup
        path             = trocr_model_path or _resolve_trocr_model()
        self.engine      = TrOCREngine(path)
        self.engine_name = "hybrid"

    def warmup(self):
        """Pre-load available OCR engines. Called at API startup."""
        # Try to load PaddleOCR
        try:
            _get_paddle()
            logger.info("PaddleOCR warmed up")
        except Exception as e:
            logger.warning("PaddleOCR warmup skipped: %s", e)

        # Try to load TrOCR
        try:
            self.engine.warmup()
        except Exception as e:
            logger.warning("TrOCR warmup skipped: %s", e)

        # Verify Tesseract — set cmd path from .env BEFORE checking version
        try:
            import pytesseract
            tesseract_cmd = os.getenv("TESSERACT_CMD")
            if tesseract_cmd:
                pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
            pytesseract.get_tesseract_version()
            logger.info("Tesseract available at: %s",
                        pytesseract.pytesseract.tesseract_cmd)
        except Exception as e:
            logger.warning("Tesseract not found: %s", e)

    @staticmethod
    def _resolve_path(image_input) -> Optional[Path]:
        if isinstance(image_input, (str, Path)):
            p = Path(image_input)
            return p if p.is_absolute() else Path.cwd() / p
        return None

    def extract_text(self, image_input,
                     use_line_segmentation: bool = False) -> OCRResult:
        """
        Main entry point. Accepts file path (PDF/image), PIL Image, or numpy array.

        For PDFs:
          • tries pdfplumber first (typed PDFs) → returns immediately if successful
          • falls back to image OCR pipeline for scanned PDFs

        For images:
          • runs full hybrid image OCR pipeline
        """
        resolved = self._resolve_path(image_input)

        # ── PDF path ──────────────────────────────────────────────────
        if resolved is not None and resolved.suffix.lower() == ".pdf":
            return self._extract_pdf_smart(str(resolved))

        # ── File path (image) ─────────────────────────────────────────
        if resolved is not None:
            if not resolved.exists():
                raise FileNotFoundError(f"Cannot load image: {resolved}")
            image_input = str(resolved)

        # ── PIL Image / numpy / file path → OCR ───────────────────────
        try:
            pil_img = self._to_pil(image_input)
            result  = _ocr_image(pil_img)
            logger.info("OCR done. engine=%s conf=%.3f chars=%d",
                        result.engine, result.confidence, len(result.text))
            return result
        except Exception as e:
            logger.error("extract_text failed: %s", e)
            return OCRResult(text="", confidence=0.0, engine="failed")

    def extract_text_from_image(self, image_input,
                                use_line_segmentation: bool = False) -> OCRResult:
        """Alias for extract_text() — used by student_answer_parser."""
        return self.extract_text(image_input, use_line_segmentation)

    def extract_from_pdf(self, pdf_path: str) -> List[OCRResult]:
        """
        Process each page of a PDF and return one OCRResult per page.
        Tries typed-text extraction first; falls back to image OCR per page.
        """
        pdf_path = str(Path(pdf_path).resolve())

        # Typed PDF → single result for whole doc, split into page-like chunks
        typed = _extract_typed_text(pdf_path)
        if typed:
            # Split by form-feed or double-newline to approximate pages
            chunks = re.split(r'\f|\n{3,}', typed)
            return [
                OCRResult(text=c.strip(), confidence=1.0, engine="pdfplumber")
                for c in chunks if c.strip()
            ] or [OCRResult(text=typed, confidence=1.0, engine="pdfplumber")]

        # Scanned PDF → render pages and OCR each
        images = _pdf_to_images(pdf_path, dpi=_OCR_DPI)
        if not images:
            return [OCRResult(text="", confidence=0.0, engine="render-failed")]

        results = []
        for i, page_img in enumerate(images):
            logger.info("OCR page %d/%d", i + 1, len(images))
            results.append(_ocr_image(page_img))
        return results

    def _extract_pdf_smart(self, pdf_path: str) -> OCRResult:
        """
        Smart PDF extractor:
          typed  → pdfplumber (conf=1.0)
          scanned → image OCR pipeline
        """
        # Try typed text
        typed = _extract_typed_text(pdf_path)
        if typed:
            return OCRResult(text=typed, confidence=1.0, engine="pdfplumber")

        # Scanned — render and OCR all pages
        images = _pdf_to_images(pdf_path, dpi=_OCR_DPI)
        if not images:
            return OCRResult(text="", confidence=0.0, engine="render-failed")

        page_results = []
        for i, page_img in enumerate(images):
            logger.info("OCR page %d/%d", i + 1, len(images))
            page_results.append(_ocr_image(page_img))

        combined = "\n\n".join(r.text for r in page_results if r.text)
        avg_conf  = float(np.mean([r.confidence for r in page_results])) if page_results else 0.0
        engines   = list(dict.fromkeys(r.engine for r in page_results))  # unique, ordered

        return OCRResult(
            text=combined,
            confidence=round(avg_conf, 4),
            engine="+".join(engines),
        )

    @staticmethod
    def _to_pil(image_input) -> Image.Image:
        """Convert various input types to a PIL Image."""
        if isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        if isinstance(image_input, np.ndarray):
            return Image.fromarray(image_input).convert("RGB")
        if isinstance(image_input, bytes):
            import io
            return Image.open(io.BytesIO(image_input)).convert("RGB")
        if isinstance(image_input, (str, Path)):
            return Image.open(str(image_input)).convert("RGB")
        raise ValueError(f"Unsupported image input type: {type(image_input)}")


# ─────────────────────────────────────────────────────────────────────────────
# Post-OCR text corrections
# ─────────────────────────────────────────────────────────────────────────────

def _post_correct(text: str) -> str:
    if not text:
        return text
    # Common OCR digit-letter confusions
    text = re.sub(r'\b0([a-zA-Z])', r'o\1', text)
    text = re.sub(r'([a-zA-Z])0\b', r'\1o', text)
    # Collapse multiple spaces
    text = re.sub(r'[ \t]{2,}', ' ', text)
    # Apply academic vocabulary corrections
    words = text.split()
    corrected = []
    for word in words:
        clean = word.lower().strip(".,;:!?()[]\"'")
        if clean in ACADEMIC_VOCAB_CORRECTIONS:
            word = word.replace(clean, ACADEMIC_VOCAB_CORRECTIONS[clean])
        corrected.append(word)
    return " ".join(corrected)