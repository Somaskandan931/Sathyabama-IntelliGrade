"""
IntelliGrade-H - OCR Module (v5 - No Tesseract)
================================================
Tesseract removed entirely. Two engine options:

  easyocr  (DEFAULT) — EasyOCR: no Tesseract, CPU/GPU, good handwriting,
                        ~200 MB download on first run, ~1-3 s/page on CPU.
                        Set OCR_ENGINE=easyocr in .env

  trocr    (QUALITY) — Microsoft TrOCR-small-handwritten: 334 MB (was 1.33 GB),
                        best accuracy for cursive/messy handwriting,
                        ~8-20 s/page on CPU, ~1-2 s on GPU.
                        Set OCR_ENGINE=trocr in .env

  ensemble (HYBRID)  — EasyOCR first; TrOCR only when EasyOCR conf < 0.50.
                        Set OCR_ENGINE=ensemble in .env

Install:
  pip install easyocr                    # for easyocr / ensemble
  pip install transformers torch         # for trocr / ensemble
"""

import logging
import re
import threading
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

logger = logging.getLogger(__name__)

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


@dataclass
class OCRResult:
    text: str
    confidence: float   # 0.0 - 1.0
    engine: str


# ─────────────────────────────────────────────────────────────────────────────
# Fast PIL Preprocessor
# ─────────────────────────────────────────────────────────────────────────────

class FastPreprocessor:
    @staticmethod
    def enhance_for_ocr(image: Image.Image, scale: float = 1.0) -> Image.Image:
        if image.mode != "RGB":
            image = image.convert("RGB")
        if scale != 1.0:
            w, h = image.size
            image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        image = image.filter(ImageFilter.SHARPEN)
        image = ImageEnhance.Contrast(image).enhance(1.6)
        image = ImageEnhance.Brightness(image).enhance(1.05)
        return image


EnhancedPreprocessor = FastPreprocessor   # backward-compat alias


# ─────────────────────────────────────────────────────────────────────────────
# EasyOCR Engine  — DEFAULT (no Tesseract, CPU/GPU, ~200 MB download once)
# ─────────────────────────────────────────────────────────────────────────────

_easyocr_lock      = threading.Lock()
_easyocr_singleton = None   # easyocr.Reader instance


def _get_easyocr():
    """Load EasyOCR reader once per process (singleton)."""
    global _easyocr_singleton
    if _easyocr_singleton is not None:
        return _easyocr_singleton
    with _easyocr_lock:
        if _easyocr_singleton is not None:
            return _easyocr_singleton
        import easyocr
        import torch
        use_gpu = torch.cuda.is_available()
        logger.info("Loading EasyOCR reader (gpu=%s) — first use only...", use_gpu)
        _easyocr_singleton = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)
        logger.info("✅ EasyOCR ready.")
        return _easyocr_singleton


class EasyOCREngine:
    """
    EasyOCR — primary OCR engine, no Tesseract required.

    Speed    : ~1-3 s/page on CPU, ~0.3 s on GPU
    Quality  : Good for both printed and handwritten text
    Download : ~200 MB on first run (model weights cached in ~/.EasyOCR/)
    """

    def recognize(self, image: Image.Image) -> OCRResult:
        reader = _get_easyocr()

        enhanced = FastPreprocessor.enhance_for_ocr(image, scale=1.5)

        import numpy as np
        img_array = np.array(enhanced)

        try:
            results = reader.readtext(img_array, detail=1, paragraph=False)
        except Exception as e:
            logger.warning("EasyOCR readtext failed: %s", e)
            return OCRResult(text="", confidence=0.0, engine="easyocr-failed")

        if not results:
            return OCRResult(text="", confidence=0.0, engine="easyocr")

        # results: list of ([bbox], text, confidence)
        texts = []
        confs = []
        for (_bbox, text, conf) in results:
            text = text.strip()
            if text:
                texts.append(text)
                confs.append(conf)

        combined_text = " ".join(texts)
        avg_conf      = float(np.mean(confs)) if confs else 0.0

        return OCRResult(
            text=_post_correct(combined_text),
            confidence=round(avg_conf, 4),
            engine="easyocr",
        )


# ─────────────────────────────────────────────────────────────────────────────
# TrOCR Engine  — QUALITY (trocr-small: 334 MB, 4x faster than trocr-base)
# ─────────────────────────────────────────────────────────────────────────────

_trocr_lock:      threading.Lock = threading.Lock()
_trocr_singleton: Optional[dict] = None


def _get_trocr(model_path: str) -> dict:
    """Load TrOCR processor + model exactly once per process."""
    global _trocr_singleton
    if _trocr_singleton is not None and _trocr_singleton.get("model_path") == model_path:
        return _trocr_singleton
    with _trocr_lock:
        if _trocr_singleton is not None and _trocr_singleton.get("model_path") == model_path:
            return _trocr_singleton
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        logger.info("Loading TrOCR model: %s  (first use only)", model_path)
        processor = TrOCRProcessor.from_pretrained(model_path)
        model     = VisionEncoderDecoderModel.from_pretrained(model_path)
        device    = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        _trocr_singleton = {
            "processor": processor, "model": model,
            "device": device, "model_path": model_path,
        }
        logger.info("✅ TrOCR loaded on %s", device)
        return _trocr_singleton


class TrOCREngine:
    """
    TrOCR-small-handwritten — 334 MB, 4× faster than trocr-base (1.33 GB).
    Best accuracy for messy/cursive handwriting.

    Adaptive strategy:
      1. Greedy decode (fastest)
      2. If confidence < 0.70 → beam-2
      3. If still < 0.55 → 1.5× scale + greedy
    """

    # trocr-small-handwritten: 334 MB — much faster than trocr-base (1.33 GB)
    MODEL_NAME      = "microsoft/trocr-small-handwritten"
    FAST_THRESHOLD  = 0.70
    SCALE_THRESHOLD = 0.55
    MAX_NEW_TOKENS  = 128

    def __init__(self, model_path: Optional[str] = None):
        self._model_path = model_path or self.MODEL_NAME

    def _infer(self, image: Image.Image, num_beams: int = 1) -> tuple:
        import torch
        import torch.nn.functional as F
        ctx = _get_trocr(self._model_path)
        processor, model, device = ctx["processor"], ctx["model"], ctx["device"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        px = processor(image, return_tensors="pt").pixel_values.to(device)
        with torch.inference_mode():
            out = model.generate(
                px,
                num_beams=num_beams,
                output_scores=(num_beams > 1),
                return_dict_in_generate=(num_beams > 1),
                max_new_tokens=self.MAX_NEW_TOKENS,
                early_stopping=(num_beams > 1),
            )
        if num_beams > 1:
            text = processor.batch_decode(out.sequences, skip_special_tokens=True)[0].strip()
            conf = float(np.mean([F.softmax(s, dim=-1).max().item()
                                  for s in out.scores])) if out.scores else 0.80
        else:
            text = processor.batch_decode(out, skip_special_tokens=True)[0].strip()
            conf = min(0.90, 0.60 + len(text.split()) * 0.01)
        return text, conf

    def recognize(self, image: Image.Image) -> OCRResult:
        enhanced   = FastPreprocessor.enhance_for_ocr(image, scale=1.0)
        text, conf = self._infer(enhanced, num_beams=1)

        if conf >= self.FAST_THRESHOLD:
            return OCRResult(text=_post_correct(text),
                             confidence=round(conf, 4), engine="trocr-fast")

        t2, c2 = self._infer(enhanced, num_beams=2)
        if c2 > conf:
            text, conf = t2, c2

        if conf < self.SCALE_THRESHOLD:
            try:
                e15      = FastPreprocessor.enhance_for_ocr(image, scale=1.5)
                t3, c3   = self._infer(e15, num_beams=1)
                if c3 > conf:
                    text, conf = t3, c3
            except Exception as e:
                logger.warning("1.5x scale fallback failed: %s", e)

        return OCRResult(text=_post_correct(text),
                         confidence=round(conf, 4), engine="trocr-adaptive")

    def recognize_lines(self, line_images: List[Image.Image]) -> OCRResult:
        texts, confs = [], []
        for i, li in enumerate(line_images):
            try:
                r = self.recognize(li)
                if r.text.strip():
                    texts.append(r.text)
                    confs.append(r.confidence)
            except Exception as e:
                logger.warning("Line %d failed: %s", i, e)
        return OCRResult(
            text=" ".join(texts),
            confidence=round(float(np.mean(confs)) if confs else 0.0, 4),
            engine="trocr-adaptive",
        )

    def fine_tune(self, dataset_path: str, output_dir: str,
                  epochs: int = 5, batch_size: int = 8):
        from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments,
                                  default_data_collator)
        from torch.utils.data import Dataset as TorchDataset
        ctx = _get_trocr(self._model_path)

        class HWDataset(TorchDataset):
            def __init__(self, data_dir, processor):
                self.data, self.processor = [], processor
                labels_file = Path(data_dir) / "labels.txt"
                img_dir     = Path(data_dir) / "images"
                with open(labels_file, encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split("\t")
                        if len(parts) == 2:
                            p = img_dir / parts[0]
                            if p.exists():
                                self.data.append((str(p), parts[1]))
            def __len__(self): return len(self.data)
            def __getitem__(self, idx):
                img_path, text = self.data[idx]
                image  = Image.open(img_path).convert("RGB")
                image  = FastPreprocessor.enhance_for_ocr(image, scale=1.5)
                enc    = self.processor(image, return_tensors="pt")
                labels = self.processor.tokenizer(
                    text, return_tensors="pt", padding="max_length",
                    max_length=128, truncation=True,
                ).input_ids.squeeze()
                labels[labels == self.processor.tokenizer.pad_token_id] = -100
                return {"pixel_values": enc.pixel_values.squeeze(), "labels": labels}

        ds   = HWDataset(dataset_path, ctx["processor"])
        args = Seq2SeqTrainingArguments(
            output_dir=output_dir, num_train_epochs=epochs,
            per_device_train_batch_size=batch_size, predict_with_generate=True,
            save_steps=500, logging_steps=100,
            fp16=__import__("torch").cuda.is_available(),
            warmup_steps=100, weight_decay=0.01,
        )
        Seq2SeqTrainer(
            model=ctx["model"], args=args, train_dataset=ds,
            data_collator=default_data_collator,
        ).train()
        ctx["model"].save_pretrained(output_dir)
        ctx["processor"].save_pretrained(output_dir)
        logger.info("Fine-tuned model saved to %s", output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble Engine  — EasyOCR first, TrOCR only on low confidence
# ─────────────────────────────────────────────────────────────────────────────

class EnsembleEngine:
    """
    EasyOCR-first ensemble.
    TrOCR is only called when EasyOCR confidence < TROCR_BOOST_THRESHOLD.
    In the common case only EasyOCR runs (~1-3 s/page on CPU).
    """
    TROCR_BOOST_THRESHOLD = 0.50

    def __init__(self, trocr_model_path: Optional[str] = None):
        self.easyocr = EasyOCREngine()
        self.trocr   = TrOCREngine(trocr_model_path)

    def recognize(self, image: Image.Image) -> OCRResult:
        easy = self.easyocr.recognize(image)
        if easy.confidence >= self.TROCR_BOOST_THRESHOLD:
            return easy
        logger.info(
            "Ensemble: EasyOCR conf=%.3f < %.2f, trying TrOCR",
            easy.confidence, self.TROCR_BOOST_THRESHOLD,
        )
        try:
            trocr = self.trocr.recognize(image)
            if trocr.confidence > easy.confidence:
                return trocr
        except Exception as e:
            logger.warning("TrOCR boost failed: %s", e)
        return easy

    def recognize_lines(self, line_images: List[Image.Image]) -> OCRResult:
        texts, confs = [], []
        for li in line_images:
            r = self.easyocr.recognize(li)
            if r.text.strip():
                texts.append(r.text)
                confs.append(r.confidence)
        return OCRResult(
            text=" ".join(texts),
            confidence=round(float(np.mean(confs)) if confs else 0.0, 4),
            engine="ensemble-easyocr",
        )


# ─────────────────────────────────────────────────────────────────────────────
# OCR Module — public interface
# ─────────────────────────────────────────────────────────────────────────────

class OCRModule:
    """
    High-level OCR interface. No Tesseract required.

    OCR_ENGINE=easyocr   → EasyOCR only       (DEFAULT, ~1-3 s/page CPU)
    OCR_ENGINE=trocr     → TrOCR-small only   (~8-20 s/page CPU, best quality)
    OCR_ENGINE=ensemble  → EasyOCR + TrOCR    (EasyOCR fast path, TrOCR fallback)
    """

    def __init__(self, engine: str = "easyocr", trocr_model_path: Optional[str] = None):
        self.engine_name = engine
        if engine == "trocr":
            self.primary = TrOCREngine(trocr_model_path)
        elif engine == "ensemble":
            self.primary = EnsembleEngine(trocr_model_path)
        else:
            self.primary = EasyOCREngine()
        # EasyOCR is always the fallback (no Tesseract)
        self._fallback_engine = EasyOCREngine()

    @staticmethod
    def _resolve_path(image_input) -> Optional[Path]:
        if isinstance(image_input, (str, Path)):
            p = Path(image_input)
            if not p.is_absolute():
                p = Path.cwd() / p
            return p
        return None

    def extract_text(self, image_input, use_line_segmentation: bool = False) -> OCRResult:
        """
        Main entry point. Accepts file path, bytes, PIL Image, or numpy array.
        use_line_segmentation only applies to trocr engine.
        """
        from backend.preprocessor import ImagePreprocessor
        preprocessor = ImagePreprocessor()

        resolved = self._resolve_path(image_input)
        if resolved is not None and resolved.suffix.lower() == ".pdf":
            return self._extract_pdf_as_single(str(resolved))
        if resolved is not None:
            if not resolved.exists():
                raise FileNotFoundError(f"Cannot load image: {resolved}")
            image_input = str(resolved)

        do_segment = use_line_segmentation and self.engine_name == "trocr"

        try:
            if do_segment:
                line_images = preprocessor.segment_lines(image_input)
                if line_images and hasattr(self.primary, "recognize_lines"):
                    result = self.primary.recognize_lines(line_images)
                else:
                    pil_img = preprocessor.preprocess_to_pil(image_input)
                    result  = self.primary.recognize(pil_img)
            else:
                pil_img = preprocessor.preprocess_to_pil(image_input)
                result  = self.primary.recognize(pil_img)

            logger.info("OCR done. Engine=%s Conf=%.3f Preview: %s",
                        result.engine, result.confidence, result.text[:80])
            return result

        except Exception as e:
            logger.warning("Primary OCR failed (%s). Using EasyOCR fallback.", e)
            try:
                pil_img = preprocessor.preprocess_to_pil(image_input)
                return self._fallback_engine.recognize(pil_img)
            except Exception as e2:
                logger.error("Fallback OCR also failed: %s", e2)
                return OCRResult(text="", confidence=0.0, engine="failed")

    def extract_from_pdf(self, pdf_path: str) -> List[OCRResult]:
        from pdf2image import convert_from_path
        pdf_path = str(Path(pdf_path).resolve())
        pages    = convert_from_path(pdf_path, dpi=150)
        results  = []
        for i, page_img in enumerate(pages):
            logger.info("PDF page %d/%d", i + 1, len(pages))
            results.append(self.extract_text(page_img, use_line_segmentation=False))
        return results

    def _extract_pdf_as_single(self, pdf_path: str) -> OCRResult:
        page_results = self.extract_from_pdf(pdf_path)
        if not page_results:
            return OCRResult(text="", confidence=0.0, engine="pdf")
        combined_text = "\n".join(r.text for r in page_results if r.text)
        avg_conf = sum(r.confidence for r in page_results) / len(page_results)
        return OCRResult(
            text=combined_text,
            confidence=round(avg_conf, 4),
            engine=f"pdf+{self.engine_name}",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Post-OCR corrections
# ─────────────────────────────────────────────────────────────────────────────

def _post_correct(text: str) -> str:
    if not text:
        return text
    text = re.sub(r'\b0([a-z])', r'o\1', text)
    text = re.sub(r'([a-z])0\b', r'\1o', text)
    text = re.sub(r' {2,}', ' ', text)
    words, corrected = text.split(), []
    for word in words:
        clean = word.lower().strip(".,;:!?()[]\"'")
        if clean in ACADEMIC_VOCAB_CORRECTIONS:
            word = word.replace(clean, ACADEMIC_VOCAB_CORRECTIONS[clean])
        corrected.append(word)
    return " ".join(corrected)