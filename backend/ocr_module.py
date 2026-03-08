"""
IntelliGrade-H - OCR Module (v6 — TrOCR Only)
==============================================
EasyOCR removed entirely. TrOCR is the sole OCR engine.

  trocr    — Microsoft TrOCR-small-handwritten (334 MB)
             Best accuracy for cursive/messy handwriting.
             ~8-20 s/page on CPU, ~1-2 s on GPU.

To use a fine-tuned model, set TROCR_MODEL_PATH in .env:
  TROCR_MODEL_PATH=models/trocr-finetuned

Install:
  pip install transformers torch
  pip install pdf2image   # for PDF support (also needs poppler)
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
    confidence: float   # 0.0 – 1.0
    engine: str


# ─────────────────────────────────────────────────────────────────────────────
# PIL Preprocessor (used before feeding image to TrOCR)
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


# ─────────────────────────────────────────────────────────────────────────────
# TrOCR Engine — sole OCR engine
# ─────────────────────────────────────────────────────────────────────────────

_trocr_lock:      threading.Lock = threading.Lock()
_trocr_singleton: Optional[dict] = None


def _get_trocr(model_path: str) -> dict:
    """Load TrOCR processor + model exactly once per process (singleton)."""
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
            "processor": processor,
            "model":     model,
            "device":    device,
            "model_path": model_path,
        }
        logger.info("✅ TrOCR loaded on %s", device)
        return _trocr_singleton


class TrOCREngine:
    """
    TrOCR-small-handwritten — 334 MB, best accuracy for messy/cursive handwriting.

    Adaptive decode strategy:
      1. Greedy (fastest)
      2. If confidence < FAST_THRESHOLD  → beam-2
      3. If still < SCALE_THRESHOLD      → 1.5× upscale + greedy
    """

    MODEL_NAME      = "microsoft/trocr-small-handwritten"
    FAST_THRESHOLD  = 0.75   # below this, try beam search
    SCALE_THRESHOLD = 0.55   # below this, try rescaled image
    MAX_NEW_TOKENS  = 256

    def __init__(self, model_path: Optional[str] = None):
        self._model_path = model_path or self.MODEL_NAME

    def warmup(self):
        """Pre-load model weights so first real request is fast."""
        _get_trocr(self._model_path)

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

        # Beam search if greedy confidence is low
        if conf < self.FAST_THRESHOLD:
            t2, c2 = self._infer(enhanced, num_beams=2)
            if c2 > conf:
                text, conf = t2, c2

        # Rescaled pass if still low
        if conf < self.SCALE_THRESHOLD:
            try:
                e15    = FastPreprocessor.enhance_for_ocr(image, scale=1.5)
                t3, c3 = self._infer(e15, num_beams=1)
                if c3 > conf:
                    text, conf = t3, c3
            except Exception as e:
                logger.warning("1.5x scale fallback failed: %s", e)

        return OCRResult(
            text=_post_correct(text),
            confidence=round(conf, 4),
            engine="trocr",
        )

    def recognize_lines(self, line_images: List[Image.Image]) -> OCRResult:
        """Recognize a list of pre-segmented line images and join results."""
        texts, confs = [], []
        for i, li in enumerate(line_images):
            try:
                r = self.recognize(li)
                if r.text.strip():
                    texts.append(r.text)
                    confs.append(r.confidence)
            except Exception as e:
                logger.warning("Line %d recognition failed: %s", i, e)
        return OCRResult(
            text=" ".join(texts),
            confidence=round(float(np.mean(confs)) if confs else 0.0, 4),
            engine="trocr",
        )

    def fine_tune(self, dataset_path: str, output_dir: str,
                  epochs: int = 5, batch_size: int = 8):
        """Fine-tune TrOCR on a custom handwriting dataset. See scripts/train_trocr.py."""
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
# OCR Module — public interface
# ─────────────────────────────────────────────────────────────────────────────

class OCRModule:
    """
    High-level OCR interface using TrOCR exclusively.

    TROCR_MODEL_PATH=microsoft/trocr-small-handwritten  (default)
    TROCR_MODEL_PATH=models/trocr-finetuned             (after fine-tuning)
    """

    def __init__(self, trocr_model_path: Optional[str] = None):
        import os
        path = trocr_model_path or os.getenv(
            "TROCR_MODEL_PATH", "microsoft/trocr-small-handwritten"
        )
        self.engine      = TrOCREngine(path)
        self.engine_name = "trocr"

    def warmup(self):
        """Pre-load TrOCR weights. Called at API startup to avoid cold-start delay."""
        self.engine.warmup()

    @staticmethod
    def _resolve_path(image_input) -> Optional[Path]:
        if isinstance(image_input, (str, Path)):
            p = Path(image_input)
            return p if p.is_absolute() else Path.cwd() / p
        return None

    def extract_text(self, image_input,
                     use_line_segmentation: bool = False) -> OCRResult:
        """
        Main entry point. Accepts file path, bytes, PIL Image, or numpy array.
        use_line_segmentation=True segments the page into lines first (better
        for dense multi-line answers on a full page).
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

        try:
            if use_line_segmentation:
                line_images = preprocessor.segment_lines(image_input)
                if line_images:
                    result = self.engine.recognize_lines(line_images)
                else:
                    pil_img = preprocessor.preprocess_to_pil(image_input)
                    result  = self.engine.recognize(pil_img)
            else:
                pil_img = preprocessor.preprocess_to_pil(image_input)
                result  = self.engine.recognize(pil_img)

            logger.info("OCR done. Engine=%s Conf=%.3f Preview: %s",
                        result.engine, result.confidence, result.text[:80])
            return result

        except Exception as e:
            logger.error("TrOCR failed: %s — returning empty result.", e)
            return OCRResult(text="", confidence=0.0, engine="trocr-failed")

    def extract_from_pdf(self, pdf_path: str) -> List[OCRResult]:
        """Process each page of a PDF and return one OCRResult per page."""
        from pdf2image import convert_from_path
        pdf_path = str(Path(pdf_path).resolve())
        pages    = convert_from_path(pdf_path, dpi=200)
        results  = []
        for i, page_img in enumerate(pages):
            logger.info("PDF page %d/%d", i + 1, len(pages))
            results.append(self.extract_text(page_img, use_line_segmentation=False))
        return results

    def _extract_pdf_as_single(self, pdf_path: str) -> OCRResult:
        page_results = self.extract_from_pdf(pdf_path)
        if not page_results:
            return OCRResult(text="", confidence=0.0, engine="trocr-pdf")
        combined_text = "\n".join(r.text for r in page_results if r.text)
        avg_conf      = sum(r.confidence for r in page_results) / len(page_results)
        return OCRResult(
            text=combined_text,
            confidence=round(avg_conf, 4),
            engine="trocr-pdf",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Post-OCR text corrections
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