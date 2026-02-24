"""
IntelliGrade-H - OCR Module
Supports TrOCR (transformer-based) and Tesseract as fallback.
"""

import os
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    text: str
    confidence: float   # 0.0 – 1.0
    engine: str


# ─────────────────────────────────────────────────────────
# TrOCR Engine
# ─────────────────────────────────────────────────────────

class TrOCREngine:
    """
    Microsoft TrOCR — transformer-based handwriting recognition.
    Model: microsoft/trocr-large-handwritten  (fine-tune on your data)
    """

    MODEL_NAME = "microsoft/trocr-large-handwritten"

    def __init__(self, model_path: Optional[str] = None):
        """
        model_path: path to a locally fine-tuned model directory.
                    Falls back to the HuggingFace hub model if None.
        """
        self._loaded = False
        self._model_path = model_path or self.MODEL_NAME

    def _load(self):
        if self._loaded:
            return
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        import torch

        logger.info(f"Loading TrOCR model from: {self._model_path}")
        self.processor = TrOCRProcessor.from_pretrained(self._model_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(self._model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        self._loaded = True
        logger.info(f"TrOCR loaded on {self.device}")

    def recognize(self, image: Image.Image) -> OCRResult:
        self._load()
        import torch

        # TrOCR expects RGB PIL Image
        if image.mode != "RGB":
            image = image.convert("RGB")

        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                pixel_values,
                output_scores=True,
                return_dict_in_generate=True,
                max_new_tokens=256
            )

        text = self.processor.batch_decode(
            output.sequences, skip_special_tokens=True
        )[0].strip()

        # compute average token confidence from sequence scores
        if output.scores:
            import torch.nn.functional as F
            probs = [F.softmax(s, dim=-1).max().item() for s in output.scores]
            confidence = float(np.mean(probs))
        else:
            confidence = 0.9  # default if scores unavailable

        return OCRResult(text=text, confidence=confidence, engine="trocr")

    def recognize_lines(self, line_images: list) -> OCRResult:
        """
        Recognize a list of line PIL Images and join into one result.
        Used when the preprocessor has segmented lines.
        """
        self._load()
        texts = []
        confidences = []

        for line_img in line_images:
            result = self.recognize(line_img)
            texts.append(result.text)
            confidences.append(result.confidence)

        full_text = " ".join(t for t in texts if t)
        avg_conf = float(np.mean(confidences)) if confidences else 0.0

        return OCRResult(text=full_text, confidence=avg_conf, engine="trocr")

    def fine_tune(self, dataset_path: str, output_dir: str,
                  epochs: int = 5, batch_size: int = 8):
        """
        Fine-tune TrOCR on a local dataset.

        dataset_path must contain:
          images/  (JPG/PNG files)
          labels.txt  (one transcription per line, matching image filenames)

        Example labels.txt:
          img_001.jpg\tThe answer is machine learning.
          img_002.jpg\tNeural networks mimic the brain.
        """
        from transformers import (
            TrOCRProcessor, VisionEncoderDecoderModel,
            Seq2SeqTrainer, Seq2SeqTrainingArguments,
            default_data_collator
        )
        from torch.utils.data import Dataset
        import torch

        self._load()

        class HandwritingDataset(Dataset):
            def __init__(self, data_dir, processor):
                self.data = []
                self.processor = processor
                labels_file = Path(data_dir) / "labels.txt"
                img_dir = Path(data_dir) / "images"

                with open(labels_file, encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split("\t")
                        if len(parts) == 2:
                            img_name, transcription = parts
                            img_path = img_dir / img_name
                            if img_path.exists():
                                self.data.append((str(img_path), transcription))

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                img_path, text = self.data[idx]
                image = Image.open(img_path).convert("RGB")
                encoding = self.processor(image, return_tensors="pt")
                labels = self.processor.tokenizer(
                    text, return_tensors="pt", padding="max_length",
                    max_length=128, truncation=True
                ).input_ids.squeeze()
                labels[labels == self.processor.tokenizer.pad_token_id] = -100
                return {
                    "pixel_values": encoding.pixel_values.squeeze(),
                    "labels": labels
                }

        train_dataset = HandwritingDataset(dataset_path, self.processor)

        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            predict_with_generate=True,
            save_steps=500,
            logging_steps=100,
            fp16=torch.cuda.is_available(),
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=default_data_collator,
        )

        trainer.train()
        trainer.save_model(output_dir)
        self.processor.save_pretrained(output_dir)
        logger.info(f"Fine-tuned model saved to {output_dir}")


# ─────────────────────────────────────────────────────────
# Tesseract Engine (fallback)
# ─────────────────────────────────────────────────────────

class TesseractEngine:
    """Pytesseract wrapper — lightweight fallback OCR."""

    def recognize(self, image: Image.Image) -> OCRResult:
        import pytesseract

        if image.mode != "RGB":
            image = image.convert("RGB")

        data = pytesseract.image_to_data(
            image, output_type=pytesseract.Output.DICT, lang="eng"
        )

        words = []
        confs = []
        for word, conf in zip(data["text"], data["conf"]):
            word = word.strip()
            if word and conf != -1:
                words.append(word)
                confs.append(int(conf))

        text = " ".join(words)
        confidence = (sum(confs) / len(confs) / 100.0) if confs else 0.0

        return OCRResult(text=text, confidence=confidence, engine="tesseract")


# ─────────────────────────────────────────────────────────
# OCR Module (main interface)
# ─────────────────────────────────────────────────────────

class OCRModule:
    """
    High-level OCR module.
    Uses TrOCR by default; falls back to Tesseract on error.
    Integrates with the ImagePreprocessor for line segmentation.
    """

    def __init__(self, engine: str = "trocr",
                 trocr_model_path: Optional[str] = None):
        self.engine_name = engine
        if engine == "trocr":
            self.primary = TrOCREngine(trocr_model_path)
        else:
            self.primary = TesseractEngine()
        self.fallback = TesseractEngine()

    def extract_text(self, image_input, use_line_segmentation: bool = True) -> OCRResult:
        """
        Main entry point. Accepts file path, bytes, PIL Image, or numpy array.
        """
        from backend.preprocessor import ImagePreprocessor
        preprocessor = ImagePreprocessor()

        try:
            if use_line_segmentation and self.engine_name == "trocr":
                line_images = preprocessor.segment_lines(image_input)
                if line_images:
                    result = self.primary.recognize_lines(line_images)
                else:
                    pil_img = preprocessor.preprocess_to_pil(image_input)
                    result = self.primary.recognize(pil_img)
            else:
                pil_img = preprocessor.preprocess_to_pil(image_input)
                result = self.primary.recognize(pil_img)

            logger.info(f"OCR complete. Confidence: {result.confidence:.2f}")
            return result

        except Exception as e:
            logger.warning(f"Primary OCR failed ({e}). Trying Tesseract fallback.")
            try:
                pil_img = preprocessor.preprocess_to_pil(image_input)
                return self.fallback.recognize(pil_img)
            except Exception as e2:
                logger.error(f"Fallback OCR also failed: {e2}")
                return OCRResult(text="", confidence=0.0, engine="failed")

    def extract_from_pdf(self, pdf_path: str) -> list:
        """
        Extract text from each page of a PDF answer sheet.
        Returns a list of OCRResult, one per page.
        """
        from pdf2image import convert_from_path

        pages = convert_from_path(pdf_path, dpi=300)
        results = []
        for i, page_img in enumerate(pages):
            logger.info(f"Processing PDF page {i + 1}/{len(pages)}")
            result = self.extract_text(page_img, use_line_segmentation=True)
            results.append(result)
        return results
