"""
IntelliGrade-H - Image Preprocessor (v3 — Handwriting Optimised)
=================================================================
Changes vs v2:
  • _upscale_small() — pages narrower than 1200 px get 2x upscale before OCR
  • _denoise() — adaptive: fastNlMeans for noisy images, bilateral for clean
  • _enhance_contrast() CLAHE clipLimit raised 2.0 → 3.0
  • _threshold() — smart: Otsu first, adaptive Gaussian fallback for faint ink
  • segment_lines() — 4 px padding top/bottom on each line crop for TrOCR
"""

import cv2
import numpy as np
from PIL import Image
import io
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

_MIN_WIDTH_PX = 1200   # pages narrower than this get 2x upscaled


class ImagePreprocessor:
    """
    Preprocesses handwritten answer sheet images for OCR.
    Pipeline: load → upscale-if-small → grayscale → adaptive-denoise
              → deskew → CLAHE(3.0) → smart-threshold
    """

    def __init__(self, target_dpi: int = 300):
        self.target_dpi = target_dpi

    def preprocess(self, image_input) -> np.ndarray:
        img = self._load_image(image_input)
        img = self._upscale_small(img)
        img = self._to_grayscale(img)
        img = self._denoise(img)
        img = self._deskew(img)
        img = self._enhance_contrast(img)
        img = self._threshold(img)
        return img

    def preprocess_to_pil(self, image_input) -> Image.Image:
        return Image.fromarray(self.preprocess(image_input))

    def segment_lines(self, image_input) -> list:
        raw   = self._load_image(image_input)
        raw   = self._upscale_small(raw)
        gray  = self._to_grayscale(raw)
        clean = self._denoise(gray)

        thresh = cv2.adaptiveThreshold(
            clean, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 15, 10,
        )
        h_proj = np.sum(thresh, axis=1)
        lines  = self._find_line_boundaries(h_proj, thresh.shape[0])

        h, _ = clean.shape
        line_images = []
        for (y1, y2) in lines:
            if y2 - y1 < 10:
                continue
            y1p = max(0, y1 - 4)
            y2p = min(h, y2 + 4)
            line_images.append(Image.fromarray(clean[y1p:y2p, :]))

        logger.debug("Segmented %d lines.", len(line_images))
        return line_images

    # ── private ───────────────────────────────────────────

    def _load_image(self, image_input) -> np.ndarray:
        if isinstance(image_input, np.ndarray):
            return image_input
        if isinstance(image_input, Image.Image):
            return np.array(image_input.convert("RGB"))
        if isinstance(image_input, io.BytesIO):
            arr = np.frombuffer(image_input.read(), np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if isinstance(image_input, (bytes, bytearray)):
            arr = np.frombuffer(image_input, np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if isinstance(image_input, (str, Path)):
            img = cv2.imread(str(image_input), cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"Cannot load image: {image_input}")
            return img
        raise TypeError(f"Unsupported image type: {type(image_input)}")

    def _upscale_small(self, img: np.ndarray) -> np.ndarray:
        """2x upscale any page narrower than _MIN_WIDTH_PX (e.g. phone photos)."""
        h, w = img.shape[:2]
        if w < _MIN_WIDTH_PX:
            scale = _MIN_WIDTH_PX / w
            nw, nh = int(w * scale), int(h * scale)
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
            logger.debug("Upscaled %dx%d → %dx%d", w, h, nw, nh)
        return img

    def _to_grayscale(self, img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def _denoise(self, img: np.ndarray) -> np.ndarray:
        """
        Adaptive: fastNlMeans for noisy images (phone photos, std>20),
        bilateral for clean scans (fast, ~3 ms).
        """
        if float(np.std(img)) > 20:
            return cv2.fastNlMeansDenoising(img, h=10, templateWindowSize=7, searchWindowSize=21)
        return cv2.bilateralFilter(img, d=5, sigmaColor=40, sigmaSpace=40)

    def _deskew(self, img: np.ndarray) -> np.ndarray:
        thresh = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5,
        )
        lines = cv2.HoughLinesP(thresh, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
        if lines is None:
            return img
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 != x1:
                a = np.degrees(np.arctan2(y2-y1, x2-x1))
                if -30 < a < 30:
                    angles.append(a)
        if not angles:
            return img
        median_angle = float(np.median(angles))
        if abs(median_angle) < 0.5:
            return img
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), median_angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    def _enhance_contrast(self, img: np.ndarray) -> np.ndarray:
        """CLAHE clipLimit=3.0 — better for faint pencil / low-contrast scans."""
        return cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(img)

    def _threshold(self, img: np.ndarray) -> np.ndarray:
        """
        Smart: Otsu for high-contrast ink; adaptive Gaussian for faint/light ink.
        """
        otsu_t, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if otsu_t < 50:
            binary = cv2.adaptiveThreshold(
                img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15,
            )
        return binary

    def _find_line_boundaries(self, h_proj: np.ndarray, height: int, min_gap: int = 5) -> list:
        in_line, boundaries, start = False, [], 0
        for y in range(height):
            if h_proj[y] > 0:
                if not in_line:
                    in_line, start = True, y
            else:
                if in_line:
                    if y - start >= min_gap:
                        boundaries.append((start, y))
                    in_line = False
        if in_line:
            boundaries.append((start, height))
        return boundaries