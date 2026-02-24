"""
IntelliGrade-H - Image Preprocessing Module
Handles grayscale, denoising, deskew, thresholding before OCR.
"""

import cv2
import numpy as np
from PIL import Image
import io
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    Preprocesses handwritten answer sheet images for OCR.
    Pipeline: grayscale → denoise → deskew → contrast enhance → threshold.
    """

    def __init__(self, target_dpi: int = 300):
        self.target_dpi = target_dpi

    # ─────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────

    def preprocess(self, image_input) -> np.ndarray:
        """
        Accept a file path, bytes, PIL Image, or np.ndarray.
        Returns a preprocessed numpy array (grayscale, binary).
        """
        img = self._load_image(image_input)
        img = self._to_grayscale(img)
        img = self._denoise(img)
        img = self._deskew(img)
        img = self._enhance_contrast(img)
        img = self._threshold(img)
        logger.info("Image preprocessing complete.")
        return img

    def preprocess_to_pil(self, image_input) -> Image.Image:
        """Preprocess and return as PIL Image (used by TrOCR)."""
        arr = self.preprocess(image_input)
        return Image.fromarray(arr)

    def segment_lines(self, image_input) -> list:
        """
        Segment the answer sheet into individual text line images.
        Returns a list of PIL Images, one per line.
        """
        img = self._load_image(image_input)
        gray = self._to_grayscale(img)
        denoised = self._denoise(gray)
        thresh = cv2.adaptiveThreshold(
            denoised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 15, 10
        )

        # horizontal projection to find line boundaries
        h_proj = np.sum(thresh, axis=1)
        lines = self._find_line_boundaries(h_proj, thresh.shape[0])

        line_images = []
        for (y1, y2) in lines:
            if y2 - y1 < 8:
                continue
            line_crop = denoised[y1:y2, :]
            line_pil = Image.fromarray(line_crop)
            line_images.append(line_pil)

        logger.info(f"Segmented {len(line_images)} lines from image.")
        return line_images

    # ─────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────

    def _load_image(self, image_input) -> np.ndarray:
        if isinstance(image_input, np.ndarray):
            return image_input
        if isinstance(image_input, Image.Image):
            return np.array(image_input.convert("RGB"))
        if isinstance(image_input, (bytes, bytearray)):
            arr = np.frombuffer(image_input, np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if isinstance(image_input, (str, Path)):
            img = cv2.imread(str(image_input), cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"Cannot load image: {image_input}")
            return img
        raise TypeError(f"Unsupported image type: {type(image_input)}")

    def _to_grayscale(self, img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def _denoise(self, img: np.ndarray) -> np.ndarray:
        return cv2.fastNlMeansDenoising(img, h=10, templateWindowSize=7, searchWindowSize=21)

    def _deskew(self, img: np.ndarray) -> np.ndarray:
        """Detect and correct skew using Hough line transform."""
        thresh = cv2.adaptiveThreshold(
            img, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 5
        )
        lines = cv2.HoughLinesP(thresh, 1, np.pi / 180, threshold=100,
                                minLineLength=100, maxLineGap=10)
        if lines is None:
            return img

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:
                continue
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if -30 < angle < 30:
                angles.append(angle)

        if not angles:
            return img

        median_angle = np.median(angles)
        if abs(median_angle) < 0.5:
            return img

        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h),
                                 flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)
        return rotated

    def _enhance_contrast(self, img: np.ndarray) -> np.ndarray:
        """CLAHE contrast enhancement."""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(img)

    def _threshold(self, img: np.ndarray) -> np.ndarray:
        """Otsu binarization."""
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def _find_line_boundaries(self, h_proj: np.ndarray, height: int,
                               min_gap: int = 5) -> list:
        """Return list of (y_start, y_end) tuples for text lines."""
        in_line = False
        boundaries = []
        start = 0

        for y in range(height):
            if h_proj[y] > 0:
                if not in_line:
                    in_line = True
                    start = y
            else:
                if in_line:
                    if y - start >= min_gap:
                        boundaries.append((start, y))
                    in_line = False

        if in_line:
            boundaries.append((start, height))

        return boundaries
