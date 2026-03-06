"""
backend/preprocessing.py
Image preprocessing pipeline for handwritten answer sheets.

Steps:
  1. Load image (path, bytes, or PIL.Image)
  2. Convert to grayscale
  3. Denoise (fastNlMeansDenoising)
  4. Deskew
  5. CLAHE contrast enhancement
  6. Adaptive thresholding (Otsu)
  7. Return cleaned PIL image ready for OCR
"""

from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from PIL import Image


ImageInput = Union[str, Path, bytes, Image.Image, np.ndarray]


# ── Public API ────────────────────────────────────────────────────────────────
def preprocess_image(image: ImageInput) -> Image.Image:
    """Full preprocessing pipeline. Returns a clean PIL image."""
    arr = _to_numpy(image)
    arr = _to_grayscale(arr)
    arr = _denoise(arr)
    arr = _deskew(arr)
    arr = _enhance_contrast(arr)
    arr = _threshold(arr)
    return Image.fromarray(arr)


def preprocess_to_bytes(image: ImageInput, fmt: str = "PNG") -> bytes:
    """Preprocess and return as raw bytes."""
    pil_img = preprocess_image(image)
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return buf.getvalue()


# ── Internal helpers ──────────────────────────────────────────────────────────
def _to_numpy(image: ImageInput) -> np.ndarray:
    if isinstance(image, np.ndarray):
        return image.copy()
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))
    if isinstance(image, bytes):
        arr = np.frombuffer(image, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    # str / Path
    bgr = cv2.imread(str(image), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read image: {image}")
    return bgr


def _to_grayscale(arr: np.ndarray) -> np.ndarray:
    if len(arr.shape) == 2:
        return arr
    return cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)


def _denoise(arr: np.ndarray) -> np.ndarray:
    """Remove salt-and-pepper noise while preserving strokes."""
    return cv2.fastNlMeansDenoising(arr, h=15, templateWindowSize=7, searchWindowSize=21)


def _deskew(arr: np.ndarray) -> np.ndarray:
    """Rotate image to correct skew detected via Hough transform."""
    _, binary = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(binary > 0))
    if coords.shape[0] < 10:
        return arr
    angle = cv2.minAreaRect(coords)[-1]
    # minAreaRect angle is in [-90, 0); convert to actual rotation
    if angle < -45:
        angle = 90 + angle
    (h, w) = arr.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(arr, M, (w, h), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    return rotated


def _enhance_contrast(arr: np.ndarray) -> np.ndarray:
    """CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(arr)


def _threshold(arr: np.ndarray) -> np.ndarray:
    """Adaptive Gaussian thresholding for clean binary output."""
    return cv2.adaptiveThreshold(
        arr, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=2,
    )
