"""
IntelliGrade-H — Diagram Detector
===================================
Detects whether a student answer image contains diagrams, figures, flowcharts,
circuit diagrams, ER diagrams, graphs, or other drawn structures.

Two-tier detection pipeline:
  Tier 1 — YOLOv8  (primary)
      Uses an object-detection model trained on document regions.
      The base yolov8n model is used for visual object detection; when a
      fine-tuned diagram model is available (YOLO_MODEL_PATH pointing to a
      .pt file), it is used instead.

  Tier 2 — Heuristic CV fallback (used when YOLO is unavailable / fails)
      Analyses the image with OpenCV to find:
        • Contour density  — many contours in a region suggest drawn objects
        • Line density      — Hough lines suggest flow charts / ER diagrams
        • Non-text regions  — areas with high edge density but low OCR text
        • Dark pixel blobs  — large filled shapes / arrows

Usage
-----
    from backend.diagram_detector import DiagramDetector

    detector = DiagramDetector()                        # auto-loads model
    result   = detector.detect("path/to/image.jpg")    # or PIL.Image / bytes

    if result.has_diagram:
        print(f"Found {result.n_diagrams} diagram region(s)")
        for d in result.diagrams:
            print(d.bbox, d.confidence, d.label)

    crops = detector.get_diagram_crops("path/to/image.jpg")
    # crops → list of PIL.Image, one per detected diagram region
"""

from __future__ import annotations

import io
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DiagramRegion:
    """A single detected diagram region."""
    bbox: Tuple[int, int, int, int]   # (x1, y1, x2, y2) in pixels
    confidence: float                  # 0.0 – 1.0
    label: str = "diagram"             # YOLO class name or heuristic label
    area_ratio: float = 0.0            # fraction of total image area


@dataclass
class DiagramDetectionResult:
    """Result returned by DiagramDetector.detect()."""
    has_diagram: bool
    diagrams: List[DiagramRegion] = field(default_factory=list)
    n_diagrams: int = 0
    detector_used: str = "none"        # "yolov8" | "heuristic_fallback"
    image_width: int = 0
    image_height: int = 0
    inference_time_ms: float = 0.0

    # Convenience: bounding boxes only
    @property
    def bboxes(self) -> List[Tuple[int, int, int, int]]:
        return [d.bbox for d in self.diagrams]

    # Largest diagram region (by area), or None
    @property
    def primary_diagram(self) -> Optional[DiagramRegion]:
        if not self.diagrams:
            return None
        return max(self.diagrams, key=lambda d: d.area_ratio)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_image(source: Union[str, Path, bytes, Image.Image]) -> Image.Image:
    """Convert any supported input type into a PIL.Image (RGB)."""
    if isinstance(source, Image.Image):
        return source.convert("RGB")
    if isinstance(source, (bytes, bytearray)):
        return Image.open(io.BytesIO(source)).convert("RGB")
    path = Path(str(source))
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")


def _pil_to_cv2(img: Image.Image):
    """PIL Image → OpenCV BGR ndarray."""
    import cv2
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# ─────────────────────────────────────────────────────────────────────────────
# YOLO detection tier
# ─────────────────────────────────────────────────────────────────────────────

# YOLO class indices / names that we treat as "diagram content"
# For yolov8n (COCO), these classes are geometry-like objects that may
# appear in hand-drawn answers.  A fine-tuned model will have its own labels.
_DIAGRAM_LABELS = {
    # COCO classes that can indicate drawn/structural content
    "clock", "book", "cell phone", "laptop", "tv", "monitor",
    # These are used by document-layout fine-tuned models
    "figure", "diagram", "table", "chart", "graph", "formula",
    "flowchart", "circuit", "er_diagram", "tree", "drawing",
}

def _is_diagram_label(label: str) -> bool:
    label_lc = label.lower()
    return any(d in label_lc for d in _DIAGRAM_LABELS)


class _YOLODetector:
    """Thin wrapper around ultralytics YOLO."""

    def __init__(self, model_path: str, conf_threshold: float = 0.35):
        from ultralytics import YOLO  # lazy import — avoids startup cost
        logger.info("Loading YOLO model: %s", model_path)
        self._model = YOLO(model_path)
        self._conf  = conf_threshold

    def detect(
        self, img: Image.Image
    ) -> Tuple[List[DiagramRegion], str]:
        """
        Returns (list[DiagramRegion], detector_name).
        For the base yolov8n COCO model we use ALL detections as proxy
        for 'interesting visual content'; the caller can filter further.
        For a fine-tuned diagram model every detected class is relevant.
        """
        import time
        w, h = img.size
        total_area = w * h or 1

        t0      = time.perf_counter()
        results = self._model(img, conf=self._conf, verbose=False)
        elapsed = (time.perf_counter() - t0) * 1000

        regions: List[DiagramRegion] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label  = self._model.names.get(cls_id, str(cls_id))
                conf   = float(box.conf[0])
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                area_ratio = ((x2 - x1) * (y2 - y1)) / total_area

                # For base COCO model: only keep if label looks diagram-like
                # For fine-tuned model: keep everything
                if _is_diagram_label(label) or conf >= 0.55:
                    regions.append(DiagramRegion(
                        bbox=(x1, y1, x2, y2),
                        confidence=round(conf, 4),
                        label=label,
                        area_ratio=round(area_ratio, 4),
                    ))

        logger.debug("YOLO found %d diagram region(s) in %.1f ms", len(regions), elapsed)
        return regions, "yolov8"


# ─────────────────────────────────────────────────────────────────────────────
# Heuristic CV fallback tier
# ─────────────────────────────────────────────────────────────────────────────

class _HeuristicDetector:
    """
    OpenCV-based heuristic that detects non-text visual regions.

    Strategy:
      1. Convert to greyscale, apply CLAHE, threshold.
      2. Detect contours — large filled contours indicate drawn shapes.
      3. Hough line transform — many lines indicate ER / flowchart diagrams.
      4. Edge density map — segment the image into grid cells; cells with
         high edge density but relatively low variance (uniformly drawn)
         are flagged as diagram regions.
      5. Merge overlapping candidate boxes with NMS.
    """

    # Minimum fraction of image area for a candidate region
    MIN_AREA_RATIO  = 0.01
    # Maximum fraction (near-full-page crops are likely the whole answer)
    MAX_AREA_RATIO  = 0.90
    # Edge density threshold (mean Canny edges per pixel in a grid cell)
    EDGE_DENSITY_TH = 0.08
    # Minimum number of Hough lines to flag as diagram
    MIN_LINES       = 8

    def detect(
        self, img: Image.Image
    ) -> Tuple[List[DiagramRegion], str]:
        import cv2

        cv_img = _pil_to_cv2(img)
        h, w   = cv_img.shape[:2]
        total  = w * h or 1
        grey   = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

        # ── 1. Adaptive threshold ──────────────────────────────────────────
        clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(grey)
        _, binary = cv2.threshold(
            enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # ── 2. Large contours ─────────────────────────────────────────────
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        candidates: List[Tuple[int, int, int, int]] = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            area_ratio = area / total
            if not (self.MIN_AREA_RATIO < area_ratio < self.MAX_AREA_RATIO):
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = bw / (bh + 1e-6)
            # Skip very thin horizontal strips (likely text lines)
            if aspect > 20 or aspect < 0.05:
                continue
            # Skip small skinny blobs (letter strokes)
            if bw < 30 or bh < 30:
                continue
            candidates.append((x, y, x + bw, y + bh))

        # ── 3. Hough lines — flag image if many straight lines exist ──────
        edges      = cv2.Canny(grey, 50, 150, apertureSize=3)
        lines      = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=60,
            minLineLength=40, maxLineGap=10
        )
        n_lines    = len(lines) if lines is not None else 0
        has_lines  = n_lines >= self.MIN_LINES

        # ── 4. Edge-density grid sweep ────────────────────────────────────
        cell_h, cell_w = max(h // 6, 1), max(w // 4, 1)
        for row in range(6):
            for col in range(4):
                y1 = row * cell_h
                y2 = min(y1 + cell_h, h)
                x1 = col * cell_w
                x2 = min(x1 + cell_w, w)
                cell_edge = edges[y1:y2, x1:x2]
                density   = cell_edge.mean() / 255.0
                if density > self.EDGE_DENSITY_TH:
                    candidates.append((x1, y1, x2, y2))

        # ── 5. NMS merge overlapping boxes ────────────────────────────────
        merged = _nms_boxes(candidates, iou_threshold=0.3)

        regions: List[DiagramRegion] = []
        for (x1, y1, x2, y2) in merged:
            area_ratio = ((x2 - x1) * (y2 - y1)) / total
            if area_ratio < self.MIN_AREA_RATIO:
                continue
            label = "lines_detected" if has_lines else "drawn_region"
            conf  = min(0.5 + area_ratio, 0.85)   # rough confidence proxy
            regions.append(DiagramRegion(
                bbox=(x1, y1, x2, y2),
                confidence=round(conf, 3),
                label=label,
                area_ratio=round(area_ratio, 4),
            ))

        # If Hough found many lines but no large contours, flag the whole image
        if has_lines and not regions:
            regions.append(DiagramRegion(
                bbox=(0, 0, w, h),
                confidence=0.45,
                label="lines_detected",
                area_ratio=1.0,
            ))

        logger.debug(
            "Heuristic found %d region(s) | lines=%d", len(regions), n_lines
        )
        return regions, "heuristic_fallback"


def _nms_boxes(
    boxes: List[Tuple[int, int, int, int]],
    iou_threshold: float = 0.3,
) -> List[Tuple[int, int, int, int]]:
    """Simple greedy NMS to merge heavily overlapping bounding boxes."""
    if not boxes:
        return []

    boxes_arr = np.array(boxes, dtype=float)
    x1 = boxes_arr[:, 0]; y1 = boxes_arr[:, 1]
    x2 = boxes_arr[:, 2]; y2 = boxes_arr[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    # Sort by area descending (keep larger boxes first)
    order = areas.argsort()[::-1]
    keep  = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        ix1 = np.maximum(x1[i], x1[order[1:]])
        iy1 = np.maximum(y1[i], y1[order[1:]])
        ix2 = np.minimum(x2[i], x2[order[1:]])
        iy2 = np.minimum(y2[i], y2[order[1:]])

        inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        order = order[1:][iou < iou_threshold]

    return [tuple(int(v) for v in boxes_arr[i]) for i in keep]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

class DiagramDetector:
    """
    Detects diagram regions in student answer images.

    Parameters
    ----------
    model_path : str | None
        Path to a YOLOv8 .pt weights file.
        Defaults to YOLO_MODEL_PATH from config / env.
        Set to "" or "none" to skip YOLO and always use heuristic fallback.
    conf_threshold : float
        Minimum confidence for YOLO detections (default 0.35).
    min_area_ratio : float
        Minimum fraction of image area for a region to be reported (default 0.01).

    Examples
    --------
    >>> detector = DiagramDetector()
    >>> result = detector.detect("answer.jpg")
    >>> result.has_diagram
    True
    >>> crops = detector.get_diagram_crops("answer.jpg")
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: Optional[float] = None,
        min_area_ratio: float = 0.01,
    ):
        try:
            from backend.config import YOLO_MODEL_PATH, DIAGRAM_CONF_THRESHOLD
        except ImportError:
            import os
            YOLO_MODEL_PATH        = os.getenv("YOLO_MODEL_PATH", "yolov8n.pt")
            DIAGRAM_CONF_THRESHOLD = float(os.getenv("DIAGRAM_CONF_THRESHOLD", "0.35"))

        _model_path = model_path if model_path is not None else YOLO_MODEL_PATH
        _conf       = conf_threshold if conf_threshold is not None else DIAGRAM_CONF_THRESHOLD

        self._min_area = min_area_ratio
        self._yolo: Optional[_YOLODetector] = None
        self._heuristic = _HeuristicDetector()

        # Try to load YOLO; fall back gracefully if unavailable
        if _model_path and _model_path.lower() not in ("", "none", "disabled"):
            try:
                self._yolo = _YOLODetector(_model_path, _conf)
                logger.info("DiagramDetector: YOLO loaded from '%s'", _model_path)
            except Exception as exc:
                logger.warning(
                    "DiagramDetector: YOLO load failed (%s). "
                    "Will use heuristic fallback.", exc
                )
        else:
            logger.info("DiagramDetector: YOLO disabled. Using heuristic only.")

    # ─────────────────────────────────────────────────────
    # Primary detection method
    # ─────────────────────────────────────────────────────

    def detect(
        self,
        source: Union[str, Path, bytes, Image.Image],
    ) -> DiagramDetectionResult:
        """
        Detect diagram regions in the given image.

        Parameters
        ----------
        source : str | Path | bytes | PIL.Image
            Path to image/PDF page, raw bytes, or a PIL Image.

        Returns
        -------
        DiagramDetectionResult
        """
        import time
        t0 = time.perf_counter()

        img = _load_image(source)
        w, h = img.size

        regions: List[DiagramRegion] = []
        detector_used = "none"

        # ── Tier 1: YOLO ──────────────────────────────────────────────────
        if self._yolo is not None:
            try:
                regions, detector_used = self._yolo.detect(img)
            except Exception as exc:
                logger.warning("YOLO inference failed: %s — falling back to heuristic.", exc)
                regions, detector_used = self._heuristic.detect(img)
        else:
            # ── Tier 2: heuristic ─────────────────────────────────────────
            regions, detector_used = self._heuristic.detect(img)

        # Filter by minimum area
        regions = [r for r in regions if r.area_ratio >= self._min_area]

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return DiagramDetectionResult(
            has_diagram      = len(regions) > 0,
            diagrams         = regions,
            n_diagrams       = len(regions),
            detector_used    = detector_used,
            image_width      = w,
            image_height     = h,
            inference_time_ms= round(elapsed_ms, 1),
        )

    # ─────────────────────────────────────────────────────
    # Crop extraction
    # ─────────────────────────────────────────────────────

    def get_diagram_crops(
        self,
        source: Union[str, Path, bytes, Image.Image],
        padding: int = 10,
    ) -> List[Image.Image]:
        """
        Return a list of PIL.Image crops, one per detected diagram region.

        Parameters
        ----------
        source  : image source (same as detect())
        padding : extra pixels to add around each bounding box

        Returns
        -------
        list[PIL.Image]  — empty list if no diagrams detected
        """
        img    = _load_image(source)
        result = self.detect(img)
        w, h   = img.size

        crops = []
        for region in result.diagrams:
            x1, y1, x2, y2 = region.bbox
            # Apply padding, clamped to image bounds
            x1p = max(0, x1 - padding)
            y1p = max(0, y1 - padding)
            x2p = min(w, x2 + padding)
            y2p = min(h, y2 + padding)
            crops.append(img.crop((x1p, y1p, x2p, y2p)))

        return crops

    # ─────────────────────────────────────────────────────
    # Convenience: detect from a list of PIL pages
    # ─────────────────────────────────────────────────────

    def detect_pages(
        self,
        pages: List[Image.Image],
    ) -> List[DiagramDetectionResult]:
        """Run detect() on each page image and return one result per page."""
        return [self.detect(page) for page in pages]