"""
IntelliGrade-H - Diagram Detector
===================================
Uses YOLOv8 to detect diagrams, flowcharts, and visual elements
inside student answer images.

When a diagram is detected, the system:
  1. Crops the diagram region from the page.
  2. Passes it to the OCR engine to extract any text labels.
  3. Routes the extracted text to the diagram-specific LLM prompt.

This enables grading of answers that include labeled diagrams,
architecture drawings, flowcharts, or ER diagrams.

Install:
  pip install ultralytics

If YOLOv8 is unavailable, a simple heuristic detector based on
connected-components ratio is used as fallback.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DiagramRegion:
    """A single detected diagram/visual region."""
    bbox: Tuple[int, int, int, int]     # (x1, y1, x2, y2) in pixels
    confidence: float                   # 0.0 – 1.0
    diagram_type: str = "diagram"       # "diagram" | "flowchart" | "table" | "figure"
    crop: Optional[Image.Image] = None  # cropped image of the diagram


@dataclass
class DiagramDetectionResult:
    """Result of diagram detection for one image."""
    has_diagram: bool
    diagrams: List[DiagramRegion] = field(default_factory=list)
    detector_used: str = "none"    # "yolov8" | "heuristic_fallback"

    @property
    def n_diagrams(self) -> int:
        return len(self.diagrams)


# ─────────────────────────────────────────────────────────────────────────────
# Diagram Detector
# ─────────────────────────────────────────────────────────────────────────────

class DiagramDetector:
    """
    Detects diagrams and figures inside student answer images using YOLOv8.

    Usage:
        detector = DiagramDetector()
        result   = detector.detect(image)           # DiagramDetectionResult
        has_diag = result.has_diagram
        crops    = [d.crop for d in result.diagrams]
    """

    # YOLOv8 model size: "yolov8n" (nano, fastest) → "yolov8x" (extra-large, most accurate)
    # For academic answer sheets, nano or small is usually sufficient.
    DEFAULT_YOLO_MODEL = "yolov8n.pt"

    # COCO classes that count as diagram/figure regions
    DIAGRAM_CLASSES = {
        # Standard COCO classes that resemble diagrams in exam contexts:
        "book",   "laptop", "cell phone",   # electronics/tech diagrams
        # For a fine-tuned model, "diagram", "flowchart", "figure" would be here
    }
    # Minimum area ratio of the image for a region to count as a diagram
    MIN_AREA_RATIO = 0.02

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.35,
    ):
        import os
        self._model_path = model_path or os.getenv("YOLO_MODEL_PATH", self.DEFAULT_YOLO_MODEL)
        self._conf = float(os.getenv("DIAGRAM_CONF_THRESHOLD", str(confidence_threshold)))
        self._model = None
        self._yolo_available = False
        self._try_load_yolo()

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def detect(self, image_input) -> DiagramDetectionResult:
        """
        Detect diagrams in an answer image.

        image_input: PIL Image, numpy array, bytes, or file path.
        Returns:     DiagramDetectionResult.
        """
        pil_image = self._to_pil(image_input)

        if self._yolo_available and self._model is not None:
            return self._detect_with_yolo(pil_image)
        else:
            return self._detect_with_heuristic(pil_image)

    def get_diagram_crops(self, image_input) -> List[Image.Image]:
        """Return only the cropped diagram images, sorted left-to-right, top-to-bottom."""
        result    = self.detect(image_input)
        pil_image = self._to_pil(image_input)

        crops = []
        sorted_diagrams = sorted(result.diagrams, key=lambda d: (d.bbox[1], d.bbox[0]))
        for diag in sorted_diagrams:
            x1, y1, x2, y2 = diag.bbox
            # Small margin
            m = 8
            x1 = max(0, x1 - m)
            y1 = max(0, y1 - m)
            x2 = min(pil_image.width, x2 + m)
            y2 = min(pil_image.height, y2 + m)
            crops.append(pil_image.crop((x1, y1, x2, y2)))
        return crops

    # ─────────────────────────────────────────────────────────────────────────
    # YOLOv8 detection
    # ─────────────────────────────────────────────────────────────────────────

    def _try_load_yolo(self):
        try:
            from ultralytics import YOLO
            self._model = YOLO(self._model_path)
            self._yolo_available = True
            logger.info("YOLOv8 model loaded: %s", self._model_path)
        except Exception as e:
            logger.warning(
                "YOLOv8 not available (%s). Using heuristic diagram detection.", e
            )
            self._yolo_available = False

    def _detect_with_yolo(self, pil_image: Image.Image) -> DiagramDetectionResult:
        img_array = np.array(pil_image)
        results   = self._model(img_array, conf=self._conf, verbose=False)

        diagrams = []
        w, h     = pil_image.size
        min_area = w * h * self.MIN_AREA_RATIO

        for result in results:
            boxes  = result.boxes
            names  = result.names

            if boxes is None:
                continue

            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0])
                label  = names.get(cls_id, "unknown").lower()
                conf   = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                area = (x2 - x1) * (y2 - y1)

                # Accept any region large enough — on exam sheets most large
                # non-text regions are diagrams; for production use a fine-tuned model.
                if area < min_area:
                    continue

                diagram_type = "figure"
                if "flow" in label:
                    diagram_type = "flowchart"
                elif "table" in label:
                    diagram_type = "table"

                diagrams.append(DiagramRegion(
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    diagram_type=diagram_type,
                    crop=pil_image.crop((x1, y1, x2, y2)),
                ))

        logger.debug("YOLOv8 detected %d diagram region(s).", len(diagrams))
        return DiagramDetectionResult(
            has_diagram=len(diagrams) > 0,
            diagrams=diagrams,
            detector_used="yolov8",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Heuristic fallback
    # ─────────────────────────────────────────────────────────────────────────

    def _detect_with_heuristic(self, pil_image: Image.Image) -> DiagramDetectionResult:
        """
        Simple heuristic: regions with low text density and high edge density
        are likely diagrams (boxes, arrows, drawings).
        """
        import cv2

        gray   = np.array(pil_image.convert("L"))
        edges  = cv2.Canny(gray, 50, 150)
        w, h   = pil_image.size

        # Dilate edges to find large connected edge regions
        kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
        dilated = cv2.dilate(edges, kernel, iterations=3)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        diagrams    = []
        min_area    = w * h * self.MIN_AREA_RATIO

        for contour in contours:
            bx, by, bw, bh = cv2.boundingRect(contour)
            area = bw * bh

            if area < min_area:
                continue

            # Heuristic: check that the aspect ratio is not extremely tall
            # (tall thin regions are more likely paragraph text than diagrams)
            aspect = bw / bh if bh > 0 else 0
            if aspect < 0.3:
                continue

            # Check edge density inside this box — diagrams have higher edge density
            roi        = edges[by:by + bh, bx:bx + bw]
            edge_ratio = roi.sum() / (255 * area)
            if edge_ratio < 0.02:
                continue

            diagrams.append(DiagramRegion(
                bbox=(bx, by, bx + bw, by + bh),
                confidence=min(0.9, edge_ratio * 5),
                diagram_type="diagram",
                crop=pil_image.crop((bx, by, bx + bw, by + bh)),
            ))

        # Keep only top-3 largest candidates
        diagrams.sort(key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]), reverse=True)
        diagrams = diagrams[:3]

        logger.debug("Heuristic detected %d potential diagram(s).", len(diagrams))
        return DiagramDetectionResult(
            has_diagram=len(diagrams) > 0,
            diagrams=diagrams,
            detector_used="heuristic_fallback",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Image input handling
    # ─────────────────────────────────────────────────────────────────────────

    def _to_pil(self, image_input) -> Image.Image:
        if isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        if isinstance(image_input, np.ndarray):
            return Image.fromarray(image_input).convert("RGB")
        if isinstance(image_input, (bytes, bytearray)):
            import io
            return Image.open(io.BytesIO(image_input)).convert("RGB")
        if isinstance(image_input, str):
            return Image.open(image_input).convert("RGB")
        raise TypeError(f"Unsupported image type: {type(image_input)}")