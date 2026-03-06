"""
IntelliGrade-H - Document Layout Detector
==========================================
Uses Detectron2 to detect answer blocks, question number regions,
and structural elements within scanned exam sheets.

This allows the system to:
  1. Separate individual question answers on multi-question sheets.
  2. Locate written regions vs printed instructions/headers.
  3. Pass per-question crops to the OCR engine independently.

Detectron2 model options (in order of speed vs accuracy):
  - faster_rcnn_R_50_FPN_3x   : fast, good accuracy (default)
  - mask_rcnn_R_101_FPN_3x    : slower, best accuracy

Install:
  pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch2.3/index.html

If Detectron2 is unavailable, the module gracefully falls back to a
simple connected-components region detector using OpenCV.
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
class LayoutRegion:
    """A single detected region on the answer sheet."""
    region_type: str           # "answer_block" | "question_number" | "header" | "text"
    bbox: Tuple[int, int, int, int]   # (x1, y1, x2, y2) in pixels
    confidence: float          # 0.0 – 1.0
    question_number: Optional[int] = None   # parsed question number if detected
    crop: Optional[Image.Image] = None      # cropped PIL image of the region


@dataclass
class LayoutResult:
    """Full layout analysis result for one page."""
    page_width: int
    page_height: int
    regions: List[LayoutRegion] = field(default_factory=list)
    answer_blocks: List[LayoutRegion] = field(default_factory=list)
    detector_used: str = "none"    # "detectron2" | "opencv_fallback"

    @property
    def n_answer_blocks(self) -> int:
        return len(self.answer_blocks)


# ─────────────────────────────────────────────────────────────────────────────
# Layout Detector
# ─────────────────────────────────────────────────────────────────────────────

class LayoutDetector:
    """
    Document layout analysis using Detectron2 (with OpenCV fallback).

    Usage:
        detector = LayoutDetector()
        result   = detector.detect(image)                # returns LayoutResult
        crops    = detector.get_answer_crops(image)      # returns list of PIL Images
    """

    # Detectron2 config for document layout — using a LayoutParser-compatible config
    D2_CONFIG_URL = (
        "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config"
    )
    # Label map used by PubLayNet (closest public model for document layout)
    LABEL_MAP = {
        0: "text",
        1: "title",
        2: "list",
        3: "table",
        4: "figure",
    }
    # We map "text" and "list" blocks → answer_block for exam sheets
    ANSWER_BLOCK_LABELS = {"text", "list"}

    def __init__(
        self,
        score_threshold: float = 0.5,
        use_gpu: bool = False,
    ):
        self._score_threshold = score_threshold
        self._use_gpu = use_gpu
        self._model = None
        self._lp_available = False
        self._try_load_layoutparser()

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def detect(self, image_input) -> LayoutResult:
        """
        Detect layout regions in an answer sheet image.

        image_input: PIL Image, numpy array, bytes, or file path.
        Returns:     LayoutResult with all detected regions and answer blocks.
        """
        pil_image = self._to_pil(image_input)
        w, h = pil_image.size

        if self._lp_available and self._model is not None:
            return self._detect_with_layoutparser(pil_image, w, h)
        else:
            return self._detect_with_opencv(pil_image, w, h)

    def get_answer_crops(self, image_input) -> List[Image.Image]:
        """
        Convenience method: detect layout and return cropped PIL images
        of each answer block, ordered top-to-bottom.
        """
        result = self.detect(image_input)
        pil_image = self._to_pil(image_input)

        # Sort answer blocks top-to-bottom (by y1)
        sorted_blocks = sorted(result.answer_blocks, key=lambda r: r.bbox[1])

        crops = []
        for block in sorted_blocks:
            x1, y1, x2, y2 = block.bbox
            # Add a small margin
            margin = 5
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(pil_image.width,  x2 + margin)
            y2 = min(pil_image.height, y2 + margin)
            crops.append(pil_image.crop((x1, y1, x2, y2)))

        return crops if crops else [pil_image]   # fallback: whole page

    # ─────────────────────────────────────────────────────────────────────────
    # Detectron2 / LayoutParser path
    # ─────────────────────────────────────────────────────────────────────────

    def _try_load_layoutparser(self):
        try:
            import layoutparser as lp
            self._model = lp.Detectron2LayoutModel(
                config_path=self.D2_CONFIG_URL,
                label_map=self.LABEL_MAP,
                extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", self._score_threshold],
            )
            self._lp_available = True
            logger.info("LayoutParser + Detectron2 loaded successfully.")
        except Exception as e:
            logger.warning(
                "Detectron2/LayoutParser not available (%s). "
                "Using OpenCV connected-components fallback.", e
            )
            self._lp_available = False

    def _detect_with_layoutparser(self, pil_image: Image.Image, w: int, h: int) -> LayoutResult:
        import layoutparser as lp
        import numpy as np

        img_array = np.array(pil_image)
        lp_layout = self._model.detect(img_array)

        regions       = []
        answer_blocks = []

        for block in lp_layout:
            x1 = int(block.block.x_1)
            y1 = int(block.block.y_1)
            x2 = int(block.block.x_2)
            y2 = int(block.block.y_2)

            label = block.type.lower() if block.type else "text"
            region_type = "answer_block" if label in self.ANSWER_BLOCK_LABELS else label

            r = LayoutRegion(
                region_type=region_type,
                bbox=(x1, y1, x2, y2),
                confidence=float(block.score) if hasattr(block, "score") else 0.9,
                crop=pil_image.crop((x1, y1, x2, y2)),
            )
            regions.append(r)
            if region_type == "answer_block":
                answer_blocks.append(r)

        return LayoutResult(
            page_width=w,
            page_height=h,
            regions=regions,
            answer_blocks=answer_blocks,
            detector_used="detectron2",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # OpenCV fallback path
    # ─────────────────────────────────────────────────────────────────────────

    def _detect_with_opencv(self, pil_image: Image.Image, w: int, h: int) -> LayoutResult:
        """
        Lightweight fallback: uses horizontal projection profiling to find
        text line groups, then groups adjacent lines into answer blocks.
        """
        import cv2

        # Convert to grayscale numpy
        gray = np.array(pil_image.convert("L"))

        # Adaptive threshold
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 15, 10,
        )

        # Dilate horizontally to connect words in lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 3))
        dilated = cv2.dilate(binary, kernel, iterations=2)

        # Find contours of text blocks
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions       = []
        answer_blocks = []

        # Sort by y (top-to-bottom)
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])

        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)

            # Filter noise — skip very small regions
            if bw < 50 or bh < 10:
                continue

            area_ratio = (bw * bh) / (w * h)
            if area_ratio < 0.001:
                continue

            r = LayoutRegion(
                region_type="answer_block",
                bbox=(x, y, x + bw, y + bh),
                confidence=0.7,    # fixed confidence for rule-based
                crop=pil_image.crop((x, y, x + bw, y + bh)),
            )
            regions.append(r)
            answer_blocks.append(r)

        # Merge overlapping / adjacent blocks
        answer_blocks = self._merge_overlapping(answer_blocks)

        logger.debug(
            "OpenCV fallback detected %d answer blocks.", len(answer_blocks)
        )

        return LayoutResult(
            page_width=w,
            page_height=h,
            regions=regions,
            answer_blocks=answer_blocks,
            detector_used="opencv_fallback",
        )

    def _merge_overlapping(
        self, blocks: List[LayoutRegion], gap_threshold: int = 20
    ) -> List[LayoutRegion]:
        """Merge blocks that are vertically close into single answer regions."""
        if not blocks:
            return blocks

        merged = [blocks[0]]
        for current in blocks[1:]:
            last = merged[-1]
            _, last_y1, _, last_y2 = last.bbox
            cur_x1, cur_y1, cur_x2, cur_y2 = current.bbox

            if cur_y1 - last_y2 <= gap_threshold:
                # Merge
                new_x1 = min(last.bbox[0], cur_x1)
                new_y1 = min(last.bbox[1], cur_y1)
                new_x2 = max(last.bbox[2], cur_x2)
                new_y2 = max(last.bbox[3], cur_y2)
                merged[-1] = LayoutRegion(
                    region_type="answer_block",
                    bbox=(new_x1, new_y1, new_x2, new_y2),
                    confidence=max(last.confidence, current.confidence),
                )
            else:
                merged.append(current)

        return merged

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
