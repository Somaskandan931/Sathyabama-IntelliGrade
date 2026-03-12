"""
question_paper_parser.py  —  IntelliGrade-H  (v7 — Dynamic)
=============================================================
Dynamically handles ANY question paper PDF — typed or scanned.

PDF extraction pipeline (dynamic, uses only what is installed):
  1. pdfplumber          — typed PDFs, instant
  2. PyMuPDF plain text  — typed PDFs, fallback
  3. PyMuPDF dict spans  — typed PDFs with complex layout
  4. Tesseract OCR       — scanned PDFs, fast (~5s)
  5. PaddleOCR           — scanned PDFs, better for tables
  6. TrOCR               — last resort for pure handwritten

Structure parsing pipeline:
  1. Groq LLM            — best accuracy, auto-detects all structures
  2. Rule-based regex    — fallback, Sathyabama-layout aware
  3. Minimal extraction  — absolute last resort

Auto-detects:
  • Part structure (Part-A, Part-B, etc.) and marks per part
  • Question numbers, question text, OR alternatives
  • Question type (mcq / short_answer / open_ended / numerical / diagram)
  • Course metadata (code, name, date, batch, semester, total marks)
"""

from __future__ import annotations

import os
import re
import json
import logging
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# Engine capability probe  (runs once at import time)
# ─────────────────────────────────────────────────────────

def _probe_engines() -> dict:
    """Detect which PDF/OCR engines are available on this machine."""
    caps = {
        "pdfplumber": False,
        "pymupdf":    False,
        "tesseract":  False,
        "paddle":     False,
        "trocr":      False,
    }

    try:
        import pdfplumber       # noqa
        caps["pdfplumber"] = True
    except ImportError:
        pass

    try:
        import fitz             # noqa
        caps["pymupdf"] = True
    except ImportError:
        pass

    try:
        import pytesseract
        # Auto-locate Tesseract binary on Windows
        if platform.system() == "Windows":
            win_paths = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
            ]
            cmd = pytesseract.pytesseract.tesseract_cmd
            if not cmd or cmd in ("tesseract", "tesseract.exe"):
                for p in win_paths:
                    if os.path.isfile(p):
                        pytesseract.pytesseract.tesseract_cmd = p
                        break
        pytesseract.get_tesseract_version()
        caps["tesseract"] = True
        logger.info("QuestionPaperParser: Tesseract ✓")
    except Exception:
        logger.info("QuestionPaperParser: Tesseract ✗ — install from https://github.com/UB-Mannheim/tesseract/wiki")

    try:
        from paddleocr import PaddleOCR   # noqa
        caps["paddle"] = True
        logger.info("QuestionPaperParser: PaddleOCR ✓")
    except Exception:
        logger.info("QuestionPaperParser: PaddleOCR ✗ — run: pip install paddleocr paddlepaddle")

    try:
        import torch            # noqa
        import transformers     # noqa
        caps["trocr"] = True
    except ImportError:
        pass

    available = [k for k, v in caps.items() if v]
    logger.info("QuestionPaperParser engines: %s", available)
    return caps


_ENGINE_CAPS: dict = _probe_engines()


# ─────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────

@dataclass
class ParsedPart:
    part_name: str
    marks_per_question: float
    num_questions: int
    instructions: str = ""


@dataclass
class ParsedQuestion:
    question_number: int
    question_text: str
    marks: float
    part_name: str
    question_type: str = "open_ended"
    is_or_option: bool = False


@dataclass
class ParsedExamPaper:
    course_code: str = ""
    course_name: str = ""
    exam_name: str = ""
    total_marks: float = 0.0
    duration_hours: float = 2.0
    date: str = ""
    batch: str = ""
    programme: str = ""
    semester: str = ""
    parts: List[ParsedPart] = field(default_factory=list)
    questions: List[ParsedQuestion] = field(default_factory=list)
    raw_text: str = ""


# ─────────────────────────────────────────────────────────
# PDF text extraction  (typed + scanned, dynamic)
# ─────────────────────────────────────────────────────────

def _pdf_to_images(pdf_path: str, dpi: int = 300) -> list:
    """Render PDF pages as PIL Images using pdf2image or PyMuPDF."""
    try:
        from pdf2image import convert_from_path
        images = convert_from_path(pdf_path, dpi=dpi)
        logger.info("pdf2image: %d pages at %d DPI", len(images), dpi)
        return images
    except ImportError:
        pass
    except Exception as e:
        logger.debug("pdf2image failed: %s", e)

    if _ENGINE_CAPS["pymupdf"]:
        try:
            import fitz
            from PIL import Image
            doc    = fitz.open(pdf_path)
            mat    = fitz.Matrix(dpi / 72.0, dpi / 72.0)
            images = []
            for page in doc:
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append(img)
            logger.info("PyMuPDF render: %d pages at %d DPI", len(images), dpi)
            return images
        except Exception as e:
            logger.error("PDF render failed: %s", e)
    return []


def _extract_pdf_text(pdf_path: str) -> str:
    """
    Extract text from any PDF — typed or scanned.
    Uses only engines confirmed available by _probe_engines().
    """

    # ── Strategy 1: pdfplumber ────────────────────────────
    if _ENGINE_CAPS["pdfplumber"]:
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
                logger.info("PDF text (pdfplumber): %d chars", len(text))
                return text
        except Exception as e:
            logger.debug("pdfplumber failed: %s", e)

    # ── Strategy 2: PyMuPDF plain ─────────────────────────
    if _ENGINE_CAPS["pymupdf"]:
        try:
            import fitz
            doc   = fitz.open(pdf_path)
            parts = [page.get_text("text") for page in doc]
            doc.close()
            text  = "\n\n".join(parts).strip()
            if len(text) > 50:
                logger.info("PDF text (PyMuPDF plain): %d chars", len(text))
                return text
        except Exception as e:
            logger.debug("PyMuPDF plain failed: %s", e)

    # ── Strategy 3: PyMuPDF dict spans ───────────────────
    if _ENGINE_CAPS["pymupdf"]:
        try:
            import fitz
            doc   = fitz.open(pdf_path)
            lines = []
            for page in doc:
                for block in page.get_text("dict")["blocks"]:
                    if block.get("type") != 0:
                        continue
                    for line in block.get("lines", []):
                        line_text = " ".join(
                            span["text"]
                            for span in line.get("spans", [])
                            if span.get("text", "").strip()
                        )
                        if line_text.strip():
                            lines.append(line_text)
            doc.close()
            text = "\n".join(lines).strip()
            if len(text) > 50:
                logger.info("PDF text (PyMuPDF dict spans): %d chars", len(text))
                return text
        except Exception as e:
            logger.debug("PyMuPDF dict failed: %s", e)

    # ── Typed strategies exhausted — PDF is likely scanned ─
    logger.warning("No selectable text found — running OCR on: %s", pdf_path)
    return _ocr_pdf(pdf_path)


def _ocr_pdf(pdf_path: str) -> str:
    """
    OCR fallback for scanned question papers.
    Tries only engines confirmed available by _probe_engines().
    """
    images = _pdf_to_images(pdf_path, dpi=300) if (
        _ENGINE_CAPS["tesseract"] or _ENGINE_CAPS["paddle"]
    ) else []

    # ── Tesseract ─────────────────────────────────────────
    if _ENGINE_CAPS["tesseract"] and images:
        logger.info("OCR: trying Tesseract on %d pages...", len(images))
        try:
            import pytesseract
            try:
                import cv2, numpy as np
                has_cv2 = True
            except ImportError:
                has_cv2 = False

            page_texts = []
            for i, img in enumerate(images):
                logger.info("Tesseract page %d/%d", i + 1, len(images))
                try:
                    proc = img
                    if has_cv2:
                        arr  = np.array(img.convert("RGB"))
                        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
                        h, w = gray.shape
                        if w < 1200:
                            gray = cv2.resize(gray, (int(w * 1200/w), int(h * 1200/w)),
                                              interpolation=cv2.INTER_CUBIC)
                        gray   = cv2.bilateralFilter(gray, 9, 75, 75)
                        clahe  = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                        gray   = clahe.apply(gray)
                        binary = cv2.adaptiveThreshold(
                            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                            cv2.THRESH_BINARY, 11, 2)
                        from PIL import Image as PILImage
                        proc = PILImage.fromarray(binary)
                    t = pytesseract.image_to_string(proc, config="--oem 3 --psm 4 -l eng").strip()
                    if t:
                        page_texts.append(t)
                except Exception as e:
                    logger.debug("Tesseract page %d: %s", i + 1, e)

            text = "\n\n".join(page_texts)
            if len(text.strip()) > 100:
                logger.info("✅ Tesseract: %d chars", len(text))
                return text
            logger.info("Tesseract: only %d chars — trying next engine", len(text.strip()))
        except Exception as e:
            logger.warning("Tesseract OCR failed: %s", e)
    elif not _ENGINE_CAPS["tesseract"]:
        logger.info("Tesseract not available — skipping")

    # ── PaddleOCR ─────────────────────────────────────────
    if _ENGINE_CAPS["paddle"] and images:
        logger.info("OCR: trying PaddleOCR on %d pages...", len(images))
        try:
            import numpy as np
            from paddleocr import PaddleOCR
            ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
            page_texts = []
            for i, img in enumerate(images):
                logger.info("PaddleOCR page %d/%d", i + 1, len(images))
                try:
                    arr    = np.array(img.convert("RGB"))
                    result = ocr.ocr(arr, cls=True)
                    if not result or not result[0]:
                        continue
                    lines = []
                    for line in result[0]:
                        if line and len(line) >= 2:
                            ti = line[1]
                            if isinstance(ti, (list, tuple)) and len(ti) >= 2:
                                t = str(ti[0]).strip()
                                if t:
                                    lines.append(t)
                    if lines:
                        page_texts.append("\n".join(lines))
                except Exception as e:
                    logger.debug("PaddleOCR page %d: %s", i + 1, e)
            text = "\n\n".join(page_texts)
            if len(text.strip()) > 100:
                logger.info("✅ PaddleOCR: %d chars", len(text))
                return text
            logger.info("PaddleOCR: only %d chars — trying next engine", len(text.strip()))
        except Exception as e:
            logger.warning("PaddleOCR failed: %s", e)
    elif not _ENGINE_CAPS["paddle"]:
        logger.info("PaddleOCR not available — skipping")

    # ── TrOCR (last resort) ───────────────────────────────
    if _ENGINE_CAPS["trocr"]:
        logger.warning(
            "Falling back to TrOCR (slow). Install Tesseract for faster OCR: "
            "https://github.com/UB-Mannheim/tesseract/wiki"
        )
        try:
            from backend.ocr_module import OCRModule
            ocr     = OCRModule()
            results = ocr.extract_from_pdf(pdf_path)
            text    = "\n\n".join(r.text for r in results if r.text.strip())
            if text:
                logger.info("TrOCR: %d chars", len(text))
                return text
        except Exception as e:
            logger.error("TrOCR failed: %s", e)

    logger.error(
        "All OCR engines failed or unavailable. "
        "Install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki"
    )
    return ""


# ─────────────────────────────────────────────────────────
# Question type auto-classifier
# ─────────────────────────────────────────────────────────

def _classify_question_type(text: str) -> str:
    t = text.lower()
    if re.search(r"\b[a-e]\s*[\)\.]\s+\w", text) or re.search(r"\(a\)|\(b\)", t):
        return "mcq"
    if re.search(r"\btrue\s+or\s+false\b", t):
        return "true_false"
    if re.search(r"_{2,}|\bfill\s+in\b|\bcomplete\s+the\b", t):
        return "fill_blank"
    if re.search(r"\b(calculate|compute|find\s+the\s+value|solve|determine)\b", t):
        return "numerical"
    if re.search(r"\b(draw|sketch|label|illustrate|diagram|flowchart)\b", t):
        return "diagram"
    if re.search(r"\b(explain|describe|discuss|analyze|analyse|compare|design|evaluate|justify|examine|elaborate)\b", t):
        return "open_ended"
    if re.search(r"\b(define|state|list|name|what\s+is|what\s+are|give|mention|identify|when|where|who)\b", t):
        return "short_answer"
    return "open_ended"


# ─────────────────────────────────────────────────────────
# LLM extraction
# ─────────────────────────────────────────────────────────

_EXTRACTION_PROMPT = """You are an expert academic document parser for a university exam system.

Parse the question paper text below and extract its COMPLETE structure.
This paper may have:
- Part-A (short answer, 2 marks each) and Part-B (long answer, 12-16 marks each)
- "Answer ALL questions" or "Answer ANY X of Y questions"
- OR alternatives (two questions where student answers one)
- Multi-line question text
- Tables with Q.No, Questions, CO, Marks columns

QUESTION PAPER TEXT:
{raw_text}

CRITICAL INSTRUCTIONS:
1. Extract EVERY SINGLE question — do not skip any, including OR alternatives.
2. For OR alternatives: include BOTH questions, set is_or_option=true for the second one.
3. question_number must be the actual integer shown in the paper.
4. marks = the marks shown next to the question OR the marks_per_question for that part.
5. Classify question_type: mcq/true_false/fill_blank/short_answer/open_ended/numerical/diagram
6. Part-A questions: usually short_answer (define, list, state, what is).
7. Part-B questions: usually open_ended (explain, describe, discuss, compare, design).
8. Never invent marks — only use what is written in the paper.
9. total_marks = sum of all non-OR question marks.

Return ONLY valid JSON (no markdown, no extra text):
{{
  "course_code": "<e.g. 11BLH41>",
  "course_name": "<e.g. Database Management Systems>",
  "exam_name": "<e.g. Continuous Assessment Exam-1>",
  "total_marks": <number>,
  "duration_hours": <number, default 2>,
  "date": "<DD.MM.YYYY or empty>",
  "batch": "<e.g. 2024-2028>",
  "programme": "<e.g. B.E-CSE>",
  "semester": "<e.g. IV>",
  "parts": [
    {{
      "part_name": "<e.g. Part-A>",
      "marks_per_question": <number>,
      "num_questions": <number>,
      "instructions": "<e.g. Answer ALL questions>"
    }}
  ],
  "questions": [
    {{
      "question_number": <integer>,
      "question_text": "<complete question text>",
      "marks": <number>,
      "part_name": "<Part-A or Part-B>",
      "question_type": "<mcq|true_false|fill_blank|short_answer|open_ended|numerical|diagram>",
      "is_or_option": <true|false>
    }}
  ]
}}"""


def _parse_with_llm(raw_text: str, llm_client) -> Dict[str, Any]:
    prompt = _EXTRACTION_PROMPT.format(raw_text=raw_text[:12000])
    try:
        response = llm_client.generate(prompt, max_tokens=4000)
        cleaned  = re.sub(r"```json\s*|```\s*", "", response.text).strip()
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        return json.loads(m.group() if m else cleaned)
    except json.JSONDecodeError:
        # Try to salvage truncated JSON
        try:
            raw = re.sub(r"```json\s*|```\s*", "", response.text).strip()
            truncated = re.sub(r',\s*\{[^}]*$', '', raw)
            if not truncated.rstrip().endswith(']}'):
                truncated = truncated.rstrip().rstrip(',') + ']}'
            return json.loads(truncated)
        except Exception:
            return {}
    except Exception as e:
        logger.error("LLM parsing failed: %s", e)
        return {}


# ─────────────────────────────────────────────────────────
# Rule-based fallback parser
# ─────────────────────────────────────────────────────────

def _parse_rule_based(raw_text: str) -> Dict[str, Any]:
    """
    Robust regex fallback. Handles Sathyabama layout and generic formats.
    Detects parts, marks, OR alternatives, and question types automatically.
    """
    result: Dict[str, Any] = {
        "course_code": "", "course_name": "", "exam_name": "",
        "total_marks": 0.0, "duration_hours": 2.0, "date": "",
        "batch": "", "programme": "", "semester": "",
        "parts": [], "questions": [],
    }

    lines = raw_text.splitlines()

    # ── Metadata extraction ───────────────────────────────
    for line in lines:
        s = line.strip()
        if not s:
            continue
        for pattern, key in [
            (r"Course\s+Code\s*[:\-]?\s*\$?([A-Z0-9]+)",              "course_code"),
            (r"Course\s+Name\s*[:\-]?\s*(.+)",                         "course_name"),
            (r"Subject\s*[:\-]?\s*(.+)",                                "course_name"),
            (r"Max\.?\s*Marks?\s*[:\-]?\s*(\d+)",                      "total_marks"),
            (r"Total\s+Marks?\s*[:\-]?\s*(\d+)",                       "total_marks"),
            (r"Time\s*[:\-]?\s*(\d[\d.]*)\s*Hours?",                   "duration_hours"),
            (r"Duration\s*[:\-]?\s*(\d[\d.]*)\s*Hours?",               "duration_hours"),
            (r"Batch\s*[:\-]?\s*(\d{4}[-–]\d{4})",                    "batch"),
            (r"Sem(?:ester)?\s*[:\-]?\s*([IVX\d]+)",                   "semester"),
            (r"Date\s*[:\-]?\s*(\d{1,2}[./]\d{1,2}[./]\d{2,4})",      "date"),
            (r"(CONTINUOUS\s+ASSESSMENT[^\n]*)",                         "exam_name"),
            (r"(CAE[\s\-]\d[^\n]*)",                                     "exam_name"),
        ]:
            m = re.search(pattern, s, re.I)
            if m:
                val = m.group(1).strip()
                if key == "course_name":
                    val = re.split(r"\s+Sem\s*:", val, flags=re.I)[0].strip()
                result[key] = float(val) if key in ("total_marks", "duration_hours") else val

    # ── Part detection ────────────────────────────────────
    parts_found: Dict[str, Dict] = {}
    part_hdr = re.compile(r"Part[-\s–—]*([A-Z])\b", re.I)

    def _mpq_from(text: str) -> Optional[float]:
        """Extract marks-per-question from a marks expression."""
        # (7 x 2 = 14)  → second number = mpq
        m = re.search(r"\(\s*(\d+)\s*[xX×]\s*(\d+)\s*=\s*\d+\s*\)", text)
        if m:
            return float(m.group(2))
        # "2 Marks Each" / "12 MARKS EACH"
        m = re.search(r"(\d+)\s*Marks?\s*Each", text, re.I)
        if m:
            return float(m.group(1))
        return None

    for idx, line in enumerate(lines):
        ph = part_hdr.search(line)
        if not ph:
            continue
        letter = ph.group(1).upper()
        pname  = f"Part-{letter}"
        if pname in parts_found:
            continue

        mpq = _mpq_from(line)
        if mpq is None and idx + 1 < len(lines):
            mpq = _mpq_from(lines[idx + 1])
        if mpq is None:
            mpq = 2.0 if letter == "A" else 12.0

        instr_m = re.search(r"(Answer\s+(?:ALL|ANY[^\n]*)\s*(?:the\s+)?questions?[^\n]*)", line, re.I)
        parts_found[pname] = {
            "part_name":          pname,
            "marks_per_question": mpq,
            "num_questions":      0,
            "instructions":       instr_m.group(1).strip() if instr_m else "",
        }
        logger.info("Detected part: %s = %.0f marks/q", pname, mpq)

    if not parts_found:
        parts_found["Part-A"] = {"part_name": "Part-A", "marks_per_question": 2.0,
                                  "num_questions": 0, "instructions": ""}

    # ── Question extraction ───────────────────────────────
    questions = _extract_sathyabama_questions(raw_text, parts_found)
    if not questions:
        questions = _extract_numbered_questions(raw_text, parts_found)

    # ── Assign parts by question number if missing ────────
    sorted_parts = sorted(parts_found.keys())
    for q in questions:
        if not q.get("part_name") or q["part_name"] not in parts_found:
            q["part_name"] = sorted_parts[0] if q["question_number"] <= 7 else (
                sorted_parts[1] if len(sorted_parts) > 1 else sorted_parts[0]
            )

    # ── Count questions per part ──────────────────────────
    for q in questions:
        p = q.get("part_name", "")
        if p in parts_found:
            parts_found[p]["num_questions"] += 1

    total = float(result.get("total_marks", 0))
    if not total:
        total = sum(q["marks"] for q in questions if not q.get("is_or_option", False))

    result["parts"]       = list(parts_found.values())
    result["questions"]   = questions
    result["total_marks"] = total
    return result


def _extract_sathyabama_questions(text: str, parts_found: Dict) -> list:
    """
    Handles Sathyabama-specific pdfplumber column-flow artefact where
    question text appears ABOVE its question number.
    """
    questions    = []
    lines        = text.splitlines()
    sorted_parts = sorted(parts_found.keys())
    current_part = sorted_parts[0] if sorted_parts else "Part-A"

    _CO_TAG      = re.compile(r"\bC[Oo0][0-9]\b")
    _PART_HDR    = re.compile(r"^Part[-\s–—]*[A-Z]\b", re.I)
    _MARKS_LINE  = re.compile(r"^\(?\s*\d+\s*[xX×]\s*\d+\s*[=]?\s*\d*\s*\)?$")
    _SKIP_LINES  = re.compile(
        r"^(Reg\s*No|Q\.?No|Questions?|SATHYABAMA|INSTITUTE|DEEMED|CONTINUOUS\s+ASSESSMENT"
        r"|Programme|Course\s+Code|Course\s+Name|Batch|Max\.?\s*Marks|Time\s*:|Sem\s*:|Date\s*:"
        r"|Answer\s+ALL|Answer\s+ANY)\b", re.I)
    _QNUM_LINE   = re.compile(r"^\s*(\d{1,2})\s*[.)]?\s*(?:C[Oo]\d)?\s*$")
    _QNUM_INLINE = re.compile(r"^\s*(\d{1,2})\s*[.)]\s+(.+)")
    _QNUM_NOPUNCT= re.compile(r"^\s*(\d{1,2})\s+([A-Z].{5,})")
    _HDR_NOISE   = re.compile(
        r"^(Reg\s*No|SATHYABAMA|INSTITUTE|DEEMED|CONTINUOUS|Programme|"
        r"Course|Batch|Max|Time|Sem|Date|Q\.?No|Questions?)\b", re.I)

    pending_text: list[str] = []
    next_is_or   = False
    current_q_num:  Optional[int] = None
    current_q_text: list[str]     = []
    current_marks:  float         = 2.0

    def _clean(s: str) -> str:
        return re.sub(r"\s{2,}", " ", _CO_TAG.sub("", s)).strip()

    def _flush():
        nonlocal current_q_num, current_q_text, next_is_or, pending_text
        if current_q_num is None:
            return
        full = _clean(" ".join(t for t in current_q_text if t))
        if len(full) < 4:
            current_q_num  = None
            current_q_text = []
            pending_text   = []
            return
        is_or = next_is_or
        next_is_or = False
        questions.append({
            "question_number": current_q_num,
            "question_text":   full,
            "marks":           current_marks,
            "part_name":       current_part,
            "question_type":   _classify_question_type(full),
            "is_or_option":    is_or,
        })
        current_q_num  = None
        current_q_text = []
        pending_text   = []

    for line in lines:
        s = line.strip()
        if not s or _SKIP_LINES.match(s) or _MARKS_LINE.match(s):
            continue

        if _PART_HDR.match(s):
            _flush()
            ph = re.search(r"Part[-\s–—]*([A-Z])\b", s, re.I)
            if ph:
                pname = f"Part-{ph.group(1).upper()}"
                if pname in parts_found:
                    current_part  = pname
                    current_marks = parts_found[pname]["marks_per_question"]
            pending_text = []
            continue

        if re.match(r"^\(?OR\)?\s*$", s, re.I):
            next_is_or = True
            pending_text = []
            continue

        m = _QNUM_LINE.match(s)
        if m:
            q_num = int(m.group(1))
            if 1 <= q_num <= 30:
                _flush()
                current_q_num  = q_num
                current_marks  = parts_found.get(current_part, {}).get("marks_per_question", 2.0)
                current_q_text = [t for t in pending_text if len(t) > 8 and not _HDR_NOISE.match(t)]
                pending_text   = []
                continue

        m = _QNUM_INLINE.match(s)
        if m:
            q_num, q_text = int(m.group(1)), _clean(m.group(2))
            if 1 <= q_num <= 30 and len(q_text) > 2:
                _flush()
                current_q_num  = q_num
                current_marks  = parts_found.get(current_part, {}).get("marks_per_question", 2.0)
                valid_pending  = [t for t in pending_text if len(t) > 8 and not _HDR_NOISE.match(t)]
                current_q_text = valid_pending + ([q_text] if q_text else [])
                pending_text   = []
                continue

        m = _QNUM_NOPUNCT.match(s)
        if m:
            q_num, q_text = int(m.group(1)), _clean(m.group(2))
            if 1 <= q_num <= 30 and len(q_text) > 5:
                _flush()
                current_q_num  = q_num
                current_marks  = parts_found.get(current_part, {}).get("marks_per_question", 2.0)
                current_q_text = [q_text]
                pending_text   = []
                continue

        cleaned = _clean(s)
        if not cleaned:
            continue
        if current_q_num is not None:
            current_q_text.append(cleaned)
        elif not _HDR_NOISE.match(s):
            pending_text.append(cleaned)

    _flush()
    return questions


def _extract_numbered_questions(text: str, parts_found: Dict) -> list:
    """Generic numbered-list fallback for any layout."""
    questions    = []
    lines        = text.splitlines()
    sorted_parts = sorted(parts_found.keys())
    current_part = sorted_parts[0] if sorted_parts else "Part-A"

    part_lines: Dict[int, str] = {}
    for i, line in enumerate(lines):
        for pname in sorted_parts:
            letter = pname.split("-")[-1]
            if re.search(rf"\bPART\s*[-–]?\s*{letter}\b", line, re.I):
                part_lines[i] = pname

    q_start = re.compile(r"^\s*(?:Q\.?\s*)?(\d{1,2})\s*[.):\]]\s+(.+)", re.I)
    q_paren = re.compile(r"^\s*\((\d{1,2})\)\s*\.?\s+(.+)")

    current_q_num:  Optional[int] = None
    current_q_text: list[str]     = []
    current_marks:  float         = 0.0

    def _flush():
        nonlocal current_q_num, current_q_text
        if current_q_num is None:
            return
        full = " ".join(current_q_text).strip()
        if len(full) >= 5:
            is_or = bool(re.search(r"\(OR\)|\bOR\b", full, re.I) and
                         questions and questions[-1]["question_number"] != current_q_num)
            questions.append({
                "question_number": current_q_num,
                "question_text":   re.sub(r"\s*\(OR\)\s*", "", full).strip(),
                "marks":           current_marks,
                "part_name":       current_part,
                "question_type":   _classify_question_type(full),
                "is_or_option":    is_or,
            })
        current_q_num  = None
        current_q_text = []

    for i, line in enumerate(lines):
        if i in part_lines:
            _flush()
            current_part = part_lines[i]

        s = line.strip()
        if not s:
            continue

        if re.match(r"^\s*\(?OR\)?\s*$", s, re.I):
            if questions:
                questions[-1]["_next_is_or"] = True
            continue

        m = q_start.match(s) or q_paren.match(s)
        if m:
            q_num  = int(m.group(1))
            q_text = m.group(2).strip()
            if q_num > 50:
                continue
            _flush()
            current_q_num  = q_num
            current_q_text = [q_text]
            current_marks  = parts_found.get(current_part, {}).get("marks_per_question", 2.0)
            mi = re.search(r"\((\d+)\s*[Mm]arks?\)?\s*$", s)
            if mi:
                current_marks = float(mi.group(1))
        elif current_q_num is not None:
            if not re.search(r"^\s*PART\s*[-–]\s*[A-Z]", s, re.I):
                current_q_text.append(s)

    _flush()

    for idx, q in enumerate(questions):
        if q.pop("_next_is_or", False) and idx + 1 < len(questions):
            questions[idx + 1]["is_or_option"] = True

    return questions


def _minimal_question_extraction(raw_text: str) -> Dict[str, Any]:
    """Absolute last resort — grab any numbered items."""
    logger.warning("Running minimal question extraction as last resort")
    questions = []
    for m in re.finditer(r"(?:^|\n)\s*(\d{1,2})\s*[.)]\s+([A-Z][^\n]{10,})", raw_text):
        q_num  = int(m.group(1))
        q_text = m.group(2).strip()
        if 1 <= q_num <= 30 and len(q_text) > 10:
            questions.append({
                "question_number": q_num,
                "question_text":   q_text,
                "marks":           2.0,
                "part_name":       "Part-A",
                "question_type":   _classify_question_type(q_text),
                "is_or_option":    False,
            })
    return {
        "course_code": "", "course_name": "", "exam_name": "",
        "total_marks": 0.0, "duration_hours": 2.0, "date": "",
        "batch": "", "programme": "", "semester": "",
        "parts": [{"part_name": "Part-A", "marks_per_question": 2.0,
                   "num_questions": len(questions), "instructions": ""}],
        "questions": questions,
    }


# ─────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────

class QuestionPaperParser:
    """
    Parse any question paper PDF — typed or scanned — into a structured
    ParsedExamPaper. Auto-detects parts, marks, question types, and OR
    alternatives. Uses only OCR engines that are actually installed.

    Usage:
        parser = QuestionPaperParser(llm_client=get_llm_client())
        paper  = parser.parse("/path/to/question_paper.pdf")
    """

    def __init__(self, llm_client=None):
        self._llm = llm_client

    def parse(self, pdf_path: str) -> ParsedExamPaper:
        logger.info("Parsing question paper: %s", pdf_path)

        raw_text = _extract_pdf_text(str(Path(pdf_path).resolve()))
        if not raw_text.strip():
            logger.error("Could not extract text from: %s", pdf_path)
            return ParsedExamPaper(raw_text="")

        logger.info("Extracted %d chars from question paper", len(raw_text))
        data: Dict[str, Any] = {}

        # ── LLM extraction ───────────────────────────────
        if self._llm:
            try:
                data = _parse_with_llm(raw_text, self._llm)
                q_count = len(data.get("questions", []))
                logger.info("LLM extracted %d questions", q_count)
                if q_count == 0:
                    logger.warning("LLM returned 0 questions — falling back to rule-based")
                    data = {}
            except Exception as e:
                logger.warning("LLM parsing failed: %s — using rule-based", e)
                data = {}

        # ── Rule-based fallback ──────────────────────────
        if not data or not data.get("questions"):
            logger.info("Using rule-based parser")
            data = _parse_rule_based(raw_text)

        # ── Minimal last-resort ──────────────────────────
        if not data.get("questions"):
            logger.warning("Rule-based found 0 questions — minimal extraction")
            data = _minimal_question_extraction(raw_text)

        # ── Build typed dataclasses ──────────────────────
        parts = [
            ParsedPart(
                part_name          = p.get("part_name", ""),
                marks_per_question = float(p.get("marks_per_question", 0)),
                num_questions      = int(p.get("num_questions", 0)),
                instructions       = p.get("instructions", ""),
            )
            for p in data.get("parts", []) if p.get("part_name")
        ]

        questions = []
        for q in data.get("questions", []):
            q_text = str(q.get("question_text", "")).strip()
            if not q_text or len(q_text) < 3:
                continue
            try:
                q_num  = int(q.get("question_number", 0))
                marks  = float(q.get("marks", 0))
            except (TypeError, ValueError):
                continue
            if q_num <= 0:
                continue
            questions.append(ParsedQuestion(
                question_number = q_num,
                question_text   = q_text,
                marks           = marks,
                part_name       = str(q.get("part_name", "")),
                question_type   = str(q.get("question_type", "open_ended")),
                is_or_option    = bool(q.get("is_or_option", False)),
            ))

        # Deduplicate
        seen: set = set()
        unique_q  = []
        for q in sorted(questions, key=lambda x: x.question_number):
            key = (q.question_number, q.is_or_option)
            if key not in seen:
                seen.add(key)
                unique_q.append(q)

        total = float(data.get("total_marks", 0))
        if not total:
            total = sum(q.marks for q in unique_q if not q.is_or_option)

        paper = ParsedExamPaper(
            course_code    = data.get("course_code", ""),
            course_name    = data.get("course_name", ""),
            exam_name      = data.get("exam_name", ""),
            total_marks    = total,
            duration_hours = float(data.get("duration_hours", 2.0)),
            date           = data.get("date", ""),
            batch          = data.get("batch", ""),
            programme      = data.get("programme", ""),
            semester       = data.get("semester", ""),
            parts          = parts,
            questions      = unique_q,
            raw_text       = raw_text,
        )

        logger.info(
            "Parsed: %s | %s | %d parts | %d questions | %.0f marks",
            paper.course_code or "(unknown)",
            paper.exam_name   or "(unknown)",
            len(paper.parts),
            len(paper.questions),
            paper.total_marks,
        )
        return paper