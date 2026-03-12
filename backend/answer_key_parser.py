"""
answer_key_parser.py  —  IntelliGrade-H
========================================
Parses a teacher's answer key PDF into {question_number: answer_text}.

Pipeline:
  1. Text scraping   — pdfplumber → PyMuPDF plain → PyMuPDF dict spans
  2. Tesseract OCR   — fast, accurate for typed/printed scanned PDFs (NEW)
  3. PaddleOCR       — better for complex layouts (fallback)
  4. TrOCR           — last resort only (handwritten PDFs)
  5. Groq LLM        — extracts structured Q → answer mapping from OCR text
  6. Rule-based      — regex fallback if LLM fails

Key fixes (v4):
  - _LLM_PROMPT is now properly defined (was missing, causing NameError)
  - Tesseract added as the FIRST OCR fallback (fast, typed-text friendly)
  - TrOCR is now last resort only — avoids slow line-by-line on typed PDFs
  - PaddleOCR added as second OCR fallback before TrOCR
  - OCR timeout protection: each engine is guarded so one slow engine
    can't crash the whole pipeline
  - Part-A / Part-B aware LLM prompt
  - Rule-based regex parser for when LLM fails

Usage:
    from backend.answer_key_parser import parse_answer_key

    answers = parse_answer_key("uploads/anskey_abc.pdf")
    # {1: "A relation is a set of tuples ...", 8: "Normalization is ..."}
"""

from __future__ import annotations

import re
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# LLM Prompt (was missing in previous version — caused NameError)
# ─────────────────────────────────────────────────────────

_LLM_PROMPT = """You are an academic answer key parser. Extract all question-answer pairs from the text below.

The answer key may have:
- PART – A (short answers, 2 marks each, questions 1–7)
- PART – B (long answers, 12 marks each, questions 8–13, with OR alternatives)
- Answers numbered as "1.", "1)", "Q1:", "Answer 1:", etc.
- Headings like "1. Define Normalization" followed by the answer text

RULES:
- Extract EVERY question number and its complete model answer
- For OR questions (e.g. Q8 OR Q9), extract both separately
- Include all sub-points, SQL code, tables, and diagrams described in text
- Do NOT truncate long answers — capture them fully
- question_number must be an integer

RAW TEXT:
{raw_text}

Respond ONLY with valid JSON (no markdown fences):
{{
  "answers": [
    {{"question_number": 1, "answer_text": "Full model answer here..."}},
    {{"question_number": 2, "answer_text": "Full model answer here..."}}
  ]
}}"""


# ─────────────────────────────────────────────────────────
# PDF text scraping  (3 strategies, typed PDFs)
# ─────────────────────────────────────────────────────────

def _scrape_pdf_text(pdf_path: str) -> str:
    """
    Extract all selectable text from a PDF using three escalating strategies.
    Returns empty string if the PDF has no selectable text layer (scanned).
    """
    # Strategy 1: pdfplumber
    try:
        import pdfplumber
        parts: list[str] = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                t = page.extract_text(x_tolerance=3, y_tolerance=3)
                if t:
                    parts.append(t)
        text = "\n\n".join(parts).strip()
        if len(text) > 50:
            logger.info("Answer key scrape (pdfplumber): %d chars", len(text))
            return text
    except ImportError:
        pass
    except Exception as e:
        logger.warning("pdfplumber failed: %s", e)

    # Strategy 2: PyMuPDF plain text
    try:
        import fitz
        doc   = fitz.open(pdf_path)
        parts = [page.get_text("text") for page in doc]
        doc.close()
        text  = "\n\n".join(parts).strip()
        if len(text) > 50:
            logger.info("Answer key scrape (PyMuPDF plain): %d chars", len(text))
            return text
    except ImportError:
        pass
    except Exception as e:
        logger.warning("PyMuPDF plain failed: %s", e)

    # Strategy 3: PyMuPDF dict span reconstruction
    try:
        import fitz
        doc   = fitz.open(pdf_path)
        lines: list[str] = []
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
            logger.info("Answer key scrape (PyMuPDF dict spans): %d chars", len(text))
            return text
    except Exception as e:
        logger.warning("PyMuPDF dict failed: %s", e)

    return ""


# ─────────────────────────────────────────────────────────
# PDF → PIL page images helper
# ─────────────────────────────────────────────────────────

def _pdf_to_images(pdf_path: str, dpi: int = 300):
    """Render each PDF page as a PIL Image. Returns list of PIL Images."""
    try:
        from pdf2image import convert_from_path
        images = convert_from_path(pdf_path, dpi=dpi)
        logger.info("pdf2image rendered %d pages at %d DPI", len(images), dpi)
        return images
    except ImportError:
        logger.warning("pdf2image not installed — trying PyMuPDF render")
    except Exception as e:
        logger.warning("pdf2image failed: %s — trying PyMuPDF", e)

    # Fallback: PyMuPDF rendering
    try:
        import fitz
        from PIL import Image
        doc    = fitz.open(pdf_path)
        images = []
        zoom   = dpi / 72.0
        mat    = fitz.Matrix(zoom, zoom)
        for page in doc:
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
        logger.info("PyMuPDF rendered %d pages at %d DPI", len(images), dpi)
        return images
    except Exception as e:
        logger.error("PDF render failed completely: %s", e)
        return []


# ─────────────────────────────────────────────────────────
# OCR Strategy 1: Tesseract (fast, great for typed/printed text)
# ─────────────────────────────────────────────────────────

def _ocr_with_tesseract(pdf_path: str, dpi: int = 300) -> str:
    """
    Use Tesseract OCR to extract text from a scanned answer key PDF.
    Best choice for typed/printed answer keys — fast (~3-5s for 10 pages).
    """
    try:
        import pytesseract
    except ImportError:
        logger.warning("pytesseract not installed. Run: pip install pytesseract")
        return ""

    try:
        import cv2
        import numpy as np
    except ImportError:
        logger.warning("OpenCV not installed — Tesseract will run without preprocessing")
        cv2 = None

    logger.info("Running Tesseract OCR on answer key: %s", pdf_path)

    images = _pdf_to_images(pdf_path, dpi=dpi)
    if not images:
        logger.error("Could not render PDF pages for Tesseract")
        return ""

    page_texts = []
    for i, img in enumerate(images):
        logger.info("Tesseract OCR page %d/%d", i + 1, len(images))
        try:
            # Preprocess for better accuracy
            if cv2 is not None:
                import numpy as np
                arr  = np.array(img.convert("RGB"))
                gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

                # Upscale if too small
                h, w = gray.shape
                if w < 1200:
                    scale = 1200 / w
                    gray  = cv2.resize(gray, (int(w * scale), int(h * scale)),
                                       interpolation=cv2.INTER_CUBIC)

                # Denoise
                gray = cv2.bilateralFilter(gray, 9, 75, 75)

                # CLAHE contrast enhancement
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                gray  = clahe.apply(gray)

                # Adaptive threshold
                binary = cv2.adaptiveThreshold(
                    gray, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 11, 2,
                )
                from PIL import Image as PILImage
                img = PILImage.fromarray(binary)

            # Run Tesseract — psm 4 works well for typed answer keys with mixed layouts
            text = pytesseract.image_to_string(
                img,
                config="--oem 3 --psm 4 -l eng",
            ).strip()

            if text:
                page_texts.append(text)
                logger.info("Tesseract page %d: %d chars", i + 1, len(text))
            else:
                logger.warning("Tesseract returned empty text for page %d", i + 1)

        except Exception as e:
            logger.warning("Tesseract failed on page %d: %s", i + 1, e)

    full_text = "\n\n".join(page_texts)
    logger.info("Tesseract OCR complete: %d chars from %d pages",
                len(full_text), len(images))
    return full_text


# ─────────────────────────────────────────────────────────
# OCR Strategy 2: PaddleOCR (better for complex layouts)
# ─────────────────────────────────────────────────────────

def _ocr_with_paddle(pdf_path: str, dpi: int = 300) -> str:
    """
    Use PaddleOCR to extract text. Better than Tesseract for complex
    layouts (tables, mixed fonts), but slower to load.
    """
    try:
        from paddleocr import PaddleOCR
    except ImportError:
        logger.warning("PaddleOCR not installed. Run: pip install paddleocr paddlepaddle")
        return ""

    logger.info("Running PaddleOCR on answer key: %s", pdf_path)

    images = _pdf_to_images(pdf_path, dpi=dpi)
    if not images:
        return ""

    try:
        import numpy as np
        ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    except Exception as e:
        logger.error("PaddleOCR init failed: %s", e)
        return ""

    page_texts = []
    for i, img in enumerate(images):
        logger.info("PaddleOCR page %d/%d", i + 1, len(images))
        try:
            import numpy as np
            arr    = np.array(img.convert("RGB"))
            result = ocr.ocr(arr, cls=True)

            if not result or not result[0]:
                continue

            lines = []
            for line in result[0]:
                if line and len(line) >= 2:
                    text_info = line[1]
                    if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                        text = str(text_info[0]).strip()
                        if text:
                            lines.append(text)

            if lines:
                page_texts.append("\n".join(lines))
                logger.info("PaddleOCR page %d: %d lines", i + 1, len(lines))

        except Exception as e:
            logger.warning("PaddleOCR page %d failed: %s", i + 1, e)

    full_text = "\n\n".join(page_texts)
    logger.info("PaddleOCR complete: %d chars from %d pages",
                len(full_text), len(images))
    return full_text


# ─────────────────────────────────────────────────────────
# OCR Strategy 3: TrOCR (last resort — handwriting only)
# ─────────────────────────────────────────────────────────

def _ocr_with_trocr(pdf_path: str) -> str:
    """
    Use TrOCR as the absolute last resort.
    Only suitable for handwritten answer keys — very slow for typed text.
    Delegates to OCRModule hybrid pipeline.
    """
    logger.warning(
        "Using TrOCR (last resort) on answer key — this will be slow: %s", pdf_path
    )
    try:
        from backend.ocr_module import OCRModule
        ocr     = OCRModule()
        results = ocr.extract_from_pdf(pdf_path)
        parts   = [r.text.strip() for r in results if r.text.strip()]
        text    = "\n\n".join(parts)
        if text:
            engine_list = ", ".join(dict.fromkeys(r.engine for r in results))
            logger.info(
                "TrOCR recovered %d chars from answer key (engines: %s)",
                len(text), engine_list,
            )
        else:
            logger.error("TrOCR returned no text. Check scan quality.")
        return text
    except Exception as e:
        logger.error("TrOCR fallback failed: %s", e)
        return ""


# ─────────────────────────────────────────────────────────
# Main OCR dispatcher for answer key PDFs
# ─────────────────────────────────────────────────────────

def _ocr_pdf(pdf_path: str) -> str:
    """
    Run OCR on a scanned answer key PDF.

    Engine priority:
      1. Tesseract  — fast, accurate for typed/printed text (~5s for 10 pages)
      2. PaddleOCR  — better for tables/complex layouts (~15s)
      3. TrOCR      — last resort for truly handwritten keys (~5 min)

    Stops at the first engine that returns meaningful text (> 100 chars).
    """
    logger.warning(
        "Answer key has no selectable text — running OCR pipeline: %s", pdf_path
    )

    # 1. Try Tesseract first (fastest for typed PDFs)
    text = _ocr_with_tesseract(pdf_path, dpi=300)
    if len(text.strip()) > 100:
        logger.info("✅ Tesseract OCR succeeded: %d chars", len(text))
        return text
    logger.warning("Tesseract returned insufficient text (%d chars), trying PaddleOCR…",
                   len(text.strip()))

    # 2. Try PaddleOCR
    text = _ocr_with_paddle(pdf_path, dpi=300)
    if len(text.strip()) > 100:
        logger.info("✅ PaddleOCR succeeded: %d chars", len(text))
        return text
    logger.warning("PaddleOCR returned insufficient text (%d chars), trying TrOCR…",
                   len(text.strip()))

    # 3. Last resort: TrOCR (only for truly handwritten answer keys)
    text = _ocr_with_trocr(pdf_path)
    return text


# ─────────────────────────────────────────────────────────
# LLM extraction
# ─────────────────────────────────────────────────────────

def _parse_llm_answer_json(raw_text: str) -> dict[int, str]:
    """Parse LLM JSON response into {question_number: answer_text}."""
    cleaned = re.sub(r"```json\s*|```\s*", "", raw_text).strip()
    # Multi-pass JSON recovery
    for attempt in range(3):
        try:
            if attempt == 0:
                data = json.loads(cleaned)
            elif attempt == 1:
                m = re.search(r"\{.*\}", cleaned, re.DOTALL)
                data = json.loads(m.group()) if m else {}
            else:
                # Character-level: fix embedded newlines inside JSON strings
                fixed = []
                in_string = False
                prev = ""
                for ch in cleaned:
                    if ch == '"' and prev != "\\":
                        in_string = not in_string
                    if in_string and ch == "\n":
                        fixed.append("\\n")
                    elif in_string and ch == "\r":
                        fixed.append("\\r")
                    elif in_string and ch == "\t":
                        fixed.append("\\t")
                    else:
                        fixed.append(ch)
                    prev = ch
                data = json.loads("".join(fixed))
            break
        except Exception:
            data = {}

    answers: dict[int, str] = {}

    # Format 1: {"answers": [{"question_number": N, "answer_text": "..."}]}
    if isinstance(data.get("answers"), list):
        for item in data["answers"]:
            qn = item.get("question_number")
            at = item.get("answer_text", "").strip()
            if qn and at:
                try:
                    answers[int(qn)] = at
                except (ValueError, TypeError):
                    pass

    # Format 2: {"1": "answer", "2": "answer"} — direct dict
    elif isinstance(data, dict):
        for k, v in data.items():
            try:
                answers[int(k)] = str(v).strip()
            except (ValueError, TypeError):
                pass

    return answers


def _extract_with_llm(raw_text: str) -> dict[int, str]:
    """
    Send answer key text to LLM for structured extraction.
    Chunks large texts (>8000 chars) into overlapping segments to avoid
    truncation — each chunk is processed separately and results merged.
    Full answer text is preserved (no truncation of long Part-B answers).
    """
    from backend.llm_provider import get_llm_client

    llm     = get_llm_client()
    CHUNK   = 8000    # chars per chunk (stays well within token budget)
    OVERLAP = 500     # char overlap to avoid cutting mid-answer

    # Split into chunks only if text is large
    if len(raw_text) <= CHUNK:
        chunks = [raw_text]
    else:
        chunks = []
        start  = 0
        while start < len(raw_text):
            end = min(start + CHUNK, len(raw_text))
            chunks.append(raw_text[start:end])
            if end == len(raw_text):
                break
            start = end - OVERLAP

    logger.info(
        "Answer key LLM extraction: %d chars → %d chunk(s)",
        len(raw_text), len(chunks),
    )

    all_answers: dict[int, str] = {}

    for i, chunk in enumerate(chunks, 1):
        prompt = _LLM_PROMPT.format(raw_text=chunk)
        try:
            response = llm.generate(prompt, max_tokens=6000)
            chunk_answers = _parse_llm_answer_json(response.text)
            # Later chunks can overwrite earlier ones for the same question
            # only if the new answer is longer (more complete)
            for qn, ans in chunk_answers.items():
                existing = all_answers.get(qn, "")
                if len(ans) > len(existing):
                    all_answers[qn] = ans
            logger.info(
                "Answer key chunk %d/%d: %d answers extracted",
                i, len(chunks), len(chunk_answers),
            )
        except Exception as e:
            logger.error("Answer key LLM chunk %d/%d failed: %s", i, len(chunks), e)

    logger.info(
        "Answer key LLM extraction complete: %d total answers for questions %s",
        len(all_answers), sorted(all_answers.keys()),
    )
    return all_answers


# ─────────────────────────────────────────────────────────
# Rule-based fallback parser
# ─────────────────────────────────────────────────────────

def _extract_rule_based(raw_text: str) -> dict[int, str]:
    """
    Regex-based answer key parser. Handles common formats:

    Format A (numbered list):
        1. Answer text here...
        2. Another answer...

    Format B (Q: style):
        Q1: Answer text here...
        Answer 1: Answer text...

    Format C (Part headers + numbered):
        PART – A
        1. Define normalization. Ans: ...
        2. What is a relation? A relation is...

        PART – B
        8. Explain the types of joins.
           Joins are used to combine rows from two or more tables...

    Format D (Answer: keyword):
        1) What is DBMS?
        Answer: A DBMS (Database Management System) is...
    """
    answers: dict[int, str] = {}
    lines = raw_text.splitlines()

    q_patterns = [
        # "1." or "1)" or "1:" or "1," (comma = Tesseract OCR misread of period) at start of line
        re.compile(r"^\s*(\d{1,2})\s*[.):\],]\s+(.+)"),
        # "Q1." or "Q.1" or "Q 1"
        re.compile(r"^\s*Q\.?\s*(\d{1,2})\s*[.):]?\s*(.+)", re.I),
        # "Answer 1:" or "Ans 1:"
        re.compile(r"^\s*(?:Answer|Ans)\.?\s*(\d{1,2})\s*[.):]\s*(.+)", re.I),
    ]

    ans_continuation = re.compile(
        r"^\s*(?:Answer|Ans|Sol(?:ution)?)\s*[.:]\s*(.+)", re.I
    )

    current_q   = None
    current_ans: list[str] = []

    def _flush():
        if current_q is not None and current_ans:
            text = " ".join(current_ans).strip()
            if len(text) > 3:
                if current_q in answers:
                    answers[current_q] += "\n" + text
                else:
                    answers[current_q] = text

    for i, line in enumerate(lines):
        stripped = line.strip()

        if not stripped:
            continue

        # Part headers — skip
        if re.search(r"^\s*PART\s*[-–—]?\s*[A-Z]\b", stripped, re.I):
            continue

        # SET headers — skip
        if re.search(r"^\s*SET\s*[-–—]?\s*[A-Z]\b", stripped, re.I):
            continue

        # "OR" separator — skip
        if re.match(r"^\s*\(?OR\)?\s*$", stripped, re.I):
            continue

        # Try each question-start pattern
        matched = False
        for pat in q_patterns:
            m = pat.match(stripped)
            if m:
                _flush()
                current_q   = int(m.group(1))
                first_text  = m.group(2).strip()

                # If captured text ends with "?" it's likely just the question text
                # Look ahead for the actual answer
                if first_text.endswith("?") and len(first_text) < 200:
                    next_text = ""
                    for j in range(i + 1, min(i + 5, len(lines))):
                        nxt = lines[j].strip()
                        if nxt:
                            cont = ans_continuation.match(nxt)
                            if cont:
                                next_text = cont.group(1)
                            elif not any(p.match(nxt) for p in q_patterns):
                                next_text = nxt
                            break
                    current_ans = [next_text] if next_text else []
                else:
                    cont = ans_continuation.match(first_text)
                    current_ans = [cont.group(1) if cont else first_text]

                matched = True
                break

        if matched:
            continue

        # Continuation of current answer
        if current_q is not None:
            looks_like_new_q = any(p.match(stripped) for p in q_patterns)
            if not looks_like_new_q:
                cont = ans_continuation.match(stripped)
                current_ans.append(cont.group(1) if cont else stripped)

    _flush()

    logger.info("Rule-based answer key extraction: %d answers", len(answers))
    return answers


# ─────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────

def parse_answer_key(pdf_path: str) -> dict[int, str]:
    """
    Parse a teacher's answer key PDF into {question_number: answer_text}.

    Pipeline:
      1. pdfplumber / PyMuPDF text scraping  (instant for typed PDFs)
      2. Tesseract OCR                        (fast for printed/scanned)
      3. PaddleOCR                            (better for complex layouts)
      4. TrOCR via OCRModule                  (last resort, handwritten)
      5. Groq LLM structured extraction
      6. Rule-based regex fallback

    Args:
        pdf_path: Absolute or relative path to the answer key PDF.

    Returns:
        Dict mapping question number → model answer text.
        Empty dict if parsing fails completely.
    """
    pdf_path = str(Path(pdf_path).resolve())
    logger.info("Parsing answer key: %s", pdf_path)

    # Step 1 — try fast text scraping (works if PDF has selectable text)
    raw_text = _scrape_pdf_text(pdf_path)

    # Step 2 — OCR fallback if PDF is image-only (scanned)
    if not raw_text.strip():
        raw_text = _ocr_pdf(pdf_path)

    if not raw_text.strip():
        logger.error(
            "Answer key yielded no text after all strategies. "
            "Check the PDF is not password-protected or corrupted: %s", pdf_path
        )
        return {}

    logger.info("Answer key raw text extracted: %d chars", len(raw_text))

    # Step 3 — LLM extracts structured mapping
    answers = _extract_with_llm(raw_text)

    # Step 4 — Rule-based fallback if LLM returned nothing
    if not answers:
        logger.warning("LLM returned no answers — using rule-based fallback")
        answers = _extract_rule_based(raw_text)

    if not answers:
        logger.error(
            "Answer key parsing returned 0 answers after all strategies. "
            "Raw text preview: %.500s", raw_text
        )
    else:
        logger.info(
            "Answer key parsed: %d answers for questions %s",
            len(answers), sorted(answers.keys())
        )

    return answers