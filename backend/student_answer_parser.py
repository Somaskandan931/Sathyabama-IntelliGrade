"""
student_answer_parser.py  —  IntelliGrade-H v4.2
=================================================
Parses a student's handwritten answer booklet PDF into structured per-question answers.

Pipeline:
  1. Extract pages as images from PDF (PyMuPDF → pdf2image fallback)
  2. Parse cover page for student metadata (roll number, register number, set, course)
  3. For each content page: run hybrid OCR (PaddleOCR → Tesseract → TrOCR)
  4. Send all raw OCR text to LLM → segment into {question_number: answer_text} dict
  5. Return ParsedStudentBooklet dataclass

Key fixes (v4.2):
  - LLM segmentation prompt improved for Part-A / Part-B aware parsing
  - Multi-chunk merging now appends rather than overwrites
  - Cover page metadata extraction handles more formats
  - Rule-based fallback added when LLM segmentation returns empty
  - Question number detection handles: "8)", "(8)", "Q8", "Q.8", "Ans 8:", "8."
  - OR alternative detection preserved
  - Graceful handling of completely blank pages (OCR returns nothing)
"""

from __future__ import annotations

import logging
import json
import re
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────
# Dataclasses
# ──────────────────────────────────────────────────────────

@dataclass
class ParsedStudentAnswer:
    question_number: int
    answer_text: str
    raw_ocr_text: str = ""          # pre-LLM OCR dump for this question's pages
    part_name: str = ""             # "Part-A" / "Part-B"
    is_or_option: bool = False      # True if student chose the OR alternative


@dataclass
class ParsedStudentBooklet:
    # ── cover page metadata ──
    roll_number: str = ""
    register_number: str = ""
    set_name: str = ""              # "Set-A", "Set-B"
    course_code: str = ""
    course_name: str = ""
    exam_name: str = ""
    programme: str = ""
    semester: str = ""
    date_of_exam: str = ""
    total_pages: int = 0

    # ── parsed answers ──
    answers: list[ParsedStudentAnswer] = field(default_factory=list)

    # ── raw full OCR (all pages concatenated) for debugging ──
    raw_full_ocr: str = ""

    def answer_map(self) -> dict[int, str]:
        """Returns {question_number: answer_text} for easy DB writes."""
        return {a.question_number: a.answer_text for a in self.answers}


# ──────────────────────────────────────────────────────────
# PDF → images
# ──────────────────────────────────────────────────────────

def _pdf_to_page_images(pdf_path: str, dpi: int = 200) -> list:
    """Returns a list of PIL Images, one per page."""
    try:
        import fitz
        doc    = fitz.open(pdf_path)
        images = []
        mat    = fitz.Matrix(dpi / 72, dpi / 72)
        for page in doc:
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
            from PIL import Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
        doc.close()
        logger.info("PyMuPDF: extracted %d page images from %s", len(images), pdf_path)
        return images
    except ImportError:
        pass

    try:
        from pdf2image import convert_from_path
        images = convert_from_path(pdf_path, dpi=dpi)
        logger.info("pdf2image: extracted %d page images from %s", len(images), pdf_path)
        return images
    except ImportError:
        raise RuntimeError(
            "Neither PyMuPDF (fitz) nor pdf2image is installed. "
            "Install one: pip install pymupdf  OR  pip install pdf2image"
        )


# ──────────────────────────────────────────────────────────
# Hybrid OCR per page (PaddleOCR → Tesseract → TrOCR via OCRModule v7)
# ──────────────────────────────────────────────────────────

def _ocr_page(image, ocr_module=None) -> str:
    if ocr_module is not None:
        result = ocr_module.extract_text_from_image(image)
        return result.text if hasattr(result, "text") else str(result)

    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from backend.ocr_module import OCRModule
        _ocr = OCRModule()
        result = _ocr.extract_text_from_image(image)
        return result.text if hasattr(result, "text") else str(result)
    except Exception as e:
        logger.warning("OCR failed on page: %s", e)
        return ""


def _ocr_all_pages(images: list, ocr_module=None) -> list[str]:
    texts = []
    for i, img in enumerate(images):
        logger.info("OCR page %d/%d …", i + 1, len(images))
        t = _ocr_page(img, ocr_module)
        texts.append(t)
        if not t.strip():
            logger.warning("Page %d OCR returned empty text", i + 1)
    return texts


# ──────────────────────────────────────────────────────────
# LLM: Cover page metadata extraction
# ──────────────────────────────────────────────────────────

def _extract_cover_metadata(cover_ocr: str, llm) -> dict:
    prompt = f"""You are parsing the cover page of a university exam answer booklet.
Extract student and exam metadata from the OCR text below.
The text may be noisy due to handwriting OCR errors.

OCR TEXT:
{cover_ocr[:3000]}

Return ONLY valid JSON (no markdown, no explanation):
{{
  "roll_number": "<roll number or empty string>",
  "register_number": "<register/hall ticket number or empty string>",
  "set_name": "<Set-A or Set-B or empty string>",
  "course_code": "<e.g. 11BLH41 or empty string>",
  "course_name": "<e.g. Database Management Systems or empty string>",
  "exam_name": "<e.g. CONTINUOUS ASSESSMENT EXAM or empty string>",
  "programme": "<e.g. BE CSE or empty string>",
  "semester": "<e.g. 04 or empty string>",
  "date_of_exam": "<dd/mm/yy or empty string>"
}}"""
    try:
        response = llm.generate(prompt, max_tokens=500)
        raw = re.sub(r"```json\s*|```\s*", "", response.text).strip()
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        return json.loads(m.group() if m else raw)
    except Exception as e:
        logger.error("Cover metadata extraction failed: %s", e)
        return {}


# ──────────────────────────────────────────────────────────
# LLM: Segment full OCR text into per-question answers
# ──────────────────────────────────────────────────────────

_SEGMENT_PROMPT = """You are parsing OCR text from a student's handwritten university exam answer booklet.
The OCR output is VERY NOISY - spelling errors, garbled words, and broken formatting are expected.
{context}

The exam has TWO parts:
- PART-A: Short answer questions (Q1 to Q7), 2 marks each
- PART-B: Long answer questions (Q8 to Q13), 12 marks each

Students write the question number before their answer. Due to OCR noise, question numbers may appear as:
  "8)" or "(8)" or "Q8" or "Q.8" or "Ans 8" or "8." or even "8 )" or a standalone number at the start of a line.

IMPORTANT: Even if the text is very garbled, try your best to identify question boundaries.

OCR TEXT:
{text_chunk}

INSTRUCTIONS:
1. Identify each question the student attempted by finding question number markers.
2. Extract the COMPLETE answer text for each question (may span several paragraphs).
3. Do NOT truncate answers - include everything the student wrote for that question.
4. For OR alternatives: record with the OR question number.
5. Ignore page break markers like "--- PAGE BREAK ---".
6. If a question number appears twice (answer continues), merge the text.
7. part_name: assign "Part-A" for Q1-Q7 and "Part-B" for Q8+.
8. If no question boundaries found, return text as question_number 0.

CRITICAL JSON RULES - follow exactly:
- Return ONLY valid JSON. No markdown fences, no text before or after the JSON object.
- Each answer_text value MUST be a single-line string. Use the two characters backslash-n to represent line breaks. Never embed a literal newline character inside a JSON string value.
- Remove all double-quote characters from inside answer_text strings. Use single quotes instead.
- Remove all backslash characters from answer_text strings except for the backslash-n line break marker.

Return ONLY valid JSON in this exact format:
{{"answers": [{{"question_number": 1, "answer_text": "answer here", "part_name": "Part-A", "is_or_option": false}}]}}
"""

def _segment_answers(full_ocr: str, llm) -> list[dict]:
    """
    Send OCR text to LLM for segmentation into per-question answers.
    Chunks if text is long. Falls back to rule-based segmentation if LLM fails.
    """
    CHUNK = 5000   # smaller chunks = better LLM JSON compliance

    def _call_llm(text_chunk: str, context: str = "") -> list[dict]:
        prompt = _SEGMENT_PROMPT.format(text_chunk=text_chunk, context=context)
        try:
            response = llm.generate(prompt, max_tokens=6000)
            raw = re.sub(r"```json\s*|```\s*", "", response.text).strip()
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            raw_json = m.group() if m else raw

            # ── Multi-pass JSON sanitization ──────────────────────────────
            # Pass 1: fix bare backslashes not part of a valid escape
            raw_json = re.sub(r'\\(?!["\\/bfnrtu0-9])', r'\\\\', raw_json)

            # Pass 2: try direct parse
            try:
                data = json.loads(raw_json)
                return data.get("answers", [])
            except json.JSONDecodeError:
                pass

            # Pass 3: replace literal newlines INSIDE string values with \\n
            # This fixes the most common failure: LLM embeds real newlines in a JSON string
            def _fix_string_newlines(s: str) -> str:
                result = []
                in_string = False
                escape_next = False
                for ch in s:
                    if escape_next:
                        result.append(ch)
                        escape_next = False
                    elif ch == '\\' and in_string:
                        result.append(ch)
                        escape_next = True
                    elif ch == '"':
                        result.append(ch)
                        in_string = not in_string
                    elif in_string and ch == '\n':
                        result.append('\\n')
                    elif in_string and ch == '\r':
                        result.append('\\r')
                    elif in_string and ch == '\t':
                        result.append('\\t')
                    else:
                        result.append(ch)
                return ''.join(result)

            raw_json2 = _fix_string_newlines(raw_json)
            try:
                data = json.loads(raw_json2)
                return data.get("answers", [])
            except json.JSONDecodeError:
                pass

            # Pass 4: extract answers array directly using regex (last resort)
            answers_match = re.search(r'"answers"\s*:\s*(\[.*?\])\s*[,}]', raw_json, re.DOTALL)
            if answers_match:
                try:
                    arr_text = answers_match.group(1)
                    # Try to parse individual answer objects
                    objs = re.findall(r'\{[^{}]+\}', arr_text, re.DOTALL)
                    results = []
                    for obj in objs:
                        obj_fixed = _fix_string_newlines(obj)
                        obj_fixed = re.sub(r'\\(?!["\\/bfnrtu0-9])', r'\\\\', obj_fixed)
                        try:
                            results.append(json.loads(obj_fixed))
                        except Exception:
                            # Extract fields manually for this object
                            qn = re.search(r'"question_number"\s*:\s*(\d+)', obj)
                            at = re.search(r'"answer_text"\s*:\s*"(.*?)"(?=\s*,\s*"|\s*})', obj, re.DOTALL)
                            pt = re.search(r'"part_name"\s*:\s*"(.*?)"', obj)
                            if qn:
                                results.append({
                                    "question_number": int(qn.group(1)),
                                    "answer_text": at.group(1).replace('\\"', '"') if at else "",
                                    "part_name": pt.group(1) if pt else ("Part-B" if int(qn.group(1)) >= 8 else "Part-A"),
                                    "is_or_option": False,
                                })
                    if results:
                        logger.info("Pass 4 rescued %d answers from malformed JSON", len(results))
                        return results
                except Exception as e2:
                    logger.warning("Pass 4 rescue failed: %s", e2)

            raise ValueError("All JSON parse attempts failed")

        except Exception as e:
            logger.error("Answer segmentation LLM call failed: %s", e)
            return []

    if len(full_ocr) <= CHUNK:
        result = _call_llm(full_ocr)
        if not result:
            logger.warning("LLM segmentation returned empty — using rule-based fallback")
            return _segment_rule_based(full_ocr)
        return result

    # Multi-chunk processing
    all_answers: dict[int, dict] = {}
    chunks = [full_ocr[i:i + CHUNK] for i in range(0, len(full_ocr), CHUNK)]
    for idx, chunk in enumerate(chunks):
        context = f"(Chunk {idx+1} of {len(chunks)} — answers may continue across chunks.)"
        partial = _call_llm(chunk, context)
        for ans in partial:
            qn = ans.get("question_number")
            if qn is None:
                continue
            try:
                qn = int(qn)
            except (TypeError, ValueError):
                continue
            if qn in all_answers:
                # Append continuation — deduplicate repeated OCR text
                existing = all_answers[qn]["answer_text"].strip()
                new_text = (ans.get("answer_text") or "").strip()
                if new_text and new_text not in existing:
                    all_answers[qn]["answer_text"] = existing + "\n" + new_text
            else:
                all_answers[qn] = ans

    result = list(all_answers.values())
    if not result:
        logger.warning("LLM multi-chunk segmentation returned empty — using rule-based fallback")
        return _segment_rule_based(full_ocr)
    return result


# ──────────────────────────────────────────────────────────
# Rule-based fallback segmentation
# ──────────────────────────────────────────────────────────

def _segment_rule_based(full_ocr: str) -> list[dict]:
    """
    Fallback segmentation using regex to detect question numbers.
    Handles very noisy OCR from handwritten booklets.
    """
    logger.info("Running rule-based answer segmentation")
    answers: dict[int, dict] = {}
    lines = full_ocr.splitlines()

    q_start_patterns = [
        re.compile(r"^\s*(\d{1,2})\s*[.)]\s+(.{0,})"),
        re.compile(r"^\s*\((\d{1,2})\)\s*\.?\s*(.{0,})"),
        re.compile(r"^\s*[Qq]\.?\s*(\d{1,2})\s*[.):]\s*(.{0,})"),
        re.compile(r"^\s*[Aa]ns(?:wer)?\s*\.?\s*(\d{1,2})\s*[.:)]\s*(.*)", re.I),
        re.compile(r"^\s*(\d{1,2})\s*[.)]\s*$"),
        re.compile(r"^\s*[Qq]\.?\s*(\d{1,2})\s*$"),
    ]

    current_q   = None
    current_ans: list[str] = []

    def _flush():
        if current_q is not None and current_ans:
            text = "\n".join(current_ans).strip()
            if len(text) > 5:
                part = "Part-B" if current_q >= 8 else "Part-A"
                if current_q in answers:
                    answers[current_q]["answer_text"] += "\n" + text
                else:
                    answers[current_q] = {
                        "question_number": current_q,
                        "answer_text": text,
                        "part_name": part,
                        "is_or_option": False,
                    }

    for line in lines:
        stripped = line.strip()
        if not stripped or "--- PAGE BREAK ---" in stripped:
            continue

        matched = False
        for pat in q_start_patterns:
            m = pat.match(stripped)
            if m:
                q_num = int(m.group(1))
                if 1 <= q_num <= 20:
                    _flush()
                    current_q   = q_num
                    first_text  = m.group(2).strip() if m.lastindex and m.lastindex >= 2 else ""
                    current_ans = [first_text] if first_text else []
                    matched = True
                    break

        if not matched and current_q is not None:
            current_ans.append(stripped)

    _flush()

    if not answers:
        logger.warning("Rule-based: no question markers found — returning full text as Q0")
        clean = full_ocr.replace("--- PAGE BREAK ---", "").strip()
        if clean:
            return [{
                "question_number": 0,
                "answer_text": clean,
                "part_name": "unknown",
                "is_or_option": False,
            }]

    result = sorted(answers.values(), key=lambda x: x["question_number"])
    logger.info("Rule-based segmentation found %d answers: Qs %s",
                len(result), [a["question_number"] for a in result])
    return result


# ──────────────────────────────────────────────────────────
# Main public function
# ──────────────────────────────────────────────────────────

def parse_student_booklet(
    pdf_path: str,
    llm=None,
    ocr_module=None,
    skip_cover_page: bool = True,
) -> ParsedStudentBooklet:
    """
    Full pipeline: PDF → images → OCR → LLM segmentation → ParsedStudentBooklet.

    Args:
        pdf_path:        Path to the student's answer booklet PDF.
        llm:             LLMClient instance. If None, creates one via get_llm_client().
        ocr_module:      Shared OCRModule instance (optional, avoids reloading weights).
        skip_cover_page: If True, page 0 (cover) is used only for metadata, not OCR content.

    Returns:
        ParsedStudentBooklet with all extracted data.
    """
    logger.info("Parsing student booklet: %s", pdf_path)

    if llm is None:
        from backend.llm_provider import get_llm_client
        llm = get_llm_client()

    booklet = ParsedStudentBooklet()

    # Step 1: PDF → page images
    images = _pdf_to_page_images(pdf_path)
    booklet.total_pages = len(images)
    logger.info("Total pages: %d", booklet.total_pages)

    if not images:
        logger.error("No pages extracted from PDF: %s", pdf_path)
        return booklet

    # Step 2: OCR all pages
    all_page_texts = _ocr_all_pages(images, ocr_module)

    # Step 3: Extract cover metadata (page 0)
    cover_ocr = all_page_texts[0] if all_page_texts else ""
    if cover_ocr.strip():
        meta = _extract_cover_metadata(cover_ocr, llm)
    else:
        meta = {}
        logger.warning("Cover page OCR returned empty text — skipping metadata extraction")

    booklet.roll_number     = meta.get("roll_number", "")
    booklet.register_number = meta.get("register_number", "")
    booklet.set_name        = meta.get("set_name", "")
    booklet.course_code     = meta.get("course_code", "")
    booklet.course_name     = meta.get("course_name", "")
    booklet.exam_name       = meta.get("exam_name", "")
    booklet.programme       = meta.get("programme", "")
    booklet.semester        = meta.get("semester", "")
    booklet.date_of_exam    = meta.get("date_of_exam", "")
    logger.info(
        "Cover: roll=%s reg=%s set=%s course=%s",
        booklet.roll_number, booklet.register_number,
        booklet.set_name, booklet.course_code,
    )

    # Step 4: Combine content page OCR
    content_texts = all_page_texts[1:] if skip_cover_page else all_page_texts
    # Filter out completely blank pages
    non_blank = [(i, t) for i, t in enumerate(content_texts) if t.strip()]
    logger.info(
        "%d / %d content pages have OCR text",
        len(non_blank), len(content_texts)
    )

    full_ocr = "\n\n--- PAGE BREAK ---\n\n".join(
        f"[Page {i+2}]\n{t}" for i, t in enumerate(content_texts)
    )
    booklet.raw_full_ocr = full_ocr

    if not full_ocr.strip():
        logger.error("All content pages returned empty OCR text for booklet: %s", pdf_path)
        return booklet

    # Step 5: LLM segmentation into per-question answers
    raw_answers = _segment_answers(full_ocr, llm)

    booklet.answers = [
        ParsedStudentAnswer(
            question_number = int(a.get("question_number", 0)),
            answer_text     = a.get("answer_text", "").strip(),
            part_name       = a.get("part_name", ""),
            is_or_option    = bool(a.get("is_or_option", False)),
        )
        for a in raw_answers
        if a.get("question_number") and a.get("answer_text", "").strip()
    ]

    logger.info(
        "Segmented %d answers: Qs %s",
        len(booklet.answers),
        sorted(a.question_number for a in booklet.answers),
    )

    return booklet