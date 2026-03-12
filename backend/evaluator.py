"""
IntelliGrade-H - Evaluation Engine (v3.1 — Groq-only)
======================================================
Orchestrates the full grading pipeline for any question type.

Supported question types:
  ┌─────────────┬────────────────────────────────────────────────────────────┐
  │ Type        │ Grading method                                             │
  ├─────────────┼────────────────────────────────────────────────────────────┤
  │ mcq         │ Deterministic: OCR → extract option → exact match         │
  │ true_false  │ Deterministic: OCR → extract T/F → exact match            │
  │ fill_blank  │ LLM: near-match with OCR tolerance                        │
  │ short_answer│ LLM + Similarity + Keyword                                │
  │ numerical   │ LLM: method + answer with tolerance                       │
  │ open_ended  │ LLM + Similarity + Rubric + Keyword + Length (full)       │
  │ diagram     │ LLM (OCR text of labels/annotations)                      │
  └─────────────┴────────────────────────────────────────────────────────────┘

Hybrid scoring formula (open_ended / short_answer):
  Final = 0.40 × LLM Score        (Groq llama-3.3-70b)
        + 0.25 × Semantic Similarity
        + 0.20 × Rubric Coverage
        + 0.10 × Keyword Coverage
        + 0.05 × Length Normalization

Set question_type="auto" to let Groq classify it automatically.
"""

import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from pathlib import Path

from backend.ocr_module import OCRModule, OCRResult
from backend.text_processor import TextProcessor, ProcessedText
from backend.similarity import SemanticSimilarityModel, SimilarityResult
from backend.llm_evaluator import LLMEvaluator, LLMEvaluation
from backend.rubric_matcher import RubricMatcher, RubricResult
from backend.diagram_detector import DiagramDetector, DiagramDetectionResult

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# Question type enum
# ─────────────────────────────────────────────────────────

class QuestionType(str, Enum):
    MCQ          = "mcq"
    TRUE_FALSE   = "true_false"
    FILL_BLANK   = "fill_blank"
    SHORT_ANSWER = "short_answer"
    NUMERICAL    = "numerical"
    OPEN_ENDED   = "open_ended"
    DIAGRAM      = "diagram"
    AUTO         = "auto"        # triggers LLM auto-classification

# Types that also use semantic similarity
SIMILARITY_TYPES = {QuestionType.OPEN_ENDED, QuestionType.SHORT_ANSWER}


# ─────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────

@dataclass
class EvaluationResult:
    final_score: float
    max_marks: float
    question_type: str            # the actual type used (never "auto")

    # LLM pipeline scores (0.0 for deterministic types)
    llm_score: float = 0.0
    similarity_score: float = 0.0
    rubric_score: float = 0.0
    keyword_score: float = 0.0
    length_score: float = 0.0

    # Deterministic grading fields
    detected_answer: str = ""
    correct_answer: str = ""
    is_correct: Optional[bool] = None

    # OCR
    extracted_text: str = ""
    ocr_confidence: float = 0.0
    ocr_engine: str = ""

    # Feedback
    strengths: list = field(default_factory=list)
    missing_concepts: list = field(default_factory=list)
    feedback: str = ""

    # Meta
    confidence: float = 0.0
    evaluation_time_sec: float = 0.0
    rubric_details: Optional[dict] = None
    llm_provider: str = ""
    auto_classified: bool = False

    # Diagram detection (populated for diagram question type)
    diagram_detected: bool = False
    n_diagrams: int = 0
    diagram_detector_used: str = ""
    diagram_bboxes: list = field(default_factory=list)

    # Dynamic paper tracking (set by API layer)
    exam_paper_id: Optional[str] = None
    exam_question_id: Optional[str] = None
    question_number: Optional[int] = None

    # component_scores dict for booklet evaluate endpoint
    @property
    def component_scores(self) -> dict:
        return {
            "llm":        self.llm_score,
            "similarity": self.similarity_score,
            "rubric":     self.rubric_score,
            "keyword":    self.keyword_score,
            "length":     self.length_score,
        }

    # Legacy aliases for API backward compat
    @property
    def mcq_correct(self): return self.is_correct
    @property
    def mcq_detected_answer(self): return self.detected_answer
    @property
    def mcq_correct_answer(self): return self.correct_answer


# ─────────────────────────────────────────────────────────
# Evaluation Engine
# ─────────────────────────────────────────────────────────

class EvaluationEngine:
    """
    Main pipeline for IntelliGrade-H.

    LLM provider: Groq (llama-3.3-70b-versatile) — configured via GROQ_API_KEY in .env
    OCR model:    TrOCR fine-tuned — configured via TROCR_MODEL_PATH in .env

    Hybrid scoring weights (open_ended / short_answer):
      LLM Score         : 0.40
      Semantic Similarity: 0.25
      Rubric Coverage   : 0.20
      Keyword Coverage  : 0.10
      Length Normaliz.  : 0.05
    """

    LLM_WEIGHT        = 0.40
    SIM_WEIGHT        = 0.25
    RUBRIC_WEIGHT     = 0.20
    KEYWORD_WEIGHT    = 0.10
    LENGTH_WEIGHT     = 0.05

    def __init__(
        self,
        ocr_engine: str = "trocr",
        trocr_model_path: Optional[str] = None,
        similarity_model: Optional[str] = None,
        llm_weight: float = 0.40,
        similarity_weight: float = 0.25,
        rubric_weight: float = 0.20,
        keyword_weight: float = 0.10,
        length_weight: float = 0.05,
        use_rubric: bool = True,
    ):
        self.LLM_WEIGHT     = llm_weight
        self.SIM_WEIGHT     = similarity_weight
        self.RUBRIC_WEIGHT  = rubric_weight
        self.KEYWORD_WEIGHT = keyword_weight
        self.LENGTH_WEIGHT  = length_weight

        import os
        _trocr_path  = trocr_model_path or os.getenv("TROCR_MODEL_PATH", "microsoft/trocr-small-handwritten")
        _sbert_model = similarity_model  or os.getenv("SBERT_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

        self.ocr              = OCRModule(trocr_model_path=_trocr_path)
        self.text_processor   = TextProcessor()
        self.similarity_model = SemanticSimilarityModel(_sbert_model)
        self.llm_evaluator    = LLMEvaluator()
        self.rubric_matcher   = RubricMatcher() if use_rubric else None
        self.diagram_detector = DiagramDetector()   # YOLO + heuristic fallback
        self._classifier      = None  # lazy-init

        logger.info(
            "EvaluationEngine initialized. OCR model: %s | LLM: Groq/%s",
            _trocr_path,
            __import__("os").getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        )

    def _get_classifier(self):
        if self._classifier is None:
            from backend.question_classifier import QuestionClassifier
            self._classifier = QuestionClassifier(llm_evaluator=self.llm_evaluator)
        return self._classifier

    # ─────────────────────────────────────────────────────
    # Main Entry Point
    # ─────────────────────────────────────────────────────

    def evaluate(
        self,
        student_image=None,
        question: str = "",
        teacher_answer: str = "",
        max_marks: float = 10.0,
        rubric_criteria: Optional[list] = None,
        question_type: str = "auto",
        correct_option: Optional[str] = None,
        correct_answer: Optional[str] = None,
        mcq_options: Optional[dict] = None,
        numerical_tolerance: float = 0.01,
        # Alternative: pass text directly (used by booklet evaluation)
        student_answer: Optional[str] = None,
        question_text: Optional[str] = None,
        question_number: Optional[int] = None,
    ) -> EvaluationResult:
        """
        Full pipeline evaluation for any question type.

        Two modes:
          1. student_image path/PIL  — full OCR → evaluate pipeline
          2. student_answer text     — skip OCR, evaluate text directly
        """
        start = time.time()

        # Allow question_text alias for question
        if question_text and not question:
            question = question_text

        # ── Mode 2: text passed directly (booklet evaluation) ──
        if student_answer is not None and student_image is None:
            ocr_result = OCRResult(
                text=student_answer, confidence=1.0, engine="direct-text"
            )
            student_text_raw = student_answer
        else:
            # ── Mode 1: OCR from image/PDF ──────────────────────
            logger.info("Step 1: Running OCR...")
            if student_image is None:
                return self._empty_result(
                    max_marks,
                    question_type if question_type != "auto" else "open_ended",
                    "No student image or answer text provided.",
                    time.time() - start,
                )

            path = Path(str(student_image))
            if path.suffix.lower() == ".pdf":
                try:
                    page_results = self.ocr.extract_from_pdf(str(path))
                    if page_results:
                        combined_text = "\n".join(r.text for r in page_results if r.text)
                        avg_conf = sum(r.confidence for r in page_results) / len(page_results)
                        ocr_result = OCRResult(text=combined_text, confidence=avg_conf, engine="trocr-pdf")
                    else:
                        ocr_result = OCRResult(text="", confidence=0.0, engine="pdf")
                except Exception as e:
                    logger.error("PDF OCR failed: %s", e)
                    ocr_result = OCRResult(text="", confidence=0.0, engine="failed")
            else:
                ocr_result = self.ocr.extract_text(str(student_image))

            student_text_raw = ocr_result.text

        if not student_text_raw.strip():
            logger.warning("OCR/text input is empty.")
            return self._empty_result(
                max_marks,
                question_type if question_type != "auto" else "open_ended",
                "No text found in student answer.",
                time.time() - start,
            )

        # ── Auto-classify ─────────────────────────────────────
        auto_classified = False
        resolved_type   = question_type

        if question_type in ("auto", QuestionType.AUTO):
            logger.info("Auto-classifying question type via Groq...")
            clf    = self._get_classifier()
            result = clf.classify(question)
            resolved_type   = result.question_type
            auto_classified = True
            logger.info(
                "Auto-classified as '%s' (conf=%.2f, method=%s)",
                resolved_type, result.confidence, result.method,
            )
        else:
            resolved_type = QuestionType(question_type).value

        qtype = QuestionType(resolved_type)

        # ── Route to pipeline ─────────────────────────────────
        if qtype == QuestionType.MCQ:
            res = self._evaluate_mcq(
                student_text_raw, ocr_result, correct_option, max_marks, start,
                mcq_options=mcq_options,
            )
        elif qtype == QuestionType.TRUE_FALSE:
            res = self._evaluate_true_false(
                student_text_raw, ocr_result, correct_answer, max_marks, start
            )
        elif qtype == QuestionType.NUMERICAL:
            res = self._evaluate_numerical_or_llm(
                student_text_raw, ocr_result, question, teacher_answer,
                correct_answer, max_marks, numerical_tolerance, start,
            )
        elif qtype == QuestionType.DIAGRAM:
            res = self._evaluate_diagram(
                student_image, student_text_raw, ocr_result, question,
                teacher_answer, max_marks, rubric_criteria, start,
            )
        else:
            res = self._evaluate_with_llm(
                student_text_raw, ocr_result, question, teacher_answer,
                max_marks, rubric_criteria, qtype, start,
            )

        res.auto_classified = auto_classified
        return res

    # ─────────────────────────────────────────────────────
    # MCQ
    # ─────────────────────────────────────────────────────

    def _evaluate_mcq(self, raw_text, ocr_result, correct_option, max_marks, start,
                      mcq_options: Optional[dict] = None):
        if not correct_option:
            return self._empty_result(
                max_marks, "mcq",
                "Correct option not provided for MCQ grading.",
                time.time() - start,
            )

        detected   = _extract_mcq_answer(raw_text)
        correct    = correct_option.strip().upper()

        # If OCR confidence is low and we have MCQ options, attempt LLM-assisted detection
        if detected is None and mcq_options and ocr_result.confidence < 0.5:
            try:
                from backend.llm_provider import get_llm_client
                from backend.evaluation_prompts import MCQ_VALIDATION_PROMPT
                prompt = MCQ_VALIDATION_PROMPT.format(
                    question="",
                    option_a=mcq_options.get("A", ""),
                    option_b=mcq_options.get("B", ""),
                    option_c=mcq_options.get("C", ""),
                    option_d=mcq_options.get("D", ""),
                    student_answer=raw_text,
                )
                data = get_llm_client().generate_json(prompt)
                detected = (data.get("detected_option") or "").strip().upper() or None
                logger.info("MCQ LLM fallback detected: %s (conf=%.2f)", detected, data.get("confidence", 0))
            except Exception as e:
                logger.warning("MCQ LLM fallback failed: %s", e)

        is_correct = (detected == correct) if detected else False
        score      = max_marks if is_correct else 0.0

        feedback = (
            f"✅ Correct! You selected option {detected}."
            if is_correct
            else f"❌ Incorrect. Detected '{detected or 'unclear'}', correct answer is '{correct}'."
        )

        return EvaluationResult(
            final_score=score, max_marks=max_marks, question_type="mcq",
            detected_answer=detected or "", correct_answer=correct,
            is_correct=is_correct,
            extracted_text=raw_text,
            ocr_confidence=round(ocr_result.confidence, 3),
            ocr_engine=ocr_result.engine,
            feedback=feedback,
            confidence=ocr_result.confidence,
            evaluation_time_sec=round(time.time() - start, 2),
        )

    # ─────────────────────────────────────────────────────
    # True / False
    # ─────────────────────────────────────────────────────

    def _evaluate_true_false(self, raw_text, ocr_result, correct_answer, max_marks, start):
        if not correct_answer:
            return self._empty_result(
                max_marks, "true_false",
                "Correct answer (True/False) not provided.",
                time.time() - start,
            )

        detected   = _extract_true_false(raw_text)
        correct    = correct_answer.strip().upper()
        is_correct = (detected == correct) if detected else False
        score      = max_marks if is_correct else 0.0

        feedback = (
            f"✅ Correct! The answer is {correct}."
            if is_correct
            else f"❌ Incorrect. Detected '{detected or 'unclear'}', correct answer is '{correct}'."
        )

        return EvaluationResult(
            final_score=score, max_marks=max_marks, question_type="true_false",
            detected_answer=detected or "", correct_answer=correct,
            is_correct=is_correct,
            extracted_text=raw_text,
            ocr_confidence=round(ocr_result.confidence, 3),
            ocr_engine=ocr_result.engine,
            feedback=feedback,
            confidence=ocr_result.confidence,
            evaluation_time_sec=round(time.time() - start, 2),
        )

    # ─────────────────────────────────────────────────────
    # Numerical
    # ─────────────────────────────────────────────────────

    def _evaluate_numerical_or_llm(
        self, raw_text, ocr_result, question, teacher_answer,
        correct_answer, max_marks, tolerance, start,
    ):
        if correct_answer:
            detected_num = _extract_number(raw_text)
            try:
                expected_num = float(correct_answer.strip().replace(",", ""))
                if detected_num is not None:
                    diff       = abs(detected_num - expected_num)
                    within_tol = diff <= abs(expected_num * tolerance) or diff < 0.001
                    score      = max_marks if within_tol else 0.0

                    feedback = (
                        f"✅ Correct numerical answer: {detected_num} (expected {expected_num})."
                        if within_tol
                        else f"❌ Numerical answer {detected_num} outside tolerance. Expected {expected_num}."
                    )

                    return EvaluationResult(
                        final_score=score, max_marks=max_marks, question_type="numerical",
                        detected_answer=str(detected_num), correct_answer=str(expected_num),
                        is_correct=within_tol,
                        extracted_text=raw_text,
                        ocr_confidence=round(ocr_result.confidence, 3),
                        ocr_engine=ocr_result.engine,
                        feedback=feedback,
                        confidence=0.85,
                        evaluation_time_sec=round(time.time() - start, 2),
                    )
            except ValueError:
                pass

        return self._evaluate_with_llm(
            raw_text, ocr_result, question, teacher_answer,
            max_marks, None, QuestionType.NUMERICAL, start,
        )

    # ─────────────────────────────────────────────────────
    # Diagram
    # ─────────────────────────────────────────────────────

    def _evaluate_diagram(
        self, student_image, raw_text, ocr_result, question, teacher_answer,
        max_marks, rubric_criteria, start,
    ):
        """
        Diagram question pipeline:
          1. Run DiagramDetector on the student image (YOLO → heuristic).
          2. Append detection context to the OCR text so the LLM is aware
             of how many / what kind of diagram regions were found.
          3. Pass the enriched text through the normal LLM evaluator.
          4. Apply a presence bonus: if the question clearly requires a
             diagram and one is detected, add a small guaranteed mark
             (up to 20% of max_marks) so that a drawn-but-poorly-labelled
             diagram still receives partial credit.
        """
        logger.info("Step: Diagram detection...")

        diagram_result: Optional[DiagramDetectionResult] = None
        diagram_context = ""

        # Only run visual detection when we have an actual image path/bytes
        if student_image is not None:
            try:
                diagram_result = self.diagram_detector.detect(student_image)
                if diagram_result.has_diagram:
                    diagram_context = (
                        f"\n\n[Diagram Detection: {diagram_result.n_diagrams} diagram region(s) "
                        f"detected via {diagram_result.detector_used}. "
                        f"The student has drawn a diagram as part of their answer.]"
                    )
                    logger.info(
                        "Diagram detected: %d region(s) via %s",
                        diagram_result.n_diagrams,
                        diagram_result.detector_used,
                    )
                else:
                    diagram_context = (
                        "\n\n[Diagram Detection: No diagram regions detected. "
                        "The student may not have drawn the required diagram.]"
                    )
                    logger.info("No diagram regions found.")
            except Exception as exc:
                logger.warning("Diagram detection failed: %s", exc)

        # Enrich OCR text with diagram context before LLM evaluation
        enriched_text = raw_text + diagram_context
        enriched_ocr  = OCRResult(
            text=enriched_text,
            confidence=ocr_result.confidence,
            engine=ocr_result.engine,
        )

        # Run LLM pipeline on enriched text
        llm_result = self._evaluate_with_llm(
            enriched_text, enriched_ocr, question, teacher_answer,
            max_marks, rubric_criteria, QuestionType.DIAGRAM, start,
        )

        # ── Diagram presence bonus ────────────────────────────────────────
        # If a diagram was detected but the LLM gave a very low score
        # (possibly because labels/annotations were illegible), give the
        # student at least 20% of max_marks for having drawn something.
        PRESENCE_FLOOR = 0.20  # 20% of max_marks
        if (
            diagram_result is not None
            and diagram_result.has_diagram
            and llm_result.final_score < (max_marks * PRESENCE_FLOOR)
        ):
            llm_result.final_score = round(max_marks * PRESENCE_FLOOR, 2)
            llm_result.feedback = (
                "[Partial credit: diagram detected but labels/content were unclear.] "
                + llm_result.feedback
            )

        # Attach diagram metadata to result
        if diagram_result is not None:
            llm_result.diagram_detected      = diagram_result.has_diagram
            llm_result.n_diagrams            = diagram_result.n_diagrams
            llm_result.diagram_detector_used = diagram_result.detector_used
            llm_result.diagram_bboxes        = [
                vars(r) if hasattr(r, "__dict__") else r
                for r in getattr(diagram_result, "regions", [])
            ]

        return llm_result

    # ─────────────────────────────────────────────────────
    # LLM-based pipeline
    # ─────────────────────────────────────────────────────

    def _evaluate_with_llm(
        self, student_text_raw, ocr_result, question, teacher_answer,
        max_marks, rubric_criteria, qtype, start,
    ):
        logger.info("Text processing...")
        processed    = self.text_processor.process(student_text_raw)
        student_text = processed.cleaned

        sim_score = 0.0
        sentence_scores = None
        if qtype in SIMILARITY_TYPES and teacher_answer:
            logger.info("Computing semantic similarity...")
            sim_result = self.similarity_model.compute_similarity(student_text, teacher_answer)
            sim_score  = sim_result.score
            # For open_ended, also compute sentence-level breakdown for richer feedback
            if qtype == QuestionType.OPEN_ENDED and len(student_text.split()) > 10:
                try:
                    sent_result  = self.similarity_model.compute_sentence_level(student_text, teacher_answer)
                    sentence_scores = sent_result.get("sentence_scores")
                except Exception as e:
                    logger.debug("Sentence-level similarity skipped: %s", e)

        logger.info("Running Groq LLM evaluation (type=%s)...", qtype.value)
        rubric_labels = [r["criterion"] for r in rubric_criteria] if rubric_criteria else None
        llm_eval: LLMEvaluation = self.llm_evaluator.evaluate(
            question        = question,
            teacher_answer  = teacher_answer,
            student_answer  = student_text,
            max_marks       = max_marks,
            rubric_criteria = rubric_labels,
            question_type   = qtype.value,
        )

        rubric_result = None
        rubric_score  = 0.0
        if self.rubric_matcher and rubric_criteria and qtype == QuestionType.OPEN_ENDED:
            logger.info("Rubric matching...")
            rubric_result = self.rubric_matcher.evaluate_rubric(
                student_answer  = student_text,
                rubric_criteria = rubric_criteria,
                question_type   = "open_ended",
            )
            rubric_score = rubric_result.coverage_ratio

        keyword_score = 0.0
        length_score  = 0.0
        if teacher_answer and qtype in SIMILARITY_TYPES:
            keyword_score = self._keyword_coverage(student_text, teacher_answer)
            length_score  = self._length_normalization(student_text, teacher_answer)

        if qtype in SIMILARITY_TYPES:
            final_score = self._hybrid_score(
                llm_score        = llm_eval.score,
                similarity       = sim_score,
                rubric_coverage  = rubric_score,
                keyword_coverage = keyword_score,
                length_norm      = length_score,
                max_marks        = max_marks,
            )
        else:
            final_score = llm_eval.score

        final_score = round(max(0.0, min(max_marks, final_score)), 2)
        elapsed     = round(time.time() - start, 2)

        return EvaluationResult(
            final_score      = final_score,
            max_marks        = max_marks,
            question_type    = qtype.value,
            llm_score        = llm_eval.score,
            similarity_score = round(sim_score, 4),
            rubric_score     = round(rubric_score, 4),
            keyword_score    = round(keyword_score, 4),
            length_score     = round(length_score, 4),
            extracted_text   = student_text,
            ocr_confidence   = round(ocr_result.confidence, 3),
            ocr_engine       = ocr_result.engine,
            strengths        = llm_eval.strengths,
            missing_concepts = llm_eval.missing_concepts,
            feedback         = llm_eval.feedback,
            confidence       = round(llm_eval.confidence, 3),
            evaluation_time_sec = elapsed,
            rubric_details   = {
                **(vars(rubric_result) if rubric_result else {}),
                **({"sentence_scores": sentence_scores} if sentence_scores else {}),
            } or None,
            llm_provider     = f"{llm_eval.provider}/{llm_eval.model}",
        )

    # ─────────────────────────────────────────────────────
    # Batch Evaluation
    # ─────────────────────────────────────────────────────

    def evaluate_batch(
        self,
        submissions: list,
        question: str,
        teacher_answer: str = "",
        max_marks: float = 10.0,
        rubric_criteria: Optional[list] = None,
        question_type: str = "auto",
        correct_option: Optional[str] = None,
        correct_answer: Optional[str] = None,
        mcq_options: Optional[dict] = None,
        numerical_tolerance: float = 0.01,
    ) -> list:
        results = []
        for i, sub in enumerate(submissions):
            logger.info("Evaluating %d/%d — student: %s", i + 1, len(submissions), sub["student_id"])
            try:
                result = self.evaluate(
                    student_image=sub["image"],
                    question=question,
                    teacher_answer=teacher_answer,
                    max_marks=max_marks,
                    rubric_criteria=rubric_criteria,
                    question_type=question_type,
                    correct_option=correct_option,
                    correct_answer=correct_answer,
                    mcq_options=mcq_options,
                    numerical_tolerance=numerical_tolerance,
                )
                results.append({"student_id": sub["student_id"], "result": result})
            except Exception as e:
                logger.error("Failed for student %s: %s", sub["student_id"], e)
                results.append({
                    "student_id": sub["student_id"],
                    "result": self._empty_result(max_marks, question_type, str(e), 0.0),
                })
        return results

    # ─────────────────────────────────────────────────────
    # Scoring helpers
    # ─────────────────────────────────────────────────────

    def _hybrid_score(
        self,
        llm_score: float,
        similarity: float,
        max_marks: float = 10.0,
        rubric_coverage: float = 0.0,
        keyword_coverage: float = 0.5,
        length_norm: float = 1.0,
    ) -> float:
        llm_ratio = (llm_score / max_marks) if max_marks > 0 else 0.0
        combined = (
            self.LLM_WEIGHT     * llm_ratio
          + self.SIM_WEIGHT     * similarity
          + self.RUBRIC_WEIGHT  * rubric_coverage
          + self.KEYWORD_WEIGHT * keyword_coverage
          + self.LENGTH_WEIGHT  * length_norm
        )
        return round(combined * max_marks, 2)

    def _keyword_coverage(self, student_text: str, teacher_answer: str) -> float:
        import re

        def extract_keywords(text: str) -> set:
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            stop = {
                "the", "and", "are", "for", "that", "this", "with", "from",
                "was", "not", "but", "can", "will", "also", "its", "used",
                "has", "have", "been", "into", "than", "then", "which",
                "when", "they", "some", "any", "each", "more", "very",
            }
            return {w for w in words if w not in stop}

        teacher_kw = extract_keywords(teacher_answer)
        student_kw = extract_keywords(student_text)

        if not teacher_kw:
            return 1.0

        covered = len(teacher_kw & student_kw)
        return round(min(1.0, covered / len(teacher_kw)), 4)

    def _length_normalization(self, student_text: str, teacher_answer: str) -> float:
        student_len = len(student_text.split())
        teacher_len = len(teacher_answer.split())
        if teacher_len == 0:
            return 1.0
        ratio = student_len / teacher_len
        return round(min(1.0, ratio / 0.6), 4)

    def _empty_result(
        self, max_marks: float, question_type: str, reason: str, elapsed: float
    ) -> EvaluationResult:
        return EvaluationResult(
            final_score=0.0, max_marks=max_marks,
            question_type=question_type if question_type != "auto" else "open_ended",
            extracted_text="", ocr_confidence=0.0, ocr_engine="none",
            feedback=reason, evaluation_time_sec=elapsed,
        )


# ─────────────────────────────────────────────────────────
# Answer extractors (module-level utilities)
# ─────────────────────────────────────────────────────────

def _extract_mcq_answer(text: str) -> Optional[str]:
    """Extract selected MCQ option letter (A–E) from OCR text."""
    import re
    VALID = set("ABCDE")
    patterns = [
        r'\b(?:answer|ans|option|choice|selected?|marked?)\s*[:\-=]?\s*\(?([A-Ea-e])\)?',
        r'^\s*\(?([A-Ea-e])\)?\s*$',
        r'\(([A-Ea-e])\)',
        r'\b([A-Ea-e])\)',
        r'[✓☑✗]\s*([A-Ea-e])',
        r'\b([A-Ea-e])\b',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            letter = m.group(1).upper()
            if letter in VALID:
                return letter
    return None


def _extract_true_false(text: str) -> Optional[str]:
    """Extract True or False from OCR text. Returns 'TRUE', 'FALSE', or None."""
    import re
    t = text.strip().lower()
    if re.search(r'\b(true|correct|yes|t)\b', t):
        if not re.search(r'\b(false|incorrect|no|f)\b', t):
            return "TRUE"
    if re.search(r'\b(false|incorrect|no|f)\b', t):
        if not re.search(r'\b(true|correct|yes|t)\b', t):
            return "FALSE"
    m = re.match(r'^\s*([tf])\s*$', t, re.IGNORECASE)
    if m:
        return "TRUE" if m.group(1).upper() == "T" else "FALSE"
    return None


def _extract_number(text: str) -> Optional[float]:
    """Extract the most likely numerical answer from OCR text."""
    import re
    matches = re.findall(r'-?\d+(?:\.\d+)?', text.replace(",", ""))
    if not matches:
        return None
    return float(matches[-1])