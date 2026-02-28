"""
IntelliGrade-H - Evaluation Engine
Orchestrates the full grading pipeline for any question type.

Supported question types:
  ┌─────────────┬───────────────────────────────────────────────────┐
  │ Type        │ Grading method                                    │
  ├─────────────┼───────────────────────────────────────────────────┤
  │ mcq         │ Deterministic: OCR → extract option → exact match │
  │ true_false  │ Deterministic: OCR → extract T/F → exact match   │
  │ fill_blank  │ LLM: near-match with OCR tolerance                │
  │ short_answer│ LLM + Similarity                                  │
  │ numerical   │ LLM: method + answer with tolerance               │
  │ open_ended  │ LLM + Similarity + Rubric (full pipeline)         │
  │ diagram     │ LLM (OCR text of labels/annotations)              │
  └─────────────┴───────────────────────────────────────────────────┘

Set question_type="auto" to let the LLM classify it automatically.
"""

import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from backend.ocr_module import OCRModule, OCRResult
from backend.text_processor import TextProcessor, ProcessedText
from backend.similarity import SemanticSimilarityModel, SimilarityResult
from backend.llm_evaluator import LLMEvaluator, LLMEvaluation
from backend.rubric_matcher import RubricMatcher, RubricResult

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

# Types that use the full LLM pipeline
LLM_TYPES        = {QuestionType.OPEN_ENDED, QuestionType.SHORT_ANSWER,
                    QuestionType.FILL_BLANK, QuestionType.NUMERICAL, QuestionType.DIAGRAM}
# Types that also use semantic similarity
SIMILARITY_TYPES = {QuestionType.OPEN_ENDED, QuestionType.SHORT_ANSWER}
# Types graded deterministically (no LLM call)
DETERMINISTIC_TYPES = {QuestionType.MCQ, QuestionType.TRUE_FALSE}


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

    # Deterministic grading fields
    detected_answer: str = ""     # MCQ option / T or F / numerical value
    correct_answer: str = ""      # ground truth
    is_correct: Optional[bool] = None   # True/False for deterministic; None for LLM types

    # OCR
    extracted_text: str = ""
    ocr_confidence: float = 0.0
    ocr_engine: str = ""

    # Feedback (all types)
    strengths: list = field(default_factory=list)
    missing_concepts: list = field(default_factory=list)
    feedback: str = ""

    # Meta
    confidence: float = 0.0
    evaluation_time_sec: float = 0.0
    rubric_details: Optional[dict] = None
    llm_provider: str = ""        # which LLM was used
    auto_classified: bool = False  # True if question_type was auto-detected

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
    Pass question_type="auto" to auto-classify using the LLM.
    """

    LLM_WEIGHT = 0.6
    SIM_WEIGHT = 0.4

    def __init__ (
            self,
            ocr_engine: str = "trocr",
            trocr_model_path: Optional[str] = None,
            similarity_model: Optional[str] = None,
            gemini_api_key: Optional[str] = None,
            llm_weight: float = 0.6,
            similarity_weight: float = 0.4,
            use_rubric: bool = True,
    ) :
        self.LLM_WEIGHT = llm_weight
        self.SIM_WEIGHT = similarity_weight

        self.ocr = OCRModule( engine=ocr_engine, trocr_model_path=trocr_model_path )
        self.text_processor = TextProcessor()
        self.similarity_model = SemanticSimilarityModel( similarity_model )

        # Initialize LLM evaluator with API key
        if gemini_api_key :
            self.llm_evaluator = LLMEvaluator( api_key=gemini_api_key )
        else :
            # Try to get from environment
            import os
            api_key = os.getenv( "GEMINI_API_KEY" )
            self.llm_evaluator = LLMEvaluator( api_key=api_key )

        self.rubric_matcher = RubricMatcher() if use_rubric else None
        self._classifier = None  # lazy-init to avoid circular imports

        logger.info( "EvaluationEngine initialized with LLM provider: %s",
                     self.llm_evaluator._get_client().active_provider if self.llm_evaluator else "none" )

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
        student_image,
        question: str,
        teacher_answer: str = "",
        max_marks: float = 10.0,
        rubric_criteria: Optional[list] = None,
        question_type: str = "auto",
        # Deterministic grading fields
        correct_option: Optional[str] = None,    # MCQ: "A"/"B"/"C"/"D"
        correct_answer: Optional[str] = None,    # True/False / numerical value
        mcq_options: Optional[dict] = None,
        numerical_tolerance: float = 0.01,       # ±1% for numerical answers
    ) -> EvaluationResult:
        """
        Full pipeline evaluation for any question type.

        Parameters
        ----------
        question_type : "auto" | "mcq" | "true_false" | "fill_blank" |
                        "short_answer" | "numerical" | "open_ended" | "diagram"
        correct_option: letter A-E (MCQ only)
        correct_answer: expected answer string (true_false, numerical, fill_blank)
        numerical_tolerance: fractional tolerance for numerical comparison (default ±1%)
        """
        start = time.time()

        # ── Step 1: OCR ──────────────────────────────────
        logger.info("Step 1: Running OCR...")
        ocr_result: OCRResult = self.ocr.extract_text(student_image)
        student_text_raw = ocr_result.text

        if not student_text_raw.strip():
            logger.warning("OCR returned empty text.")
            return self._empty_result(
                max_marks, question_type if question_type != "auto" else "open_ended",
                "OCR returned empty text.", time.time() - start
            )

        # ── Step 2: Auto-classify if needed ──────────────
        auto_classified = False
        resolved_type   = question_type

        if question_type == "auto" or question_type == QuestionType.AUTO:
            logger.info("Step 2: Auto-classifying question type...")
            clf    = self._get_classifier()
            result = clf.classify(question)
            resolved_type   = result.question_type
            auto_classified = True
            logger.info(
                "Auto-classified as '%s' (conf=%.2f, method=%s)",
                resolved_type, result.confidence, result.method
            )
        else:
            resolved_type = QuestionType(question_type).value

        qtype = QuestionType(resolved_type)

        # ── Step 3: Route to correct pipeline ────────────
        if qtype == QuestionType.MCQ:
            res = self._evaluate_mcq(student_text_raw, ocr_result,
                                     correct_option, max_marks, start)
        elif qtype == QuestionType.TRUE_FALSE:
            res = self._evaluate_true_false(student_text_raw, ocr_result,
                                            correct_answer, max_marks, start)
        elif qtype == QuestionType.NUMERICAL:
            res = self._evaluate_numerical_or_llm(
                student_text_raw, ocr_result, question, teacher_answer,
                correct_answer, max_marks, numerical_tolerance, start
            )
        else:
            # All LLM-based types: fill_blank, short_answer, open_ended, diagram
            res = self._evaluate_with_llm(
                student_text_raw, ocr_result, question, teacher_answer,
                max_marks, rubric_criteria, qtype, start
            )

        res.auto_classified = auto_classified
        return res

    # ─────────────────────────────────────────────────────
    # MCQ
    # ─────────────────────────────────────────────────────

    def _evaluate_mcq(
        self,
        raw_text: str,
        ocr_result: OCRResult,
        correct_option: Optional[str],
        max_marks: float,
        start: float,
    ) -> EvaluationResult:
        if not correct_option:
            return self._empty_result(
                max_marks, "mcq",
                "Correct option not provided for MCQ grading.",
                time.time() - start
            )

        detected   = _extract_mcq_answer(raw_text)
        correct    = correct_option.strip().upper()
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

    def _evaluate_true_false(
        self,
        raw_text: str,
        ocr_result: OCRResult,
        correct_answer: Optional[str],
        max_marks: float,
        start: float,
    ) -> EvaluationResult:
        if not correct_answer:
            return self._empty_result(
                max_marks, "true_false",
                "Correct answer (True/False) not provided.",
                time.time() - start
            )

        detected   = _extract_true_false(raw_text)
        correct    = correct_answer.strip().upper()    # "TRUE" or "FALSE"
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
    # Numerical — try exact/tolerance first, then LLM for working
    # ─────────────────────────────────────────────────────

    def _evaluate_numerical_or_llm(
        self,
        raw_text: str,
        ocr_result: OCRResult,
        question: str,
        teacher_answer: str,
        correct_answer: Optional[str],
        max_marks: float,
        tolerance: float,
        start: float,
    ) -> EvaluationResult:
        # Try numeric exact/tolerance match first if correct_answer is a number
        if correct_answer:
            detected_num = _extract_number(raw_text)
            try:
                expected_num = float(correct_answer.strip().replace(",", ""))
                if detected_num is not None:
                    diff = abs(detected_num - expected_num)
                    within_tol = diff <= abs(expected_num * tolerance) or diff < 0.001
                    score = max_marks if within_tol else 0.0
                    is_correct = within_tol

                    if within_tol:
                        feedback = f"✅ Correct numerical answer: {detected_num} (expected {expected_num})."
                    else:
                        feedback = f"❌ Numerical answer {detected_num} is outside tolerance. Expected {expected_num}."

                    return EvaluationResult(
                        final_score=score, max_marks=max_marks, question_type="numerical",
                        detected_answer=str(detected_num), correct_answer=str(expected_num),
                        is_correct=is_correct,
                        extracted_text=raw_text,
                        ocr_confidence=round(ocr_result.confidence, 3),
                        ocr_engine=ocr_result.engine,
                        feedback=feedback,
                        confidence=0.85,
                        evaluation_time_sec=round(time.time() - start, 2),
                    )
            except ValueError:
                pass   # correct_answer is not a plain number — fall through to LLM

        # Fall through: use LLM to grade method + answer
        return self._evaluate_with_llm(
            raw_text, ocr_result, question, teacher_answer,
            max_marks, None, QuestionType.NUMERICAL, start
        )

    # ─────────────────────────────────────────────────────
    # LLM-based pipeline (open_ended, short_answer, fill_blank, diagram, numerical fallback)
    # ─────────────────────────────────────────────────────

    def _evaluate_with_llm(
        self,
        student_text_raw: str,
        ocr_result: OCRResult,
        question: str,
        teacher_answer: str,
        max_marks: float,
        rubric_criteria: Optional[list],
        qtype: QuestionType,
        start: float,
    ) -> EvaluationResult:
        # Text processing (spellcheck + normalize)
        logger.info("Text processing...")
        processed    = self.text_processor.process(student_text_raw)
        student_text = processed.cleaned

        # Semantic similarity (only for open_ended and short_answer)
        sim_score = 0.0
        if qtype in SIMILARITY_TYPES and teacher_answer:
            logger.info("Computing semantic similarity...")
            sim_result = self.similarity_model.compute_similarity(student_text, teacher_answer)
            sim_score  = sim_result.score

        # LLM evaluation
        logger.info("Running LLM evaluation (type=%s)...", qtype.value)
        rubric_labels = [r["criterion"] for r in rubric_criteria] if rubric_criteria else None
        llm_eval: LLMEvaluation = self.llm_evaluator.evaluate(
            question=question,
            teacher_answer=teacher_answer,
            student_answer=student_text,
            max_marks=max_marks,
            rubric_criteria=rubric_labels,
            question_type=qtype.value,
        )

        # Rubric matching (open_ended only)
        rubric_result = None
        rubric_score  = 0.0
        if self.rubric_matcher and rubric_criteria and qtype == QuestionType.OPEN_ENDED:
            logger.info("Rubric matching...")
            rubric_result = self.rubric_matcher.evaluate_rubric(
                student_answer=student_text,
                rubric_criteria=rubric_criteria,
                question_type="open_ended",
            )
            rubric_score = rubric_result.earned_rubric_marks

        # Hybrid score
        if qtype in SIMILARITY_TYPES:
            final_score = self._hybrid_score(llm_eval.score, sim_score, max_marks)
        else:
            final_score = llm_eval.score   # LLM score only for non-similarity types

        final_score = round(max(0.0, min(max_marks, final_score)), 2)
        elapsed     = round(time.time() - start, 2)

        return EvaluationResult(
            final_score=final_score,
            max_marks=max_marks,
            question_type=qtype.value,
            llm_score=llm_eval.score,
            similarity_score=round(sim_score, 4),
            rubric_score=rubric_score,
            extracted_text=student_text,
            ocr_confidence=round(ocr_result.confidence, 3),
            ocr_engine=ocr_result.engine,
            strengths=llm_eval.strengths,
            missing_concepts=llm_eval.missing_concepts,
            feedback=llm_eval.feedback,
            confidence=round(llm_eval.confidence, 3),
            evaluation_time_sec=elapsed,
            rubric_details=vars(rubric_result) if rubric_result else None,
            llm_provider=f"{llm_eval.provider}/{llm_eval.model}",
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
    # Helpers
    # ─────────────────────────────────────────────────────

    def _hybrid_score(self, llm_score: float, similarity: float, max_marks: float) -> float:
        return (self.LLM_WEIGHT * llm_score) + (self.SIM_WEIGHT * similarity * max_marks)

    def _empty_result(self, max_marks: float, question_type: str, reason: str, elapsed: float) -> EvaluationResult:
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
    """Extract True or False from OCR text. Returns 'TRUE' or 'FALSE' or None."""
    import re
    t = text.strip().lower()
    # Check explicit words first
    if re.search(r'\b(true|correct|yes|t)\b', t):
        if not re.search(r'\b(false|incorrect|no|f)\b', t):
            return "TRUE"
    if re.search(r'\b(false|incorrect|no|f)\b', t):
        if not re.search(r'\b(true|correct|yes|t)\b', t):
            return "FALSE"
    # Single letter T or F
    m = re.match(r'^\s*([tf])\s*$', t, re.IGNORECASE)
    if m:
        return "TRUE" if m.group(1).upper() == "T" else "FALSE"
    return None


def _extract_number(text: str) -> Optional[float]:
    """Extract the most likely numerical answer from OCR text."""
    import re
    # Find all numbers (including decimals and negatives)
    matches = re.findall(r'-?\d+(?:\.\d+)?', text.replace(",", ""))
    if not matches:
        return None
    # Return the last number found (most likely to be the final answer)
    return float(matches[-1])