"""
IntelliGrade-H - Evaluation Engine
Orchestrates OCR → Text Processing → Similarity → LLM → Rubric → Hybrid Score.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional

from backend.ocr_module import OCRModule, OCRResult
from backend.text_processor import TextProcessor, ProcessedText
from backend.similarity import SemanticSimilarityModel, SimilarityResult
from backend.llm_evaluator import LLMEvaluator, LLMEvaluation
from backend.rubric_matcher import RubricMatcher, RubricResult

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    # Scores
    final_score: float
    max_marks: float
    llm_score: float
    similarity_score: float     # 0–1 cosine similarity
    rubric_score: float         # marks from rubric (optional)

    # OCR
    extracted_text: str
    ocr_confidence: float
    ocr_engine: str

    # Feedback
    strengths: list = field(default_factory=list)
    missing_concepts: list = field(default_factory=list)
    feedback: str = ""

    # Meta
    confidence: float = 0.0
    evaluation_time_sec: float = 0.0
    rubric_details: Optional[dict] = None


class EvaluationEngine:
    """
    Main pipeline for IntelliGrade-H.

    Pipeline:
    image → OCR → text processing → semantic similarity
           → LLM evaluation → rubric matching → hybrid score
    """

    # Hybrid scoring weights
    LLM_WEIGHT = 0.6
    SIM_WEIGHT = 0.4

    def __init__(
        self,
        ocr_engine: str = "trocr",
        trocr_model_path: Optional[str] = None,
        similarity_model: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        llm_weight: float = 0.6,
        similarity_weight: float = 0.4,
        use_rubric: bool = True
    ):
        self.LLM_WEIGHT = llm_weight
        self.SIM_WEIGHT = similarity_weight

        self.ocr = OCRModule(engine=ocr_engine, trocr_model_path=trocr_model_path)
        self.text_processor = TextProcessor()
        self.similarity_model = SemanticSimilarityModel(similarity_model)
        self.llm_evaluator = LLMEvaluator(api_key=gemini_api_key)
        self.rubric_matcher = RubricMatcher() if use_rubric else None

        logger.info("EvaluationEngine initialized.")

    # ─────────────────────────────────────────────────────────
    # Main Entry Point
    # ─────────────────────────────────────────────────────────

    def evaluate(
        self,
        student_image,          # file path, bytes, or PIL Image
        question: str,
        teacher_answer: str,
        max_marks: float = 10.0,
        rubric_criteria: Optional[list] = None
    ) -> EvaluationResult:
        """
        Full pipeline evaluation.

        rubric_criteria (optional):
            list of {"criterion": str, "marks": float}
        """
        start = time.time()

        # ── Step 1: OCR ──────────────────────────────────────
        logger.info("Step 1: Running OCR...")
        ocr_result: OCRResult = self.ocr.extract_text(student_image)
        student_text_raw = ocr_result.text

        if not student_text_raw.strip():
            logger.warning("OCR returned empty text.")
            return self._empty_result(max_marks, "OCR returned empty text.", time.time() - start)

        # ── Step 2: Text Processing ──────────────────────────
        logger.info("Step 2: Text processing...")
        processed: ProcessedText = self.text_processor.process(student_text_raw)
        student_text = processed.cleaned

        # ── Step 3: Semantic Similarity ──────────────────────
        logger.info("Step 3: Computing semantic similarity...")
        sim_result: SimilarityResult = self.similarity_model.compute_similarity(
            student_text, teacher_answer
        )

        # ── Step 4: LLM Evaluation ───────────────────────────
        logger.info("Step 4: LLM evaluation...")
        rubric_labels = [r["criterion"] for r in rubric_criteria] if rubric_criteria else None
        llm_eval: LLMEvaluation = self.llm_evaluator.evaluate(
            question=question,
            teacher_answer=teacher_answer,
            student_answer=student_text,
            max_marks=max_marks,
            rubric_criteria=rubric_labels
        )

        # ── Step 5: Rubric Matching ──────────────────────────
        rubric_result = None
        rubric_score = 0.0
        if self.rubric_matcher and rubric_criteria:
            logger.info("Step 5: Rubric matching...")
            rubric_result: RubricResult = self.rubric_matcher.evaluate_rubric(
                student_answer=student_text,
                rubric_criteria=rubric_criteria
            )
            rubric_score = rubric_result.earned_rubric_marks

        # ── Step 6: Hybrid Scoring ───────────────────────────
        logger.info("Step 6: Computing hybrid score...")
        final_score = self._hybrid_score(
            llm_score=llm_eval.score,
            similarity=sim_result.score,
            max_marks=max_marks
        )
        final_score = round(max(0.0, min(max_marks, final_score)), 2)

        elapsed = round(time.time() - start, 2)
        logger.info(f"Evaluation complete in {elapsed}s. Score: {final_score}/{max_marks}")

        return EvaluationResult(
            final_score=final_score,
            max_marks=max_marks,
            llm_score=llm_eval.score,
            similarity_score=round(sim_result.score, 4),
            rubric_score=rubric_score,
            extracted_text=student_text,
            ocr_confidence=round(ocr_result.confidence, 3),
            ocr_engine=ocr_result.engine,
            strengths=llm_eval.strengths,
            missing_concepts=llm_eval.missing_concepts,
            feedback=llm_eval.feedback,
            confidence=round(llm_eval.confidence, 3),
            evaluation_time_sec=elapsed,
            rubric_details=vars(rubric_result) if rubric_result else None
        )

    # ─────────────────────────────────────────────────────────
    # Batch Evaluation
    # ─────────────────────────────────────────────────────────

    def evaluate_batch(
        self,
        submissions: list,
        question: str,
        teacher_answer: str,
        max_marks: float = 10.0,
        rubric_criteria: Optional[list] = None
    ) -> list:
        """
        submissions: list of {"student_id": str, "image": image_input}
        Returns list of {"student_id": str, "result": EvaluationResult}
        """
        results = []
        for i, sub in enumerate(submissions):
            logger.info(f"Evaluating submission {i+1}/{len(submissions)} — student: {sub['student_id']}")
            try:
                result = self.evaluate(
                    student_image=sub["image"],
                    question=question,
                    teacher_answer=teacher_answer,
                    max_marks=max_marks,
                    rubric_criteria=rubric_criteria
                )
                results.append({"student_id": sub["student_id"], "result": result})
            except Exception as e:
                logger.error(f"Failed for student {sub['student_id']}: {e}")
                results.append({
                    "student_id": sub["student_id"],
                    "result": self._empty_result(max_marks, str(e), 0.0)
                })
        return results

    # ─────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────

    def _hybrid_score(
        self,
        llm_score: float,
        similarity: float,
        max_marks: float
    ) -> float:
        """
        Formula:
          Final = (LLM_WEIGHT × llm_score) + (SIM_WEIGHT × similarity × max_marks)
        """
        return (self.LLM_WEIGHT * llm_score) + (self.SIM_WEIGHT * similarity * max_marks)

    def _empty_result(self, max_marks: float, reason: str, elapsed: float) -> EvaluationResult:
        return EvaluationResult(
            final_score=0.0,
            max_marks=max_marks,
            llm_score=0.0,
            similarity_score=0.0,
            rubric_score=0.0,
            extracted_text="",
            ocr_confidence=0.0,
            ocr_engine="none",
            feedback=reason,
            evaluation_time_sec=elapsed
        )
