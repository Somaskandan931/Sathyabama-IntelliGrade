"""
IntelliGrade-H: Hybrid Scoring Engine
Combines LLM score + semantic similarity + rubric matching into a final mark.

Formula (from documentation):
    Final Score = 0.6 × LLM_Score + 0.4 × Similarity_Score × Max_Marks
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from .evaluator import LLMEvaluator, EvaluationResult
from .similarity import SemanticSimilarity
from .rubric_matcher import ZeroShotRubricMatcher, RubricItem, RubricResult, RubricBuilder
from .text_processor import TextProcessor
from .ocr_module import HandwritingOCR

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Result Container
# ─────────────────────────────────────────────

@dataclass
class GradingReport:
    """Complete grading output for one student answer."""
    student_id: str
    question: str
    max_marks: float

    # Component scores
    llm_score: float
    similarity_score: float
    rubric_score: float
    rubric_max: float

    # Final
    final_score: float
    confidence: float

    # Qualitative feedback
    strengths: List[str]
    missing_concepts: List[str]
    improvement_suggestions: List[str]
    narrative_feedback: str

    # Optional extras
    extracted_text: str = ""
    rubric_details: List[RubricResult] = field(default_factory=list)
    error: Optional[str] = None

    # ── display helpers ──────────────────────

    def summary(self) -> str:
        lines = [
            f"━━━ Grading Report ━━━",
            f"Student : {self.student_id}",
            f"Score   : {self.final_score:.2f} / {self.max_marks}",
            f"         (LLM={self.llm_score:.2f}  Sim={self.similarity_score:.2f}  "
            f"Rubric={self.rubric_score:.2f}/{self.rubric_max:.2f})",
            f"Confidence: {self.confidence:.0%}",
            "",
            "Strengths:",
            *[f"  • {s}" for s in self.strengths],
            "",
            "Missing Concepts:",
            *[f"  • {m}" for m in self.missing_concepts],
            "",
            "Suggestions:",
            *[f"  • {sg}" for sg in self.improvement_suggestions],
            "",
            f"Feedback: {self.narrative_feedback}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "student_id": self.student_id,
            "question": self.question,
            "max_marks": self.max_marks,
            "final_score": self.final_score,
            "llm_score": self.llm_score,
            "similarity_score": self.similarity_score,
            "rubric_score": self.rubric_score,
            "rubric_max": self.rubric_max,
            "confidence": self.confidence,
            "strengths": self.strengths,
            "missing_concepts": self.missing_concepts,
            "improvement_suggestions": self.improvement_suggestions,
            "narrative_feedback": self.narrative_feedback,
            "extracted_text": self.extracted_text,
            "rubric_details": [
                {
                    "key": r.key,
                    "present": r.present,
                    "marks_awarded": r.marks_awarded,
                    "max_marks": r.max_marks,
                }
                for r in self.rubric_details
            ],
            "error": self.error,
        }


# ─────────────────────────────────────────────
# Hybrid Scoring Engine
# ─────────────────────────────────────────────

class HybridScoringEngine:
    """
    Orchestrates all evaluation components and produces a GradingReport.

    Weights (configurable):
        llm_weight        : 0.60  (LLM score contribution)
        similarity_weight : 0.40  (semantic similarity contribution)
        rubric_bonus      : optional bonus from rubric matching
    """

    def __init__(
        self,
        llm_weight: float = 0.60,
        similarity_weight: float = 0.40,
        use_rubric: bool = True,
        llm_api_key: str = None,
        ocr_model_path: str = None,
    ):
        if abs(llm_weight + similarity_weight - 1.0) > 1e-6:
            raise ValueError("llm_weight + similarity_weight must sum to 1.0")

        self.llm_weight = llm_weight
        self.similarity_weight = similarity_weight
        self.use_rubric = use_rubric

        logger.info("Initialising HybridScoringEngine components...")
        self.text_processor = TextProcessor()
        self.ocr = HandwritingOCR(model_path=ocr_model_path)
        self.similarity = SemanticSimilarity()
        self.llm = LLMEvaluator(api_key=llm_api_key)
        self.rubric_matcher = ZeroShotRubricMatcher() if use_rubric else None

    # ── public API ───────────────────────────

    def grade_text(
        self,
        question: str,
        teacher_answer: str,
        student_answer: str,
        max_marks: float = 10.0,
        rubric_items: Optional[List[RubricItem]] = None,
        student_id: str = "unknown",
    ) -> GradingReport:
        """
        Grade when the student answer is already extracted text.
        """
        # 1. Clean text
        clean_student = self.text_processor.process(student_answer)
        clean_teacher = self.text_processor.process(teacher_answer)

        # 2. Semantic similarity
        sim_score = self.similarity.compute(clean_teacher, clean_student)

        # 3. LLM evaluation
        rubric_str = self._rubric_items_to_str(rubric_items) if rubric_items else \
            "Assess concept understanding, accuracy, explanation depth, structure, and examples."
        llm_result: EvaluationResult = self.llm.evaluate(
            question=question,
            teacher_answer=clean_teacher,
            student_answer=clean_student,
            max_marks=max_marks,
            rubric=rubric_str,
        )

        # 4. Rubric matching
        rubric_score, rubric_max, rubric_details = 0.0, 0.0, []
        if self.use_rubric and self.rubric_matcher:
            items = rubric_items or RubricBuilder.default_rubric(max_marks)
            rubric_score, rubric_max, rubric_details = \
                self.rubric_matcher.total_rubric_score(clean_student, items)

        # 5. Hybrid final score
        final_score = self._compute_final(
            llm_score=llm_result.llm_score,
            sim_score=sim_score,
            max_marks=max_marks,
        )

        return GradingReport(
            student_id=student_id,
            question=question,
            max_marks=max_marks,
            llm_score=round(llm_result.llm_score, 2),
            similarity_score=round(sim_score, 4),
            rubric_score=round(rubric_score, 2),
            rubric_max=round(rubric_max, 2),
            final_score=round(final_score, 2),
            confidence=round(llm_result.confidence, 4),
            strengths=llm_result.strengths,
            missing_concepts=llm_result.missing_concepts,
            improvement_suggestions=llm_result.improvement_suggestions,
            narrative_feedback=llm_result.raw_feedback,
            extracted_text=student_answer,
            rubric_details=rubric_details,
            error=llm_result.error,
        )

    def grade_image(
        self,
        question: str,
        teacher_answer: str,
        student_image,          # path, bytes, or PIL Image
        max_marks: float = 10.0,
        rubric_items: Optional[List[RubricItem]] = None,
        student_id: str = "unknown",
    ) -> GradingReport:
        """
        Grade from a raw handwritten image — runs OCR first.
        """
        logger.info(f"Extracting text from image for student: {student_id}")
        extracted_text = self.ocr.extract_text(student_image)
        report = self.grade_text(
            question=question,
            teacher_answer=teacher_answer,
            student_answer=extracted_text,
            max_marks=max_marks,
            rubric_items=rubric_items,
            student_id=student_id,
        )
        report.extracted_text = extracted_text
        return report

    # ── private helpers ──────────────────────

    def _compute_final(self, llm_score: float, sim_score: float, max_marks: float) -> float:
        """
        Final Score = LLM_weight × LLM_Score + Sim_weight × Sim_Score × Max_Marks
        Clamps output to [0, max_marks].
        """
        raw = self.llm_weight * llm_score + self.similarity_weight * sim_score * max_marks
        return max(0.0, min(raw, max_marks))

    @staticmethod
    def _rubric_items_to_str(items: List[RubricItem]) -> str:
        lines = ["Rubric Criteria:"]
        for item in items:
            lines.append(f"  - {item.key} ({item.max_marks} marks): {item.description}")
        return "\n".join(lines)


# ─────────────────────────────────────────────
# Batch Grader
# ─────────────────────────────────────────────

class BatchGrader:
    """Grade multiple submissions efficiently."""

    def __init__(self, engine: HybridScoringEngine):
        self.engine = engine

    def grade_batch(
        self,
        question: str,
        teacher_answer: str,
        submissions: List[dict],   # [{"student_id": ..., "answer_text": ...}]
        max_marks: float = 10.0,
    ) -> List[GradingReport]:
        """
        submissions: list of dicts with keys 'student_id' and 'answer_text' (or 'image_path').
        """
        reports = []
        for i, sub in enumerate(submissions, 1):
            sid = sub.get("student_id", f"student_{i}")
            logger.info(f"Grading {i}/{len(submissions)} — student: {sid}")
            try:
                if "answer_text" in sub:
                    report = self.engine.grade_text(
                        question=question,
                        teacher_answer=teacher_answer,
                        student_answer=sub["answer_text"],
                        max_marks=max_marks,
                        student_id=sid,
                    )
                else:
                    report = self.engine.grade_image(
                        question=question,
                        teacher_answer=teacher_answer,
                        student_image=sub["image_path"],
                        max_marks=max_marks,
                        student_id=sid,
                    )
            except Exception as exc:
                logger.error(f"Failed to grade {sid}: {exc}")
                report = GradingReport(
                    student_id=sid,
                    question=question,
                    max_marks=max_marks,
                    llm_score=0, similarity_score=0,
                    rubric_score=0, rubric_max=0,
                    final_score=0, confidence=0,
                    strengths=[], missing_concepts=[],
                    improvement_suggestions=[],
                    narrative_feedback="Grading failed.",
                    error=str(exc),
                )
            reports.append(report)
        return reports
