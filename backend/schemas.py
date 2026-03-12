"""
backend/schemas.py
Pydantic v2 request / response schemas for the FastAPI layer.

v3.0 — Added ExamPaper, ExamPart, ExamQuestion, and paper-based evaluation schemas.
       All legacy schemas preserved unchanged for backward compatibility.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional, List

from pydantic import BaseModel, Field


# ── Question (legacy) ─────────────────────────────────────────────────────────
class QuestionCreate(BaseModel):
    paper_id: str = Field(..., description="Exam / paper identifier")
    question_text: str = Field(..., min_length=5)
    teacher_answer: str = Field(..., min_length=5)
    max_marks: float = Field(10.0, gt=0)


class QuestionRead(QuestionCreate):
    id: uuid.UUID
    created_at: datetime

    class Config:
        from_attributes = True


# ── Rubric ────────────────────────────────────────────────────────────────────
class RubricCreate(BaseModel):
    question_id: uuid.UUID
    element: str = Field(..., min_length=2, description="e.g. 'Definition'")
    max_marks: float = Field(2.0, gt=0)


class RubricRead(RubricCreate):
    id: uuid.UUID
    created_at: datetime

    class Config:
        from_attributes = True


# ── Student ───────────────────────────────────────────────────────────────────
class StudentCreate(BaseModel):
    name: str = Field(..., min_length=2)
    roll_number: str = Field(..., min_length=1)
    email: Optional[str] = None


class StudentRead(StudentCreate):
    id: uuid.UUID
    created_at: datetime

    class Config:
        from_attributes = True


# ── Submission ────────────────────────────────────────────────────────────────
class SubmissionRead(BaseModel):
    id: uuid.UUID
    student_id: uuid.UUID
    question_id: uuid.UUID
    extracted_text: Optional[str] = None
    submitted_at: datetime

    class Config:
        from_attributes = True


# ── Evaluation / Result (legacy + extended) ───────────────────────────────────
class EvaluationResponse(BaseModel):
    submission_id: uuid.UUID
    question_id: uuid.UUID
    student_id: uuid.UUID

    # Dynamic paper fields (None for legacy evaluations)
    exam_paper_id: Optional[str] = None
    exam_question_id: Optional[uuid.UUID] = None
    question_number: Optional[int] = None

    # Scores
    llm_score: float = Field(..., description="Score from LLM")
    similarity_score: float = Field(..., description="Cosine similarity [0,1]")
    final_score: float = Field(..., description="Hybrid weighted score")
    max_marks: float

    # Qualitative
    strengths: list[str] = []
    missing_concepts: list[str] = []
    feedback: str = ""
    rubric_coverage: dict[str, float] = {}

    # MCQ fields
    question_type: Optional[str] = None
    mcq_detected_answer: Optional[str] = None
    mcq_correct_answer: Optional[str] = None
    mcq_correct: Optional[bool] = None

    # OCR
    ocr_text: Optional[str] = None
    ocr_confidence: Optional[float] = None

    # Meta
    confidence: float
    word_count: int
    evaluation_time_sec: float
    percentage: float


class ResultRead(BaseModel):
    id: uuid.UUID
    submission_id: uuid.UUID
    llm_score: Optional[float]
    similarity_score: Optional[float]
    final_score: Optional[float]
    max_marks: Optional[float]
    strengths: Optional[str]
    missing_concepts: Optional[str]
    feedback: Optional[str]
    confidence: Optional[float]
    evaluation_time_sec: Optional[float]
    created_at: datetime

    # Dynamic paper fields
    exam_paper_id: Optional[str] = None
    exam_question_id: Optional[str] = None
    question_number: Optional[int] = None

    class Config:
        from_attributes = True


# ── OCR only ──────────────────────────────────────────────────────────────────
class OCRResponse(BaseModel):
    extracted_text: str
    char_count: int
    processing_time_sec: float


# ── Health ────────────────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    models_loaded: dict[str, bool] = {}


# ═════════════════════════════════════════════════════════════════════════════
# v3.0 — Dynamic Exam Paper schemas
# ═════════════════════════════════════════════════════════════════════════════

# ── ExamPart ──────────────────────────────────────────────────────────────────
class ExamPartSchema(BaseModel):
    """One part of an exam paper (e.g. Part-A, Part-B)."""
    part_name: str
    marks_per_question: float
    num_questions: int
    instructions: Optional[str] = None

    class Config:
        from_attributes = True


# ── ExamQuestion ──────────────────────────────────────────────────────────────
class ExamQuestionSchema(BaseModel):
    """A single question as parsed from the question paper PDF."""
    id: Optional[uuid.UUID] = None
    question_number: int
    question_text: str
    marks: float                          # always from PDF, never hardcoded
    part_name: Optional[str] = None
    question_type: Optional[str] = None  # mcq / open_ended / short_answer / etc.
    is_or_option: bool = False            # True for OR-alternative questions
    teacher_answer: Optional[str] = None  # populated after answer key upload
    correct_option: Optional[str] = None  # MCQ: A/B/C/D

    class Config:
        from_attributes = True


# ── ExamPaper ─────────────────────────────────────────────────────────────────
class ExamPaperSchema(BaseModel):
    """Full exam paper as stored/returned from the database."""
    paper_id: str                         # slug e.g. "S11BLH41_CAE1_Set-A"
    course_code: Optional[str] = None
    course_name: Optional[str] = None
    exam_name: Optional[str] = None
    total_marks: Optional[float] = None
    duration_hours: Optional[float] = None
    exam_date: Optional[str] = None
    batch: Optional[str] = None
    programme: Optional[str] = None
    semester: Optional[str] = None
    set_name: Optional[str] = None
    parsed_at: Optional[datetime] = None
    parts: List[ExamPartSchema] = []
    questions: List[ExamQuestionSchema] = []

    class Config:
        from_attributes = True


class ExamPaperListItem(BaseModel):
    """Lightweight row for listing all papers."""
    paper_id: str
    course_code: Optional[str] = None
    course_name: Optional[str] = None
    exam_name: Optional[str] = None
    total_marks: Optional[float] = None
    set_name: Optional[str] = None
    parsed_at: Optional[datetime] = None
    num_questions: int = 0

    class Config:
        from_attributes = True


# ── Paper Upload Response ─────────────────────────────────────────────────────
class PaperUploadResponse(BaseModel):
    paper_id: str
    status: str                           # "created" | "updated"
    course_code: Optional[str] = None
    course_name: Optional[str] = None
    exam_name: Optional[str] = None
    total_marks: Optional[float] = None
    set_name: Optional[str] = None
    num_parts: int = 0
    num_questions: int = 0
    parts: List[ExamPartSchema] = []
    questions: List[ExamQuestionSchema] = []
    parse_warnings: List[str] = []       # non-fatal issues during parsing


# ── Answer Key Upload Response ────────────────────────────────────────────────
class AnswerKeyUploadResponse(BaseModel):
    paper_id: str
    questions_updated: int
    status: str = "ok"


# ── Paper-based Evaluate Request ──────────────────────────────────────────────
class EvaluateByPaperRequest(BaseModel):
    """
    Evaluate a student submission for a specific question from a known exam paper.
    The system loads max_marks, question_type, and teacher_answer from the DB.
    No hardcoding needed.
    """
    submission_id: uuid.UUID = Field(..., description="Submission ID from /upload")
    exam_paper_id: str = Field(..., description="paper_id slug from /papers")
    question_number: int = Field(..., ge=1, description="Question number in the paper")


# ── Legacy Evaluate Request (unchanged) ──────────────────────────────────────
class EvaluateRequest(BaseModel):
    submission_id: uuid.UUID
    question: str
    teacher_answer: str
    max_marks: Optional[float] = None
    question_type: str = "auto"
    exam_question_id: Optional[uuid.UUID] = None   # v3: if set, overrides max_marks from DB