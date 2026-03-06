"""
backend/schemas.py
Pydantic v2 request / response schemas for the FastAPI layer.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ── Question ──────────────────────────────────────────────────────────────────
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


# ── Evaluation / Result ───────────────────────────────────────────────────────
class EvaluationResponse(BaseModel):
    submission_id: uuid.UUID
    question_id: uuid.UUID
    student_id: uuid.UUID

    # Scores
    llm_score: float = Field(..., description="Score from Gemini LLM")
    similarity_score: float = Field(..., description="Cosine similarity [0,1]")
    final_score: float = Field(..., description="Hybrid weighted score")
    max_marks: float

    # Qualitative
    strengths: list[str] = []
    missing_concepts: list[str] = []
    feedback: str = ""
    rubric_coverage: dict[str, float] = {}

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
