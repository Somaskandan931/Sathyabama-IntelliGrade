"""
IntelliGrade-H - FastAPI Backend
REST API for the evaluation system.
"""

import os
import uuid
import logging
import time
import asyncio
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager

import aiofiles
from fastapi import (
    FastAPI, UploadFile, File, Form, HTTPException,
    Depends, BackgroundTasks
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.orm import Session
from dotenv import load_dotenv

load_dotenv()

from backend.database import get_db, init_db, Submission, Result, Question, Student, Rubric
from backend.evaluator import EvaluationEngine, EvaluationResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_FILE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "20"))

# ─────────────────────────────────────────────────────────
# Global engine instance (lazy init)
# ─────────────────────────────────────────────────────────

_engine: Optional[EvaluationEngine] = None

def get_engine() -> EvaluationEngine:
    global _engine
    if _engine is None:
        _engine = EvaluationEngine(
            ocr_engine=os.getenv("OCR_ENGINE", "trocr"),
            gemini_api_key=os.getenv("GEMINI_API_KEY"),
            llm_weight=float(os.getenv("LLM_WEIGHT", "0.6")),
            similarity_weight=float(os.getenv("SIMILARITY_WEIGHT", "0.4"))
        )
    return _engine


# ─────────────────────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    logger.info("IntelliGrade-H API started.")
    yield
    logger.info("IntelliGrade-H API shutting down.")

app = FastAPI(
    title="IntelliGrade-H API",
    description="AI-powered handwritten answer evaluation system",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────
# Pydantic Schemas
# ─────────────────────────────────────────────────────────

class RubricItem(BaseModel):
    criterion: str
    marks: float

class EvaluateRequest(BaseModel):
    submission_id: int
    question: str
    teacher_answer: str
    max_marks: float = 10.0
    rubric_criteria: Optional[List[RubricItem]] = None

class EvaluationResponse(BaseModel):
    submission_id: int
    final_score: float
    max_marks: float
    llm_score: float
    similarity_score: float
    ocr_text: str
    ocr_confidence: float
    strengths: list
    missing_concepts: list
    feedback: str
    confidence: float
    evaluation_time_sec: float
    rubric_details: Optional[dict]


# ─────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"message": "IntelliGrade-H API is running.", "version": "1.0.0"}


@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": time.time()}


# ── Upload answer sheet ───────────────────────────────────

@app.post("/upload", summary="Upload a student answer sheet image")
async def upload_answer_sheet(
    file: UploadFile = File(...),
    student_code: str = Form(...),
    db: Session = Depends(get_db)
):
    # Validate file type
    allowed_types = {"image/jpeg", "image/png", "application/pdf"}
    if file.content_type not in allowed_types:
        raise HTTPException(400, f"Unsupported file type: {file.content_type}")

    # Validate file size
    contents = await file.read()
    if len(contents) > MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(400, f"File too large. Max {MAX_FILE_MB} MB.")

    # Save file
    ext = Path(file.filename).suffix
    filename = f"{uuid.uuid4()}{ext}"
    file_path = UPLOAD_DIR / filename

    async with aiofiles.open(file_path, "wb") as f:
        await f.write(contents)

    # Get or create student
    student = db.query(Student).filter_by(student_code=student_code).first()
    if not student:
        student = Student(student_code=student_code)
        db.add(student)
        db.flush()

    # Create submission record
    submission = Submission(
        student_id=student.id,
        image_path=str(file_path),
        question_id=None
    )
    db.add(submission)
    db.commit()
    db.refresh(submission)

    return {
        "submission_id": submission.id,
        "filename": filename,
        "student_code": student_code,
        "message": "File uploaded successfully."
    }


# ── OCR only ─────────────────────────────────────────────

@app.post("/ocr/{submission_id}", summary="Run OCR on an uploaded submission")
async def run_ocr(
    submission_id: int,
    db: Session = Depends(get_db)
):
    submission = db.query(Submission).filter_by(id=submission_id).first()
    if not submission:
        raise HTTPException(404, "Submission not found.")

    engine = get_engine()
    ocr_result = engine.ocr.extract_text(submission.image_path)

    # Update submission
    submission.extracted_text = ocr_result.text
    submission.ocr_confidence = ocr_result.confidence
    db.commit()

    return {
        "submission_id": submission_id,
        "extracted_text": ocr_result.text,
        "confidence": ocr_result.confidence,
        "engine": ocr_result.engine
    }


# ── Full Evaluation ───────────────────────────────────────

@app.post("/evaluate", response_model=EvaluationResponse, summary="Evaluate a submission")
async def evaluate_submission(
    request: EvaluateRequest,
    db: Session = Depends(get_db)
):
    submission = db.query(Submission).filter_by(id=request.submission_id).first()
    if not submission:
        raise HTTPException(404, "Submission not found.")

    rubric = None
    if request.rubric_criteria:
        rubric = [{"criterion": r.criterion, "marks": r.marks}
                  for r in request.rubric_criteria]

    engine = get_engine()

    try:
        result: EvaluationResult = engine.evaluate(
            student_image=submission.image_path,
            question=request.question,
            teacher_answer=request.teacher_answer,
            max_marks=request.max_marks,
            rubric_criteria=rubric
        )
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        raise HTTPException(500, f"Evaluation failed: {str(e)}")

    # Save result to DB
    import json
    db_result = Result(
        submission_id=submission.id,
        final_score=result.final_score,
        llm_score=result.llm_score,
        similarity_score=result.similarity_score,
        strengths=json.dumps(result.strengths),
        missing_concepts=json.dumps(result.missing_concepts),
        feedback=result.feedback,
        confidence=result.confidence,
        evaluation_time_sec=result.evaluation_time_sec
    )
    db.add(db_result)

    # Update submission
    submission.extracted_text = result.extracted_text
    submission.ocr_confidence = result.ocr_confidence

    db.commit()
    db.refresh(db_result)

    return EvaluationResponse(
        submission_id=submission.id,
        final_score=result.final_score,
        max_marks=result.max_marks,
        llm_score=result.llm_score,
        similarity_score=result.similarity_score,
        ocr_text=result.extracted_text,
        ocr_confidence=result.ocr_confidence,
        strengths=result.strengths,
        missing_concepts=result.missing_concepts,
        feedback=result.feedback,
        confidence=result.confidence,
        evaluation_time_sec=result.evaluation_time_sec,
        rubric_details=result.rubric_details
    )


# ── Get Result ────────────────────────────────────────────

@app.get("/result/{submission_id}", summary="Get evaluation result for a submission")
async def get_result(
    submission_id: int,
    db: Session = Depends(get_db)
):
    submission = db.query(Submission).filter_by(id=submission_id).first()
    if not submission:
        raise HTTPException(404, "Submission not found.")

    result = db.query(Result).filter_by(submission_id=submission_id).first()
    if not result:
        raise HTTPException(404, "Result not yet available. Run /evaluate first.")

    import json
    return {
        "submission_id": submission_id,
        "final_score": result.final_score,
        "llm_score": result.llm_score,
        "similarity_score": result.similarity_score,
        "strengths": json.loads(result.strengths or "[]"),
        "missing_concepts": json.loads(result.missing_concepts or "[]"),
        "feedback": result.feedback,
        "confidence": result.confidence,
        "evaluation_time_sec": result.evaluation_time_sec,
        "created_at": result.created_at.isoformat()
    }


# ── Rubric Upload ─────────────────────────────────────────

class RubricUploadRequest(BaseModel):
    question_id: int
    criteria: List[RubricItem]

@app.post("/rubric", summary="Upload rubric criteria for a question")
async def upload_rubric(
    request: RubricUploadRequest,
    db: Session = Depends(get_db)
):
    question = db.query(Question).filter_by(id=request.question_id).first()
    if not question:
        raise HTTPException(404, "Question not found.")

    # Remove old rubric
    db.query(Rubric).filter_by(question_id=request.question_id).delete()

    for item in request.criteria:
        rubric = Rubric(
            question_id=request.question_id,
            criterion=item.criterion,
            marks=item.marks
        )
        db.add(rubric)

    db.commit()
    return {"message": f"Rubric updated with {len(request.criteria)} criteria."}


# ── Stats ─────────────────────────────────────────────────

@app.get("/stats", summary="Get overall system statistics")
async def get_stats(db: Session = Depends(get_db)):
    total_submissions = db.query(Submission).count()
    evaluated = db.query(Result).count()
    results = db.query(Result).all()

    avg_score = 0.0
    avg_time = 0.0
    if results:
        avg_score = sum(r.final_score for r in results) / len(results)
        avg_time = sum(r.evaluation_time_sec for r in results if r.evaluation_time_sec) / len(results)

    return {
        "total_submissions": total_submissions,
        "evaluated": evaluated,
        "average_score": round(avg_score, 2),
        "average_evaluation_time_sec": round(avg_time, 2)
    }


# ─────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.api:app", host="0.0.0.0", port=8000, reload=True)
