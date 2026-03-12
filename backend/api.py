"""
IntelliGrade-H — FastAPI Backend (v5.0)
========================================
All parsing logic has been moved to dedicated modules:

  backend.question_paper_parser  — question paper PDF  → ParsedExamPaper
  backend.answer_key_parser      — answer key PDF       → {q_num: answer_text}
  backend.student_answer_parser  — student booklet PDF  → ParsedStudentBooklet

Endpoints:
  POST /paper/upload                          — upload question paper PDF
  POST /answer-key/upload                     — upload answer key PDF
  GET  /papers                                — list all exam papers
  GET  /paper/{paper_id}                      — get full paper structure
  PATCH /paper/{paper_id}/question/{q_num}    — manually update a question

  POST /evaluate/paper                        — evaluate against a paper question
  POST /upload                                — upload student answer sheet (legacy)
  POST /ocr/{id}                              — run OCR on a submission
  POST /evaluate                              — evaluate a submission (manual)
  GET  /result/{id}                           — get evaluation result

  POST /booklet/upload                        — upload student answer booklet
  POST /booklet/{id}/evaluate                 — evaluate all answers in booklet
  GET  /booklets                              — list all booklets
  GET  /booklet/{id}                          — get booklet answers
  PATCH /booklet/{id}/answer/{q_num}          — manually correct a student answer

  POST /rubric                                — upload rubric criteria
  GET  /stats                                 — system statistics
  GET  /metrics                               — AI scoring accuracy metrics
  GET  /metrics/compute                       — ad-hoc metric computation
  GET  /submissions                           — list all submissions
"""

import os
import re
import uuid
import json
import logging
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager

import aiofiles
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from sqlalchemy.orm import Session
from dotenv import load_dotenv

load_dotenv()

from backend.config import get_settings as _get_settings

from backend.database import (
    get_db, init_db,
    Submission, Result, Question, Student, Rubric, MetricsSnapshot, SessionLocal,
    ExamPaper, ExamPart, ExamQuestion,
    StudentBooklet, StudentAnswerText,
)
from backend.evaluator import EvaluationEngine, EvaluationResult
from backend.metrics import compute_metrics, compute_mcq_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_BASE_DIR  = Path(__file__).resolve().parent.parent
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", str(_BASE_DIR / "uploads"))).resolve()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_FILE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "20"))

_executor = ThreadPoolExecutor(
    max_workers=int(os.getenv("OCR_WORKERS", "2")),  # 2 is safer — avoids cloud rate limits
    thread_name_prefix="ocr-worker",
)


# ─────────────────────────────────────────────────────────
# In-memory metrics cache
# ─────────────────────────────────────────────────────────

_metrics_cache: dict = {
    "open_ended":      None,
    "mcq":             None,
    "last_updated":    None,
    "total_evaluated": 0,
}


# ─────────────────────────────────────────────────────────
# Evaluation engine — lazy singleton
# ─────────────────────────────────────────────────────────

_engine: Optional[EvaluationEngine] = None
_engine_mutex = __import__("threading").Lock()


def get_engine() -> EvaluationEngine:
    global _engine
    if _engine is None:
        with _engine_mutex:
            if _engine is None:
                _engine = EvaluationEngine(
                    ocr_engine        = os.getenv("OCR_ENGINE", "trocr"),
                    trocr_model_path  = os.getenv("TROCR_MODEL_PATH", "microsoft/trocr-small-handwritten"),
                    similarity_model  = os.getenv("SBERT_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
                    llm_weight        = float(os.getenv("LLM_WEIGHT",        "0.40")),
                    similarity_weight = float(os.getenv("SIMILARITY_WEIGHT", "0.25")),
                    rubric_weight     = float(os.getenv("RUBRIC_WEIGHT",     "0.20")),
                    keyword_weight    = float(os.getenv("KEYWORD_WEIGHT",    "0.10")),
                    length_weight     = float(os.getenv("LENGTH_WEIGHT",     "0.05")),
                )
    return _engine


def _warm_ocr():
    """Pre-warm all available OCR engines (PaddleOCR → Tesseract → TrOCR)."""
    try:
        from backend.ocr_module import OCRModule
        OCRModule().warmup()
        logger.info("✅ OCR engines warmed up.")
    except Exception as e:
        logger.warning("OCR warm-up failed (will load on first request): %s", e)


# ─────────────────────────────────────────────────────────
# Background metrics
# ─────────────────────────────────────────────────────────

def _recompute_metrics_background(db_session_factory):
    db: Session = db_session_factory()
    try:
        results = db.query(Result).all()
        if not results:
            return

        open_ai, open_teacher     = [], []
        mcq_pred, mcq_correct_lst = [], []

        for r in results:
            if r.question_type in ("open_ended", "short_answer", "diagram"):
                if r.final_score is not None and r.max_marks:
                    open_ai.append(r.final_score)
                    # NOTE: No real teacher ground-truth scores are stored per-result.
                    # We use similarity_score × max_marks as a proxy. This means
                    # Pearson r / MAE / Kappa reflect AI self-consistency, NOT
                    # accuracy against a human rater. Treat these as indicative only.
                    teacher_proxy = r.max_marks * (r.similarity_score or 0.5)
                    open_teacher.append(teacher_proxy)
            elif r.question_type == "mcq":
                if r.mcq_detected_answer and r.mcq_correct_answer:
                    mcq_pred.append(r.mcq_detected_answer)
                    mcq_correct_lst.append(r.mcq_correct_answer)

        open_ended_dict = mcq_dict = None

        if len(open_ai) >= 2:
            report = compute_metrics(open_ai, open_teacher, question_type="open_ended")
            open_ended_dict = {
                "n_samples":                report.n_samples,
                "mae":                      report.mae,
                "pearson_r":                report.pearson_r,
                "cohen_kappa":              report.cohen_kappa,
                "accuracy_within_1_mark":   report.accuracy_within_1,
                "accuracy_within_0_5_mark": report.accuracy_within_0_5,
                "mean_ai_score":            report.mean_ai_score,
                "mean_teacher_score":       report.mean_teacher_score,
            }
            _metrics_cache["open_ended"] = open_ended_dict

        if mcq_pred:
            mcq_report = compute_mcq_metrics(mcq_pred, mcq_correct_lst)
            mcq_dict = {
                "n_samples":    mcq_report.n_samples,
                "accuracy":     mcq_report.mcq_accuracy,
                "accuracy_pct": round(mcq_report.mcq_accuracy * 100, 1),
                "n_correct":    mcq_report.mcq_n_correct,
                "n_wrong":      mcq_report.mcq_n_wrong,
            }
            _metrics_cache["mcq"] = mcq_dict

        now_str = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _metrics_cache["last_updated"]    = now_str
        _metrics_cache["total_evaluated"] = len(results)

        MetricsSnapshot.upsert(
            db=db,
            open_ended_dict=open_ended_dict,
            mcq_dict=mcq_dict,
            total_evaluated=len(results),
        )
    except Exception as e:
        logger.error("Background metrics failed: %s", e, exc_info=True)
    finally:
        db.close()


# ─────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────

def _load_metrics_from_db():
    db: Session = SessionLocal()
    try:
        snapshot = MetricsSnapshot.load(db)
        if snapshot is None:
            return
        _metrics_cache["open_ended"]      = snapshot.open_ended
        _metrics_cache["mcq"]             = snapshot.mcq
        _metrics_cache["total_evaluated"] = snapshot.total_evaluated
        _metrics_cache["last_updated"]    = snapshot.computed_at.strftime("%Y-%m-%dT%H:%M:%SZ")
        logger.info("✅ Metrics loaded from DB (%d results).", snapshot.total_evaluated)
    except Exception as e:
        logger.warning("Could not load metrics snapshot: %s", e)
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    _load_metrics_from_db()
    cfg = _get_settings()
    logger.info(
        "Config: model=%s | sbert=%s | weights=LLM%.2f/SIM%.2f/RUB%.2f/KW%.2f/LEN%.2f",
        cfg.groq_model, cfg.sbert_model,
        cfg.llm_weight, cfg.similarity_weight, cfg.rubric_weight,
        cfg.keyword_weight, cfg.length_weight,
    )
    loop = asyncio.get_running_loop()
    loop.run_in_executor(_executor, _warm_ocr)
    logger.info("IntelliGrade-H API v5.0 started. Upload dir: %s", UPLOAD_DIR)
    yield
    logger.info("IntelliGrade-H API shutting down.")
    _executor.shutdown(wait=False)


app = FastAPI(
    title="IntelliGrade-H API",
    description="AI-powered handwritten answer evaluation.",
    version="5.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────
# Pydantic request/response schemas
# ─────────────────────────────────────────────────────────

class RubricItem(BaseModel):
    criterion: str
    marks: float


class MCQOptions(BaseModel):
    A: Optional[str] = None
    B: Optional[str] = None
    C: Optional[str] = None
    D: Optional[str] = None
    E: Optional[str] = None


class EvaluateRequest(BaseModel):
    submission_id:       int
    question:            str
    question_type:       str = "auto"
    teacher_answer:      Optional[str] = None
    max_marks:           float = 10.0
    rubric_criteria:     Optional[List[RubricItem]] = None
    mcq_options:         Optional[MCQOptions] = None
    correct_option:      Optional[str] = None
    correct_answer:      Optional[str] = None
    numerical_tolerance: float = 0.01
    exam_question_id:    Optional[int] = None

    @field_validator("question_type")
    @classmethod
    def validate_question_type(cls, v):
        v = v.lower().strip()
        valid = {"auto", "mcq", "true_false", "fill_blank", "short_answer", "numerical", "open_ended", "diagram"}
        if v not in valid:
            raise ValueError(f"question_type must be one of: {', '.join(sorted(valid))}")
        return v

    @field_validator("correct_option")
    @classmethod
    def validate_correct_option(cls, v):
        if v is not None:
            v = v.strip().upper()
            if v not in ("A", "B", "C", "D", "E"):
                raise ValueError("correct_option must be A–E")
        return v


class EvaluateByPaperRequest(BaseModel):
    """
    Evaluate against a specific question from a known exam paper.
    Marks, question text, and type are loaded from the DB automatically.
    exam_paper_id accepts a slug string OR a numeric DB id string.
    """
    submission_id:       int
    exam_paper_id:       str
    question_number:     int
    correct_option:      Optional[str] = None
    correct_answer:      Optional[str] = None
    numerical_tolerance: float = 0.01


class EvaluationResponse(BaseModel):
    submission_id:       int
    question_type:       str
    final_score:         float
    max_marks:           float
    llm_score:           float
    similarity_score:    float
    mcq_correct:         Optional[bool]
    mcq_detected_answer: Optional[str]
    mcq_correct_answer:  Optional[str]
    ocr_text:            str
    ocr_confidence:      float
    strengths:           list
    missing_concepts:    list
    feedback:            str
    confidence:          float
    evaluation_time_sec: float
    rubric_details:      Optional[dict]
    exam_paper_id:       Optional[int] = None
    exam_question_id:    Optional[int] = None
    question_number:     Optional[int] = None


class PatchQuestionRequest(BaseModel):
    teacher_answer: Optional[str] = None
    correct_option: Optional[str] = None
    question_type:  Optional[str] = None


class RubricUploadRequest(BaseModel):
    question_id: int
    criteria:    List[RubricItem]


# ─────────────────────────────────────────────────────────
# Thread-pool helpers (one per parser module)
# ─────────────────────────────────────────────────────────

def _parse_paper_in_thread(pdf_path: str):
    """Run QuestionPaperParser in the thread pool."""
    from backend.question_paper_parser import QuestionPaperParser
    from backend.llm_provider import get_llm_client
    return QuestionPaperParser(llm_client=get_llm_client()).parse(pdf_path)


def _parse_answer_key_in_thread(pdf_path: str) -> dict:
    """Run answer_key_parser in the thread pool."""
    from backend.answer_key_parser import parse_answer_key
    return parse_answer_key(pdf_path)


def _parse_booklet_in_thread(pdf_path: str):
    """Run student_answer_parser in the thread pool."""
    from backend.student_answer_parser import parse_student_booklet
    from backend.llm_provider import get_llm_client
    from backend.ocr_module import OCRModule
    return parse_student_booklet(pdf_path, llm=get_llm_client(), ocr_module=OCRModule())


# ─────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────

def _run_ocr_on_path(image_path: str, engine: EvaluationEngine, ocr_engine: Optional[str] = None):
    from backend.ocr_module import OCRResult, OCRModule
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Uploaded file not found: {image_path}")
    ocr = OCRModule(trocr_model_path=os.getenv("TROCR_MODEL_PATH")) if ocr_engine else engine.ocr
    if path.suffix.lower() == ".pdf":
        try:
            pages = ocr.extract_from_pdf(str(path))
            if not pages:
                return OCRResult(text="", confidence=0.0, engine="pdf")
            text = "\n".join(r.text for r in pages if r.text)
            conf = sum(r.confidence for r in pages) / len(pages)
            return OCRResult(text=text, confidence=conf, engine="pdf+trocr")
        except Exception as e:
            logger.error("PDF OCR failed: %s", e)
            return OCRResult(text="", confidence=0.0, engine="failed")
    return ocr.extract_text(str(path))


def _mcq_options_to_dict(opts: Optional[MCQOptions]) -> Optional[dict]:
    return {k: v for k, v in opts.model_dump().items() if v is not None} if opts else None


def _make_paper_id(course_code: str, exam_name: str, set_name: str = "") -> str:
    base = f"{course_code}_{exam_name}_{set_name}".strip("_")
    base = re.sub(r"[^A-Za-z0-9_\-]", "_", base)
    base = re.sub(r"_+", "_", base).strip("_")
    return base[:180]


# ─────────────────────────────────────────────────────────
# Root / Health
# ─────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "message": "IntelliGrade-H API is running.",
        "version": "5.0.0",
        "llm_provider": "groq",
    }


@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": time.time()}


# ─────────────────────────────────────────────────────────
# Question Paper
# ─────────────────────────────────────────────────────────

@app.post("/paper/upload", summary="Upload a question paper PDF — auto-extracts structure")
async def upload_question_paper(
    file:     UploadFile = File(...),
    set_name: str        = Form(""),
    db:       Session    = Depends(get_db),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted.")

    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(413, f"File too large. Max {MAX_FILE_MB} MB.")

    file_path = UPLOAD_DIR / f"qpaper_{uuid.uuid4().hex}.pdf"
    async with aiofiles.open(file_path, "wb") as f:
        await f.write(file_bytes)

    loop = asyncio.get_running_loop()
    try:
        parsed = await asyncio.wait_for(
            loop.run_in_executor(_executor, _parse_paper_in_thread, str(file_path)),
            timeout=120
        )
    except Exception as e:
        raise HTTPException(500, f"Paper parsing failed: {e}")

    paper_id = _make_paper_id(parsed.course_code or "UNKNOWN", parsed.exam_name or "EXAM", set_name)

    existing = db.query(ExamPaper).filter_by(paper_id=paper_id).first()
    if existing:
        db.query(ExamPart).filter_by(paper_id_fk=existing.id).delete()
        db.query(ExamQuestion).filter_by(paper_id_fk=existing.id).delete()
        db.flush()
        paper_row = existing
    else:
        paper_row = ExamPaper()
        db.add(paper_row)

    # Compute correct total marks: OR pairs count only once
    def _correct_total(questions):
        total = 0.0
        i = 0
        while i < len(questions):
            q = questions[i]
            if not q.is_or_option:
                # If next question is its OR alternative, count only this one
                if i + 1 < len(questions) and questions[i + 1].is_or_option:
                    total += float(q.marks or 0)
                    i += 2
                else:
                    total += float(q.marks or 0)
                    i += 1
            else:
                i += 1  # skip OR alternatives — already counted their primary
        return total

    correct_total = _correct_total(parsed.questions) if parsed.questions else parsed.total_marks

    paper_row.paper_id       = paper_id
    paper_row.pdf_path       = str(file_path)
    paper_row.course_code    = parsed.course_code
    paper_row.course_name    = parsed.course_name
    paper_row.exam_name      = parsed.exam_name
    paper_row.total_marks    = correct_total or parsed.total_marks
    paper_row.duration_hours = parsed.duration_hours
    paper_row.exam_date      = parsed.date
    paper_row.batch          = parsed.batch
    paper_row.programme      = parsed.programme
    paper_row.semester       = parsed.semester
    paper_row.set_name       = set_name
    paper_row.raw_text       = parsed.raw_text[:10000]
    paper_row.parsed_at      = __import__("datetime").datetime.utcnow()
    db.flush()

    for part in parsed.parts:
        db.add(ExamPart(
            paper_id_fk        = paper_row.id,
            part_name          = part.part_name,
            marks_per_question = part.marks_per_question,
            num_questions      = part.num_questions,
            instructions       = part.instructions,
        ))

    for q in parsed.questions:
        db.add(ExamQuestion(
            paper_id_fk     = paper_row.id,
            question_number = q.question_number,
            question_text   = q.question_text,
            marks           = q.marks,
            part_name       = q.part_name,
            question_type   = q.question_type,
            is_or_option    = q.is_or_option,
        ))

    db.commit()
    db.refresh(paper_row)

    questions_out = (
        db.query(ExamQuestion)
        .filter_by(paper_id_fk=paper_row.id)
        .order_by(ExamQuestion.question_number)
        .all()
    )

    return {
        "paper_id":      paper_id,
        "db_id":         paper_row.id,
        "status":        "updated" if existing else "created",
        "course_code":   paper_row.course_code,
        "course_name":   paper_row.course_name,
        "exam_name":     paper_row.exam_name,
        "total_marks":   paper_row.total_marks,
        "num_questions": len(questions_out),
        "duration_hours": paper_row.duration_hours,
        "date":          paper_row.exam_date,
        "batch":         paper_row.batch,
        "programme":     paper_row.programme,
        "semester":      paper_row.semester,
        "set_name":      paper_row.set_name,
        "parts": [
            {
                "part_name":          p.part_name,
                "marks_per_question": p.marks_per_question,
                "num_questions":      p.num_questions,
                "instructions":       p.instructions,
            }
            for p in paper_row.parts
        ],
        "questions": [
            {
                "id":              q.id,
                "question_number": q.question_number,
                "question_text":   q.question_text,
                "marks":           q.marks,
                "part_name":       q.part_name,
                "question_type":   q.question_type,
                "is_or_option":    q.is_or_option,
                "has_answer_key":  bool(q.teacher_answer),
                "teacher_answer":  q.teacher_answer or "",
            }
            for q in questions_out
        ],
        "message": f"Paper parsed. {len(questions_out)} questions extracted.",
    }


@app.post("/answer-key/upload", summary="Upload an answer key PDF — links teacher answers to questions")
async def upload_answer_key(
    file:     UploadFile = File(...),
    paper_id: str        = Form(...),
    db:       Session    = Depends(get_db),
):
    paper_row = db.query(ExamPaper).filter_by(paper_id=paper_id).first()
    if not paper_row:
        raise HTTPException(404, f"ExamPaper '{paper_id}' not found.")

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted.")

    file_bytes = await file.read()
    file_path  = UPLOAD_DIR / f"anskey_{uuid.uuid4().hex}.pdf"
    async with aiofiles.open(file_path, "wb") as f:
        await f.write(file_bytes)

    loop = asyncio.get_running_loop()
    try:
        answers: dict = await asyncio.wait_for(
            loop.run_in_executor(
                _executor, _parse_answer_key_in_thread, str(file_path)
            ),
            timeout=120
        )
    except Exception as e:
        raise HTTPException(500, f"Answer key parsing failed: {e}")

    if not answers:
        return {
            "paper_id":                paper_row.paper_id,
            "answers_linked":          0,
            "total_answers_extracted": 0,
            "message": "No answers extracted. Ensure the PDF has selectable text.",
        }

    updated = 0
    for q_num, answer_text in answers.items():
        for row in db.query(ExamQuestion).filter_by(
            paper_id_fk=paper_row.id, question_number=q_num
        ).all():
            row.teacher_answer = answer_text
            updated += 1
    db.commit()

    all_questions = (
        db.query(ExamQuestion)
        .filter_by(paper_id_fk=paper_row.id)
        .order_by(ExamQuestion.question_number)
        .all()
    )
    parts = db.query(ExamPart).filter_by(paper_id_fk=paper_row.id).all()

    return {
        "paper_id":                paper_row.paper_id,
        "course_code":             paper_row.course_code,
        "course_name":             paper_row.course_name,
        "exam_name":               paper_row.exam_name,
        "total_marks":             paper_row.total_marks,
        "set_name":                paper_row.set_name,
        "exam_paper_id":           paper_row.id,
        "answers_linked":          updated,
        "total_answers_extracted": len(answers),
        "parts": [
            {
                "part_name":          p.part_name,
                "marks_per_question": p.marks_per_question,
                "num_questions":      p.num_questions,
            }
            for p in parts
        ],
        "questions": [
            {
                "id":              q.id,
                "question_number": q.question_number,
                "question_text":   q.question_text,
                "marks":           q.marks,
                "part_name":       q.part_name,
                "question_type":   q.question_type,
                "is_or_option":    q.is_or_option,
                "teacher_answer":  q.teacher_answer or "",
                "has_answer":      bool(q.teacher_answer),
            }
            for q in all_questions
        ],
        "message": f"Answer key processed. {updated} questions updated.",
    }


@app.get("/papers", summary="List all exam papers")
async def list_papers(db: Session = Depends(get_db)):
    papers = db.query(ExamPaper).order_by(ExamPaper.parsed_at.desc()).all()
    return [
        {
            "id":           p.id,
            "paper_id":     p.paper_id,
            "course_code":  p.course_code,
            "course_name":  p.course_name,
            "exam_name":    p.exam_name,
            "total_marks":  p.total_marks,
            "set_name":     p.set_name,
            "exam_date":    p.exam_date,
            "num_questions": len(p.questions),
            "parsed_at":    p.parsed_at.isoformat() if p.parsed_at else None,
        }
        for p in papers
    ]


@app.get("/paper/{paper_id}", summary="Get full structure of an exam paper")
async def get_paper(paper_id: str, db: Session = Depends(get_db)):
    paper_row = db.query(ExamPaper).filter_by(paper_id=paper_id).first()
    if not paper_row:
        raise HTTPException(404, f"Paper '{paper_id}' not found.")

    questions = (
        db.query(ExamQuestion)
        .filter_by(paper_id_fk=paper_row.id)
        .order_by(ExamQuestion.question_number)
        .all()
    )
    return {
        "id":            paper_row.id,
        "paper_id":      paper_row.paper_id,
        "course_code":   paper_row.course_code,
        "course_name":   paper_row.course_name,
        "exam_name":     paper_row.exam_name,
        "total_marks":   paper_row.total_marks,
        "duration_hours": paper_row.duration_hours,
        "exam_date":     paper_row.exam_date,
        "batch":         paper_row.batch,
        "programme":     paper_row.programme,
        "semester":      paper_row.semester,
        "set_name":      paper_row.set_name,
        "parts": [
            {
                "part_name":          p.part_name,
                "marks_per_question": p.marks_per_question,
                "num_questions":      p.num_questions,
                "instructions":       p.instructions,
            }
            for p in paper_row.parts
        ],
        "questions": [
            {
                "id":              q.id,
                "question_number": q.question_number,
                "question_text":   q.question_text,
                "marks":           q.marks,
                "part_name":       q.part_name,
                "question_type":   q.question_type,
                "is_or_option":    q.is_or_option,
                "has_answer_key":  bool(q.teacher_answer),
                "teacher_answer":  q.teacher_answer or "",
            }
            for q in questions
        ],
    }


@app.delete("/paper/{paper_id}", summary="Delete an exam paper and all its questions")
async def delete_paper(paper_id: str, db: Session = Depends(get_db)):
    paper_row = db.query(ExamPaper).filter_by(paper_id=paper_id).first()
    if not paper_row:
        raise HTTPException(404, f"Paper '{paper_id}' not found.")
    # Cascade deletes ExamPart and ExamQuestion via relationship cascade
    db.delete(paper_row)
    db.commit()
    return {"status": "deleted", "paper_id": paper_id}


@app.patch("/paper/{paper_id}/question/{question_number}", summary="Update a question's answer or metadata")
async def patch_question(
    paper_id:        str,
    question_number: int,
    payload:         PatchQuestionRequest,
    db:              Session = Depends(get_db),
):
    paper_row = db.query(ExamPaper).filter_by(paper_id=paper_id).first()
    if not paper_row:
        raise HTTPException(404, f"Paper '{paper_id}' not found.")

    row = db.query(ExamQuestion).filter_by(
        paper_id_fk=paper_row.id, question_number=question_number
    ).first()
    if not row:
        raise HTTPException(404, f"Question {question_number} not found in paper '{paper_id}'.")

    if payload.teacher_answer is not None:
        row.teacher_answer = payload.teacher_answer
    if payload.correct_option is not None:
        row.correct_option = payload.correct_option
    if payload.question_type is not None:
        row.question_type = payload.question_type
    db.commit()

    return {"status": "ok", "paper_id": paper_id, "question_number": question_number}


# ─────────────────────────────────────────────────────────
# Evaluate — paper-based
# ─────────────────────────────────────────────────────────

@app.post("/evaluate/paper", response_model=EvaluationResponse, summary="Evaluate against a paper question")
async def evaluate_by_paper(
    request:          EvaluateByPaperRequest,
    background_tasks: BackgroundTasks,
    db:               Session = Depends(get_db),
):
    submission = db.query(Submission).filter_by(id=request.submission_id).first()
    if not submission:
        raise HTTPException(404, "Submission not found.")
    if not Path(submission.image_path).exists():
        raise HTTPException(404, f"Uploaded file missing: {submission.image_path}")

    # Resolve paper — accept slug or numeric id
    paper_row = db.query(ExamPaper).filter_by(paper_id=request.exam_paper_id).first()
    if not paper_row and request.exam_paper_id.isdigit():
        paper_row = db.query(ExamPaper).filter_by(id=int(request.exam_paper_id)).first()
    if not paper_row:
        raise HTTPException(404, f"ExamPaper '{request.exam_paper_id}' not found.")

    exam_q = db.query(ExamQuestion).filter_by(
        paper_id_fk=paper_row.id,
        question_number=request.question_number,
        is_or_option=False,
    ).first()
    if not exam_q:
        raise HTTPException(404, f"Question {request.question_number} not found in '{paper_row.paper_id}'.")

    rubric = [{"criterion": r.criterion, "marks": r.marks} for r in exam_q.rubrics] if exam_q.rubrics else None

    engine = get_engine()
    loop   = asyncio.get_running_loop()

    def _do_evaluate():
        return engine.evaluate(
            student_image       = submission.image_path,
            question            = exam_q.question_text,
            teacher_answer      = exam_q.teacher_answer or "",
            max_marks           = exam_q.marks,
            rubric_criteria     = rubric,
            question_type       = exam_q.question_type,
            correct_option      = request.correct_option or exam_q.correct_option,
            correct_answer      = request.correct_answer,
            numerical_tolerance = request.numerical_tolerance,
        )

    try:
        result: EvaluationResult = await loop.run_in_executor(_executor, _do_evaluate)
    except Exception as e:
        raise HTTPException(500, f"Evaluation failed: {e}")

    submission.exam_paper_id    = paper_row.id
    submission.exam_question_id = exam_q.id

    existing = db.query(Result).filter_by(submission_id=submission.id).first()
    if existing:
        db.delete(existing)
        db.flush()

    db.add(Result(
        submission_id       = submission.id,
        final_score         = result.final_score,
        max_marks           = result.max_marks,
        llm_score           = result.llm_score,
        similarity_score    = result.similarity_score,
        strengths           = json.dumps(result.strengths),
        missing_concepts    = json.dumps(result.missing_concepts),
        feedback            = result.feedback,
        confidence          = result.confidence,
        evaluation_time_sec = result.evaluation_time_sec,
        question_type       = result.question_type,
        mcq_correct         = result.is_correct,
        mcq_detected_answer = result.detected_answer,
        mcq_correct_answer  = result.correct_answer,
    ))
    db.commit()
    background_tasks.add_task(_recompute_metrics_background, SessionLocal)

    return EvaluationResponse(
        submission_id       = submission.id,
        question_type       = result.question_type,
        final_score         = result.final_score,
        max_marks           = result.max_marks,
        llm_score           = result.llm_score,
        similarity_score    = result.similarity_score,
        mcq_correct         = result.is_correct,
        mcq_detected_answer = result.detected_answer or None,
        mcq_correct_answer  = result.correct_answer or None,
        ocr_text            = result.extracted_text,
        ocr_confidence      = result.ocr_confidence,
        strengths           = result.strengths,
        missing_concepts    = result.missing_concepts,
        feedback            = result.feedback,
        confidence          = result.confidence,
        evaluation_time_sec = result.evaluation_time_sec,
        rubric_details      = result.rubric_details,
        exam_paper_id       = paper_row.id,
        exam_question_id    = exam_q.id,
        question_number     = exam_q.question_number,
    )


# ─────────────────────────────────────────────────────────
# Legacy upload / OCR / evaluate
# ─────────────────────────────────────────────────────────

@app.post("/upload", summary="Upload a student answer sheet image or PDF")
async def upload_answer_sheet(
    file:         UploadFile = File(...),
    student_code: str        = Form(...),
    db:           Session    = Depends(get_db),
):
    content_type = file.content_type or ""
    allowed      = {"image/jpeg", "image/png", "application/pdf"}
    if content_type not in allowed:
        suffix = Path(file.filename or "").suffix.lower()
        ct_map = {".pdf": "application/pdf", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
        content_type = ct_map.get(suffix, "")
        if not content_type:
            raise HTTPException(415, f"Unsupported file type: {file.content_type}")

    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(413, f"File too large. Max {MAX_FILE_MB} MB.")

    ext       = Path(file.filename or "upload").suffix or ".jpg"
    file_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
    async with aiofiles.open(file_path, "wb") as f:
        await f.write(file_bytes)

    student = db.query(Student).filter_by(student_code=student_code).first()
    if not student:
        student = Student(student_code=student_code)
        db.add(student)
        db.commit()
        db.refresh(student)

    submission = Submission(student_id=student.id, image_path=str(file_path), question_id=None)
    db.add(submission)
    db.commit()
    db.refresh(submission)

    return {
        "submission_id": submission.id,
        "filename":      file_path.name,
        "student_code":  student_code,
        "message":       "File uploaded successfully.",
    }


@app.post("/ocr/{submission_id}", summary="Run OCR on an uploaded submission")
async def run_ocr(
    submission_id:   int,
    engine_override: Optional[str] = Query(None, alias="engine"),
    db:              Session = Depends(get_db),
):
    submission = db.query(Submission).filter_by(id=submission_id).first()
    if not submission:
        raise HTTPException(404, "Submission not found.")

    loop = asyncio.get_running_loop()
    try:
        ocr_result = await loop.run_in_executor(
            _executor, _run_ocr_on_path, submission.image_path, get_engine(), engine_override
        )
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))

    submission.extracted_text = ocr_result.text
    submission.ocr_confidence = ocr_result.confidence
    db.commit()

    return {
        "submission_id":  submission_id,
        "extracted_text": ocr_result.text,
        "confidence":     ocr_result.confidence,
        "engine":         ocr_result.engine,
    }


@app.post("/evaluate", response_model=EvaluationResponse, summary="Evaluate a submission (manual marks)")
async def evaluate_submission(
    request:          EvaluateRequest,
    background_tasks: BackgroundTasks,
    db:               Session = Depends(get_db),
):
    submission = db.query(Submission).filter_by(id=request.submission_id).first()
    if not submission:
        raise HTTPException(404, "Submission not found.")
    if not Path(submission.image_path).exists():
        raise HTTPException(404, f"Uploaded file missing: {submission.image_path}")

    rubric = [{"criterion": r.criterion, "marks": r.marks} for r in request.rubric_criteria] if request.rubric_criteria else None

    if request.exam_question_id:
        eq = db.query(ExamQuestion).filter_by(id=request.exam_question_id).first()
        if eq:
            request.max_marks      = eq.marks
            request.question       = request.question or eq.question_text
            request.teacher_answer = request.teacher_answer or eq.teacher_answer or ""
            if not rubric and eq.rubrics:
                rubric = [{"criterion": r.criterion, "marks": r.marks} for r in eq.rubrics]

    engine = get_engine()
    loop   = asyncio.get_running_loop()

    def _do_evaluate():
        return engine.evaluate(
            student_image       = submission.image_path,
            question            = request.question,
            teacher_answer      = request.teacher_answer or "",
            max_marks           = request.max_marks,
            rubric_criteria     = rubric,
            question_type       = request.question_type,
            correct_option      = request.correct_option,
            correct_answer      = request.correct_answer,
            mcq_options         = _mcq_options_to_dict(request.mcq_options),
            numerical_tolerance = request.numerical_tolerance,
        )

    try:
        result: EvaluationResult = await loop.run_in_executor(_executor, _do_evaluate)
    except Exception as e:
        raise HTTPException(500, f"Evaluation failed: {e}")

    existing = db.query(Result).filter_by(submission_id=submission.id).first()
    if existing:
        db.delete(existing)
        db.flush()

    db.add(Result(
        submission_id       = submission.id,
        final_score         = result.final_score,
        max_marks           = result.max_marks,
        llm_score           = result.llm_score,
        similarity_score    = result.similarity_score,
        strengths           = json.dumps(result.strengths),
        missing_concepts    = json.dumps(result.missing_concepts),
        feedback            = result.feedback,
        confidence          = result.confidence,
        evaluation_time_sec = result.evaluation_time_sec,
        question_type       = result.question_type,
        mcq_correct         = result.is_correct,
        mcq_detected_answer = result.detected_answer,
        mcq_correct_answer  = result.correct_answer,
    ))
    submission.extracted_text = result.extracted_text
    submission.ocr_confidence = result.ocr_confidence
    db.commit()
    background_tasks.add_task(_recompute_metrics_background, SessionLocal)

    return EvaluationResponse(
        submission_id       = submission.id,
        question_type       = result.question_type,
        final_score         = result.final_score,
        max_marks           = result.max_marks,
        llm_score           = result.llm_score,
        similarity_score    = result.similarity_score,
        mcq_correct         = result.is_correct,
        mcq_detected_answer = result.detected_answer or None,
        mcq_correct_answer  = result.correct_answer or None,
        ocr_text            = result.extracted_text,
        ocr_confidence      = result.ocr_confidence,
        strengths           = result.strengths,
        missing_concepts    = result.missing_concepts,
        feedback            = result.feedback,
        confidence          = result.confidence,
        evaluation_time_sec = result.evaluation_time_sec,
        rubric_details      = result.rubric_details,
    )


@app.get("/result/{submission_id}", summary="Get evaluation result")
async def get_result(submission_id: int, db: Session = Depends(get_db)):
    submission = db.query(Submission).filter_by(id=submission_id).first()
    if not submission:
        raise HTTPException(404, "Submission not found.")
    result = db.query(Result).filter_by(submission_id=submission_id).first()
    if not result:
        raise HTTPException(404, "Result not yet available. Run /evaluate first.")

    paper_info = question_info = None
    if submission.exam_paper_id:
        paper = db.query(ExamPaper).filter_by(id=submission.exam_paper_id).first()
        if paper:
            paper_info = {"paper_id": paper.paper_id, "course_name": paper.course_name}
    if submission.exam_question_id:
        eq = db.query(ExamQuestion).filter_by(id=submission.exam_question_id).first()
        if eq:
            question_info = {
                "question_number": eq.question_number,
                "question_text":   eq.question_text,
                "marks":           eq.marks,
                "part_name":       eq.part_name,
            }

    return {
        "submission_id":       submission_id,
        "question_type":       result.question_type,
        "final_score":         result.final_score,
        "max_marks":           result.max_marks or 10.0,
        "llm_score":           result.llm_score,
        "similarity_score":    result.similarity_score,
        "mcq_correct":         result.mcq_correct,
        "mcq_detected_answer": result.mcq_detected_answer,
        "mcq_correct_answer":  result.mcq_correct_answer,
        "strengths":           json.loads(result.strengths or "[]"),
        "missing_concepts":    json.loads(result.missing_concepts or "[]"),
        "feedback":            result.feedback,
        "confidence":          result.confidence,
        "evaluation_time_sec": result.evaluation_time_sec,
        "created_at":          result.created_at.isoformat(),
        "ocr_text":            submission.extracted_text or "",
        "ocr_confidence":      submission.ocr_confidence or 0.0,
        "exam_paper":          paper_info,
        "exam_question":       question_info,
    }


@app.post("/rubric", summary="Upload rubric criteria for an exam question")
async def upload_rubric(request: RubricUploadRequest, db: Session = Depends(get_db)):
    # Try ExamQuestion first (current model), fall back to legacy Question
    eq = db.query(ExamQuestion).filter_by(id=request.question_id).first()
    if eq:
        db.query(Rubric).filter_by(exam_question_id=eq.id).delete()
        for item in request.criteria:
            db.add(Rubric(exam_question_id=eq.id, criterion=item.criterion, marks=item.marks))
        db.commit()
    else:
        # Legacy path — old-style question_id
        if not db.query(Question).filter_by(id=request.question_id).first():
            raise HTTPException(404, "Question not found (checked both ExamQuestion and legacy Question tables).")
        db.query(Rubric).filter_by(question_id=request.question_id).delete()
        for item in request.criteria:
            db.add(Rubric(question_id=request.question_id, criterion=item.criterion, marks=item.marks))
        db.commit()
    return {"message": f"Rubric updated with {len(request.criteria)} criteria."}


# ─────────────────────────────────────────────────────────
# Student Booklet
# ─────────────────────────────────────────────────────────

@app.post("/booklet/upload", summary="Upload a student handwritten answer booklet PDF")
async def upload_student_booklet(
    file:         UploadFile = File(...),
    paper_id:     str        = Form(...),
    student_code: str        = Form(""),
    db:           Session    = Depends(get_db),
):
    paper_row = db.query(ExamPaper).filter_by(paper_id=paper_id).first()
    if not paper_row:
        raise HTTPException(404, f"ExamPaper '{paper_id}' not found. Upload the question paper first.")
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted.")

    student_row = None
    if student_code.strip():
        student_row = db.query(Student).filter_by(student_code=student_code.strip()).first()

    file_bytes = await file.read()
    file_path  = UPLOAD_DIR / f"booklet_{uuid.uuid4().hex}.pdf"
    async with aiofiles.open(file_path, "wb") as f:
        await f.write(file_bytes)

    loop = asyncio.get_running_loop()
    try:
        booklet_data = await asyncio.wait_for(
            loop.run_in_executor(
                _executor, _parse_booklet_in_thread, str(file_path)
            ),
            timeout=600  # 10 minutes — 16 pages × ~30s worst case
        )
    except asyncio.TimeoutError:
        raise HTTPException(504, "Booklet OCR timed out after 10 minutes. Try again or check OCR setup.")
    except asyncio.CancelledError:
        raise  # let uvicorn handle shutdown cleanly
    except Exception as e:
        raise HTTPException(500, f"Booklet parsing failed: {e}")

    # Upsert student from cover metadata
    if student_row is None and (booklet_data.roll_number or booklet_data.register_number):
        code = booklet_data.register_number or booklet_data.roll_number
        student_row = db.query(Student).filter_by(student_code=code).first()
        if not student_row:
            student_row = Student(student_code=code, name=f"Student {code}")
            db.add(student_row)
            db.flush()

    booklet_row = StudentBooklet(
        student_id      = student_row.id if student_row else None,
        exam_paper_id   = paper_row.id,
        pdf_path        = str(file_path),
        total_pages     = booklet_data.total_pages,
        roll_number     = booklet_data.roll_number,
        register_number = booklet_data.register_number,
        set_name        = booklet_data.set_name,
        course_code     = booklet_data.course_code,
        course_name     = booklet_data.course_name,
        exam_name       = booklet_data.exam_name,
        programme       = booklet_data.programme,
        semester        = booklet_data.semester,
        date_of_exam    = booklet_data.date_of_exam,
        raw_full_ocr    = booklet_data.raw_full_ocr[:50000],
    )
    db.add(booklet_row)
    db.flush()

    answer_rows = []
    for ans in booklet_data.answers:
        eq_row = db.query(ExamQuestion).filter_by(
            paper_id_fk=paper_row.id,
            question_number=ans.question_number,
            is_or_option=ans.is_or_option,
        ).first()
        row = StudentAnswerText(
            booklet_id       = booklet_row.id,
            exam_question_id = eq_row.id if eq_row else None,
            question_number  = ans.question_number,
            part_name        = ans.part_name,
            is_or_option     = ans.is_or_option,
            answer_text      = ans.answer_text,
            raw_ocr_text     = ans.raw_ocr_text,
            evaluated        = False,
        )
        db.add(row)
        answer_rows.append(row)

    db.commit()

    return {
        "booklet_id":      booklet_row.id,
        "paper_id":        paper_id,
        "roll_number":     booklet_row.roll_number,
        "register_number": booklet_row.register_number,
        "set_name":        booklet_row.set_name,
        "course_code":     booklet_row.course_code,
        "total_pages":     booklet_row.total_pages,
        "answers_found":   len(answer_rows),
        "questions": [
            {
                "question_number": a.question_number,
                "part_name":       a.part_name,
                "is_or_option":    a.is_or_option,
                "answer_text":     a.answer_text or "",
                "preview":         (a.answer_text or "")[:120] + ("…" if len(a.answer_text or "") > 120 else ""),
                "linked_to_eq":    a.exam_question_id is not None,
            }
            for a in answer_rows
        ],
        "raw_ocr_preview": booklet_row.raw_full_ocr[:500] if booklet_row.raw_full_ocr else "",
        "message": (
            f"Booklet parsed. {len(answer_rows)} answers found. "
            f"Call POST /booklet/{booklet_row.id}/evaluate to score."
        ),
    }


@app.get("/booklets", summary="List all student booklets")
async def list_booklets(paper_id: Optional[str] = None, db: Session = Depends(get_db)):
    q = db.query(StudentBooklet)
    if paper_id:
        paper_row = db.query(ExamPaper).filter_by(paper_id=paper_id).first()
        if paper_row:
            q = q.filter_by(exam_paper_id=paper_row.id)
    return [
        {
            "booklet_id":      r.id,
            "id":              r.id,
            "roll_number":     r.roll_number,
            "register_number": r.register_number,
            "set_name":        r.set_name,
            "course_code":     r.course_code,
            "total_pages":     r.total_pages,
            "num_answers":     len(r.answers),
            "answers_found":   len(r.answers),
            "paper_id":        r.exam_paper.paper_id if r.exam_paper else None,
            "parsed_at":       r.parsed_at.isoformat() if r.parsed_at else None,
        }
        for r in q.order_by(StudentBooklet.parsed_at.desc()).all()
    ]


@app.get("/booklet/{booklet_id}", summary="Get a booklet's parsed answers")
async def get_booklet(booklet_id: int, db: Session = Depends(get_db)):
    row = db.query(StudentBooklet).filter_by(id=booklet_id).first()
    if not row:
        raise HTTPException(404, f"Booklet id={booklet_id} not found.")
    return {
        "booklet_id":      row.id,
        "roll_number":     row.roll_number,
        "register_number": row.register_number,
        "set_name":        row.set_name,
        "course_code":     row.course_code,
        "course_name":     row.course_name,
        "exam_name":       row.exam_name,
        "total_pages":     row.total_pages,
        "parsed_at":       row.parsed_at.isoformat() if row.parsed_at else None,
        "answers": [
            {
                "id":              a.id,
                "question_number": a.question_number,
                "part_name":       a.part_name,
                "is_or_option":    a.is_or_option,
                "answer_text":     a.answer_text,
                "evaluated":       a.evaluated,
                "submission_id":   a.submission_id,
            }
            for a in row.answers
        ],
    }


class _BookletEvalReq(BaseModel):
    paper_id: Optional[str] = None  # optional paper_id override from bulk-upload flow


@app.post("/booklet/{booklet_id}/evaluate", summary="Evaluate all answers in a student booklet")
async def evaluate_booklet(
    booklet_id:       int,
    background_tasks: BackgroundTasks,
    request:          _BookletEvalReq = _BookletEvalReq(),
    db:               Session = Depends(get_db),
):
    booklet_row = db.query(StudentBooklet).filter_by(id=booklet_id).first()
    if not booklet_row:
        raise HTTPException(404, f"Booklet id={booklet_id} not found.")

    # Allow caller to pass a paper_id override (used by bulk-upload)
    paper_row = booklet_row.exam_paper
    if request.paper_id:
        override = db.query(ExamPaper).filter_by(paper_id=request.paper_id).first()
        if override:
            paper_row = override
    if not paper_row:
        raise HTTPException(400, "Booklet is not linked to an exam paper.")

    unevaluated = [a for a in booklet_row.answers if not a.evaluated]
    if not unevaluated:
        return {"message": "All answers already evaluated.", "booklet_id": booklet_id}

    engine          = get_engine()
    results_summary = []
    total_obtained  = total_possible = 0.0

    for ans_row in unevaluated:
        # Find the matching ExamQuestion
        eq_row = None
        if ans_row.exam_question_id:
            eq_row = db.query(ExamQuestion).filter_by(id=ans_row.exam_question_id).first()
        if eq_row is None:
            # Try exact match first (same is_or_option flag)
            eq_row = db.query(ExamQuestion).filter_by(
                paper_id_fk=paper_row.id,
                question_number=ans_row.question_number,
                is_or_option=ans_row.is_or_option,
            ).first()
        if eq_row is None:
            # Fallback: try the other OR flag (student may not mark OR correctly)
            eq_row = db.query(ExamQuestion).filter_by(
                paper_id_fk=paper_row.id,
                question_number=ans_row.question_number,
            ).first()
        if eq_row is None:
            results_summary.append({
                "question_number": ans_row.question_number,
                "status": "skipped",
                "reason": "No matching question found in exam paper. Check that the question paper was parsed correctly.",
            })
            continue

        teacher_answer = eq_row.teacher_answer or ""
        max_marks      = float(eq_row.marks or 2.0)
        question_type  = eq_row.question_type or "open_ended"
        student_answer = ans_row.answer_text or ""

        if not student_answer.strip():
            results_summary.append({"question_number": ans_row.question_number, "status": "skipped", "reason": "Empty student answer"})
            continue
        if not teacher_answer.strip():
            results_summary.append({"question_number": ans_row.question_number, "status": "skipped", "reason": "No teacher answer — upload answer key first"})
            continue

        # ── Evaluate BEFORE writing to DB so a failure never leaves
        #    orphaned Submission rows or corrupts the session. ──────────────
        try:
            rubric_list = (
                [{"criterion": r.criterion, "marks": r.marks} for r in eq_row.rubrics]
                if eq_row.rubrics else None
            )
            eval_result = engine.evaluate(
                student_answer  = student_answer,
                question        = eq_row.question_text or "",
                teacher_answer  = teacher_answer,
                max_marks       = max_marks,
                rubric_criteria = rubric_list,
                question_type   = question_type,
            )
            final_score        = round(max(0.0, min(max_marks, eval_result.final_score)), 2)
            sim_score          = eval_result.similarity_score
            llm_eval_feedback  = eval_result.feedback
            llm_eval_strengths = eval_result.strengths
            llm_eval_missing   = eval_result.missing_concepts
            llm_eval_conf      = eval_result.confidence
            llm_eval_score_raw = eval_result.llm_score
        except Exception as e:
            logger.error("Evaluation error Q%d: %s", ans_row.question_number, e)
            results_summary.append({"question_number": ans_row.question_number, "status": "error", "reason": str(e)})
            continue  # session is still clean — no DB writes attempted for this question

        # Evaluation succeeded — now write to DB
        sub = Submission(
            student_id       = booklet_row.student_id,
            exam_paper_id    = paper_row.id,
            exam_question_id = eq_row.id,
            image_path       = f"booklet_{booklet_id}_q{ans_row.question_number}",
            extracted_text   = student_answer,
            ocr_confidence   = 1.0,
        )
        db.add(sub)
        db.flush()

        db.add(Result(
            submission_id       = sub.id,
            question_type       = question_type,
            final_score         = final_score,
            max_marks           = max_marks,
            confidence          = llm_eval_conf,
            evaluation_time_sec = 0.0,
            feedback            = llm_eval_feedback,
            llm_score           = llm_eval_score_raw,
            similarity_score    = sim_score,
            strengths           = json.dumps(llm_eval_strengths),
            missing_concepts    = json.dumps(llm_eval_missing),
        ))
        db.flush()

        ans_row.submission_id = sub.id
        ans_row.evaluated     = True
        db.commit()

        total_obtained += final_score
        total_possible += max_marks
        results_summary.append({
            "question_number":      ans_row.question_number,
            "part_name":            ans_row.part_name,
            "question_type":        question_type,
            "max_marks":            max_marks,
            "score":                final_score,
            "percentage":           round(100 * final_score / max_marks, 1) if max_marks else 0,
            "feedback":             llm_eval_feedback,
            "strengths":            llm_eval_strengths,
            "missing_concepts":     llm_eval_missing,
            "component_scores": {
                "llm":        llm_eval_score_raw,
                "similarity": sim_score,
                "rubric":     getattr(eval_result, "rubric_score", 0.0),
                "keyword":    getattr(eval_result, "keyword_score", 0.0),
                "length":     getattr(eval_result, "length_score", 0.0),
            },
            # Diagram detection fields (populated for diagram questions only)
            "diagram_detected":       getattr(eval_result, "diagram_detected", False),
            "n_diagrams":             getattr(eval_result, "n_diagrams", 0),
            "diagram_detector_used":  getattr(eval_result, "diagram_detector_used", ""),
            "diagram_bboxes":         getattr(eval_result, "diagram_bboxes", []),
            "status":                 "evaluated",
            "submission_id":          sub.id,
        })

    background_tasks.add_task(_recompute_metrics_background, SessionLocal)

    return {
        "booklet_id":     booklet_id,
        "roll_number":    booklet_row.roll_number,
        "register_number": booklet_row.register_number,
        "paper_id":       paper_row.paper_id,
        "total_obtained": round(total_obtained, 2),
        "total_possible": round(total_possible, 2),
        "percentage":     round(100 * total_obtained / total_possible, 1) if total_possible else 0,
        "questions":      results_summary,
        "message": (
            f"Evaluated {sum(1 for r in results_summary if r['status'] == 'evaluated')}"
            f" / {len(unevaluated)} answers."
        ),
    }


@app.delete("/booklet/{booklet_id}", summary="Delete a student booklet and all its answers")
async def delete_booklet(booklet_id: int, db: Session = Depends(get_db)):
    row = db.query(StudentBooklet).filter_by(id=booklet_id).first()
    if not row:
        raise HTTPException(404, f"Booklet id={booklet_id} not found.")
    db.delete(row)
    db.commit()
    return {"status": "deleted", "booklet_id": booklet_id}


@app.patch("/booklet/{booklet_id}/answer/{question_number}", summary="Manually correct a student answer")
async def patch_student_answer(
    booklet_id:      int,
    question_number: int,
    payload:         dict,
    db:              Session = Depends(get_db),
):
    rows = db.query(StudentAnswerText).filter_by(
        booklet_id=booklet_id, question_number=question_number
    ).all()
    if not rows:
        raise HTTPException(404, f"No answer for booklet={booklet_id} Q{question_number}.")
    for row in rows:
        if "answer_text" in payload:
            row.answer_text = payload["answer_text"]
            row.evaluated   = False
    db.commit()
    return {"message": f"Answer for Q{question_number} updated. Re-evaluate to score."}


# ─────────────────────────────────────────────────────────
# Stats & Metrics
# ─────────────────────────────────────────────────────────

@app.get("/stats", summary="System statistics")
async def get_stats(db: Session = Depends(get_db)):
    results = db.query(Result).all()
    avg_score = avg_time = 0.0
    if results:
        avg_score = sum(r.final_score for r in results if r.final_score) / len(results)
        timed     = [r.evaluation_time_sec for r in results if r.evaluation_time_sec]
        avg_time  = sum(timed) / len(timed) if timed else 0.0

    return {
        "total_submissions":           db.query(Submission).count(),
        "evaluated":                   db.query(Result).count(),
        "mcq_evaluated":               sum(1 for r in results if r.question_type == "mcq"),
        "open_ended_evaluated":        sum(1 for r in results if r.question_type == "open_ended"),
        "average_score":               round(avg_score, 2),
        "average_evaluation_time_sec": round(avg_time, 2),
        "total_exam_papers":           db.query(ExamPaper).count(),
    }


@app.get("/metrics", summary="AI scoring accuracy metrics")
async def get_metrics(refresh: bool = False):
    if refresh or _metrics_cache["last_updated"] is None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(_executor, _recompute_metrics_background, SessionLocal)
    return {
        "last_updated":    _metrics_cache.get("last_updated"),
        "total_evaluated": _metrics_cache.get("total_evaluated", 0),
        "open_ended":      _metrics_cache.get("open_ended"),
        "mcq":             _metrics_cache.get("mcq"),
        "open_ended_note": (
            "Teacher scores are estimated via similarity_score × max_marks (no real ground-truth stored). "
            "Pearson r / MAE / Kappa reflect AI self-consistency, not accuracy against a human rater."
        ),
    }


@app.get("/metrics/print", summary="Print metrics report to server log (debug)")
async def print_metrics_to_log():
    """Calls print_metrics_report() — useful for server-side debugging."""
    from backend.metrics import print_metrics_report, compute_metrics, MetricsReport
    oe = _metrics_cache.get("open_ended")
    mcq = _metrics_cache.get("mcq")
    if oe:
        r = MetricsReport(question_type="open_ended", n_samples=oe.get("n_samples", 0),
                          mae=oe.get("mae", 0), pearson_r=oe.get("pearson_r", 0),
                          cohen_kappa=oe.get("cohen_kappa", 0),
                          accuracy_within_1=oe.get("accuracy_within_1_mark", 0),
                          accuracy_within_0_5=oe.get("accuracy_within_0_5_mark", 0),
                          mean_ai_score=oe.get("mean_ai_score", 0),
                          mean_teacher_score=oe.get("mean_teacher_score", 0))
        print_metrics_report(r)
    if mcq:
        r = MetricsReport(question_type="mcq", n_samples=mcq.get("n_samples", 0),
                          mcq_accuracy=mcq.get("accuracy", 0),
                          mcq_n_correct=mcq.get("n_correct", 0),
                          mcq_n_wrong=mcq.get("n_wrong", 0))
        print_metrics_report(r)
    return {"status": "printed to server log"}


@app.get("/metrics/compute", summary="Ad-hoc metric computation from score lists")
async def compute_metrics_adhoc(
    ai_scores:      str,
    teacher_scores: str,
    max_marks:      float = 10.0,
    question_type:  str   = "open_ended",
):
    try:
        ai = [float(x.strip()) for x in ai_scores.split(",") if x.strip()]
        gt = [float(x.strip()) for x in teacher_scores.split(",") if x.strip()]
    except ValueError:
        raise HTTPException(400, "Scores must be comma-separated floats.")
    if len(ai) != len(gt):
        raise HTTPException(400, f"Length mismatch: {len(ai)} vs {len(gt)}")
    if len(ai) < 2:
        raise HTTPException(400, "At least 2 score pairs required.")

    report   = compute_metrics(ai, gt, max_marks=max_marks, question_type=question_type)
    response = {"question_type": report.question_type, "n_samples": report.n_samples}

    if question_type == "mcq":
        response["mcq"] = {
            "accuracy":     report.mcq_accuracy,
            "accuracy_pct": f"{report.mcq_accuracy * 100:.1f}%",
            "n_correct":    report.mcq_n_correct,
            "n_wrong":      report.mcq_n_wrong,
        }
    else:
        response["open_ended"] = {
            "mae":                      report.mae,
            "pearson_r":                report.pearson_r,
            "cohen_kappa":              report.cohen_kappa,
            "accuracy_within_1_mark":   report.accuracy_within_1,
            "accuracy_within_0_5_mark": report.accuracy_within_0_5,
            "mean_ai_score":            report.mean_ai_score,
            "mean_teacher_score":       report.mean_teacher_score,
        }
    return response


@app.get("/submissions", summary="List all submissions with results")
async def list_submissions(db: Session = Depends(get_db)):
    return [
        {
            "submission_id":    sub.id,
            "student_code":     sub.student.student_code if sub.student else None,
            "image_path":       sub.image_path,
            "submitted_at":     sub.submitted_at.isoformat() if sub.submitted_at else None,
            "ocr_confidence":   sub.ocr_confidence,
            "exam_paper_id":    sub.exam_paper_id,
            "exam_question_id": sub.exam_question_id,
            "result": {
                "question_type": sub.result.question_type,
                "final_score":   sub.result.final_score,
                "max_marks":     sub.result.max_marks,
                "mcq_correct":   sub.result.mcq_correct,
                "feedback":      sub.result.feedback,
            } if sub.result else None,
        }
        for sub in db.query(Submission).all()
    ]


# ─────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.api:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=True,
        reload_excludes=["tests/*", "*.pyc"],
    )