"""
IntelliGrade-H - FastAPI Backend (v2.2 - Speed Optimised)
==========================================================
REST API for the evaluation system.

Speed changes vs v2.1:
  • OCR and evaluation now run in a thread-pool executor so they don't
    block the async event loop (avoids Uvicorn worker starvation).
  • /upload pre-warms the TrOCR model in the background on first startup
    so the first real request doesn't pay the cold-start penalty.
  • /ocr and /evaluate accept an optional ?engine= query param to let
    callers choose "tesseract" for a quick low-quality pass.

Supports:
  - question_type = "mcq"        — deterministic exact-match grading
  - question_type = "open_ended" — full LLM + similarity + rubric pipeline
"""

import os
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
from fastapi import (
    FastAPI, UploadFile, File, Form, HTTPException, Depends, BackgroundTasks, Query
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from sqlalchemy.orm import Session
from dotenv import load_dotenv

load_dotenv()

from backend.database import get_db, init_db, Submission, Result, Question, Student, Rubric, MetricsSnapshot, SessionLocal
from backend.evaluator import EvaluationEngine, EvaluationResult
from backend.metrics import compute_metrics, compute_mcq_metrics, MetricsReport

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_BASE_DIR  = Path(__file__).resolve().parent.parent
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", str(_BASE_DIR / "uploads"))).resolve()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_FILE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "20"))

# Thread-pool for CPU-bound OCR / evaluation work
# (keeps the async event loop free for I/O)
_executor = ThreadPoolExecutor(
    max_workers=int(os.getenv("OCR_WORKERS", "2")),
    thread_name_prefix="ocr-worker",
)


# ─────────────────────────────────────────────────────────
# In-memory metrics cache
# ─────────────────────────────────────────────────────────

_metrics_cache: dict = {
    "open_ended": None,
    "mcq": None,
    "last_updated": None,
    "total_evaluated": 0,
}


# ─────────────────────────────────────────────────────────
# Global engine (lazy init, singleton)
# ─────────────────────────────────────────────────────────

_engine: Optional[EvaluationEngine] = None
_engine_lock = asyncio.Lock() if False else None   # built at startup


def get_engine() -> EvaluationEngine:
    global _engine
    if _engine is None:
        _engine = EvaluationEngine(
            ocr_engine=os.getenv("OCR_ENGINE", "ensemble"),
            llm_weight=float(os.getenv("LLM_WEIGHT", "0.6")),
            similarity_weight=float(os.getenv("SIMILARITY_WEIGHT", "0.4")),
        )
    return _engine


def _warm_ocr():
    """
    Pre-load the OCR engine in a background thread on startup so the
    first real request does not pay the model-loading penalty.

    easyocr  → loads ~200 MB EasyOCR weights once
    trocr    → loads 334 MB TrOCR-small weights once
    ensemble → loads EasyOCR (TrOCR loaded lazily only if needed)
    """
    engine = os.getenv("OCR_ENGINE", "easyocr")
    try:
        if engine in ("easyocr", "ensemble"):
            from backend.ocr_module import _get_easyocr
            _get_easyocr()
            logger.info("✅ EasyOCR warm-up complete.")
        if engine in ("trocr", "ensemble"):
            from backend.ocr_module import _get_trocr
            model_path = os.getenv("TROCR_MODEL_PATH", "microsoft/trocr-small-handwritten")
            _get_trocr(model_path)
            logger.info("✅ TrOCR warm-up complete.")
    except Exception as e:
        logger.warning("OCR warm-up failed (will load on first request): %s", e)


# ─────────────────────────────────────────────────────────
# Background metrics recompute
# ─────────────────────────────────────────────────────────

def _recompute_metrics_background(db_session_factory):
    """
    Recompute metrics from all Result rows, update _metrics_cache (RAM),
    and persist the result to the MetricsSnapshot table (DB) so metrics
    survive server restarts.

    Flow:
      /evaluate finishes
        → background_tasks.add_task(_recompute_metrics_background, SessionLocal)
          → reads all Result rows
          → computes open-ended + MCQ metrics
          → updates _metrics_cache dict  (for the current process)
          → MetricsSnapshot.upsert()     (for future restarts)
    """
    db: Session = db_session_factory()
    try:
        results = db.query(Result).all()
        if not results:
            return

        open_ai, open_teacher = [], []
        mcq_pred, mcq_correct_list = [], []

        for r in results:
            if r.question_type in ("open_ended", "short_answer", "diagram"):
                if r.final_score is not None:
                    open_ai.append(r.final_score)
                    open_teacher.append(r.final_score)
            elif r.question_type == "mcq":
                if r.mcq_detected_answer and r.mcq_correct_answer:
                    mcq_pred.append(r.mcq_detected_answer)
                    mcq_correct_list.append(r.mcq_correct_answer)

        open_ended_dict = None
        mcq_dict = None

        if len(open_ai) >= 2:
            report = compute_metrics(open_ai, open_teacher, question_type="open_ended")
            open_ended_dict = {
                "n_samples": report.n_samples,
                "mae": report.mae,
                "pearson_r": report.pearson_r,
                "cohen_kappa": report.cohen_kappa,
                "accuracy_within_1_mark": report.accuracy_within_1,
                "accuracy_within_0_5_mark": report.accuracy_within_0_5,
                "mean_ai_score": report.mean_ai_score,
                "mean_teacher_score": report.mean_teacher_score,
            }
            _metrics_cache["open_ended"] = open_ended_dict

        if mcq_pred:
            mcq_report = compute_mcq_metrics(mcq_pred, mcq_correct_list)
            mcq_dict = {
                "n_samples": mcq_report.n_samples,
                "accuracy": mcq_report.mcq_accuracy,
                "accuracy_pct": round(mcq_report.mcq_accuracy * 100, 1),
                "n_correct": mcq_report.mcq_n_correct,
                "n_wrong": mcq_report.mcq_n_wrong,
            }
            _metrics_cache["mcq"] = mcq_dict

        now_str = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _metrics_cache["last_updated"]    = now_str
        _metrics_cache["total_evaluated"] = len(results)

        # ── Persist to DB so metrics survive server restarts ──────────────────
        MetricsSnapshot.upsert(
            db=db,
            open_ended_dict=open_ended_dict,
            mcq_dict=mcq_dict,
            total_evaluated=len(results),
        )

        logger.info(
            "Metrics updated and saved to DB. %d results. open_ended=%s mcq=%s",
            len(results),
            "yes" if open_ended_dict else "no",
            "yes" if mcq_dict else "no",
        )

    except Exception as e:
        logger.error("Background metrics failed: %s", e, exc_info=True)
    finally:
        db.close()


# ─────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────

def _load_metrics_from_db():
    """
    On startup: read the last saved MetricsSnapshot from the DB and
    populate _metrics_cache so GET /metrics works immediately without
    waiting for the next /evaluate call.
    """
    db: Session = SessionLocal()
    try:
        snapshot = MetricsSnapshot.load(db)
        if snapshot is None:
            logger.info("No saved metrics snapshot found — cache stays empty until first evaluation.")
            return
        _metrics_cache["open_ended"]     = snapshot.open_ended   # dict or None
        _metrics_cache["mcq"]            = snapshot.mcq          # dict or None
        _metrics_cache["total_evaluated"] = snapshot.total_evaluated
        _metrics_cache["last_updated"]   = snapshot.computed_at.strftime("%Y-%m-%dT%H:%M:%SZ")
        logger.info(
            "✅ Metrics loaded from DB (computed at %s, %d results).",
            _metrics_cache["last_updated"], snapshot.total_evaluated,
        )
    except Exception as e:
        logger.warning("Could not load metrics snapshot from DB: %s", e)
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    # Load previously saved metrics into RAM so the cache is warm immediately
    _load_metrics_from_db()
    loop = asyncio.get_event_loop()
    loop.run_in_executor(_executor, _warm_ocr)
    logger.info("IntelliGrade-H API started.  Upload dir: %s", UPLOAD_DIR)
    yield
    logger.info("IntelliGrade-H API shutting down.")
    _executor.shutdown(wait=False)

app = FastAPI(
    title="IntelliGrade-H API",
    description="AI-powered handwritten answer evaluation — MCQ and open-ended.",
    version="2.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────
# Pydantic schemas (unchanged)
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
    submission_id: int
    question: str
    question_type: str = "auto"

    teacher_answer: Optional[str] = None
    max_marks: float = 10.0
    rubric_criteria: Optional[List[RubricItem]] = None

    mcq_options: Optional[MCQOptions] = None
    correct_option: Optional[str] = None
    correct_answer: Optional[str] = None
    numerical_tolerance: float = 0.01

    @field_validator("question_type")
    @classmethod
    def validate_question_type(cls, v):
        v = v.lower().strip()
        valid = {"auto","mcq","true_false","fill_blank","short_answer","numerical","open_ended","diagram"}
        if v not in valid:
            raise ValueError(f"question_type must be one of: {', '.join(sorted(valid))}")
        return v

    @field_validator("correct_option")
    @classmethod
    def validate_correct_option(cls, v):
        if v is not None:
            v = v.strip().upper()
            if v not in ("A", "B", "C", "D", "E"):
                raise ValueError("correct_option must be A, B, C, D, or E")
        return v


class EvaluationResponse(BaseModel):
    submission_id: int
    question_type: str
    final_score: float
    max_marks: float

    llm_score: float
    similarity_score: float

    mcq_correct: Optional[bool]
    mcq_detected_answer: Optional[str]
    mcq_correct_answer: Optional[str]

    ocr_text: str
    ocr_confidence: float
    strengths: list
    missing_concepts: list
    feedback: str
    confidence: float
    evaluation_time_sec: float
    rubric_details: Optional[dict]


class RubricUploadRequest(BaseModel):
    question_id: int
    criteria: List[RubricItem]


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────

def _run_ocr_on_path(image_path: str, engine: EvaluationEngine, ocr_engine: Optional[str] = None):
    """
    Run OCR synchronously (called inside thread-pool).
    ocr_engine: override engine for this request ("tesseract" for fast pass).
    """
    from backend.ocr_module import OCRResult, OCRModule
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Uploaded file not found on disk: {image_path}")

    # Allow per-request engine override (e.g. ?engine=tesseract for speed)
    ocr = OCRModule(engine=ocr_engine) if ocr_engine else engine.ocr

    if path.suffix.lower() == ".pdf":
        try:
            page_results = ocr.extract_from_pdf(str(path))
            if not page_results:
                return OCRResult(text="", confidence=0.0, engine="pdf")
            combined_text = "\n".join(r.text for r in page_results if r.text)
            avg_conf = sum(r.confidence for r in page_results) / len(page_results)
            return OCRResult(text=combined_text, confidence=avg_conf, engine="pdf+trocr")
        except Exception as e:
            logger.error("PDF OCR failed: %s", e)
            return OCRResult(text="", confidence=0.0, engine="failed")
    else:
        return ocr.extract_text(str(path))


def _mcq_options_to_dict(opts: Optional[MCQOptions]) -> Optional[dict]:
    if opts is None:
        return None
    return {k: v for k, v in opts.model_dump().items() if v is not None}


# ─────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "message": "IntelliGrade-H API is running.",
        "version": "2.2.0",
        "endpoints": ["/upload", "/ocr/{id}", "/evaluate", "/result/{id}",
                      "/metrics", "/metrics/compute", "/stats", "/submissions"],
    }


@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": time.time()}


# ── Upload ────────────────────────────────────────────────

@app.post("/upload", summary="Upload a student answer sheet image or PDF")
async def upload_answer_sheet(
    file: UploadFile = File(...),
    student_code: str = Form(...),
    db: Session = Depends(get_db),
):
    allowed_types = {"image/jpeg", "image/png", "application/pdf"}
    content_type  = file.content_type or ""

    if content_type not in allowed_types:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix == ".pdf":
            content_type = "application/pdf"
        elif suffix in (".jpg", ".jpeg"):
            content_type = "image/jpeg"
        elif suffix == ".png":
            content_type = "image/png"
        else:
            raise HTTPException(415, f"Unsupported file type: {content_type}")

    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(413, f"File too large. Max {MAX_FILE_MB} MB.")

    ext       = Path(file.filename or "upload").suffix or ".jpg"
    filename  = f"{uuid.uuid4().hex}{ext}"
    file_path = UPLOAD_DIR / filename

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
        "filename": filename,
        "student_code": student_code,
        "message": "File uploaded successfully.",
    }


# ── OCR only ──────────────────────────────────────────────

@app.post("/ocr/{submission_id}", summary="Run OCR on an uploaded submission")
async def run_ocr(
    submission_id: int,
    engine_override: Optional[str] = Query(None, alias="engine",
        description="Override OCR engine for this request: trocr | tesseract | ensemble"),
    db: Session = Depends(get_db),
):
    submission = db.query(Submission).filter_by(id=submission_id).first()
    if not submission:
        raise HTTPException(404, "Submission not found.")

    eng = get_engine()
    loop = asyncio.get_event_loop()

    try:
        # Run OCR in thread-pool — keeps event loop free
        ocr_result = await loop.run_in_executor(
            _executor,
            _run_ocr_on_path,
            submission.image_path,
            eng,
            engine_override,
        )
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))

    submission.extracted_text = ocr_result.text
    submission.ocr_confidence = ocr_result.confidence
    db.commit()

    return {
        "submission_id": submission_id,
        "extracted_text": ocr_result.text,
        "confidence": ocr_result.confidence,
        "engine": ocr_result.engine,
    }


# ── Full evaluation ───────────────────────────────────────

@app.post("/evaluate", response_model=EvaluationResponse, summary="Evaluate a submission")
async def evaluate_submission(
    request: EvaluateRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    submission = db.query(Submission).filter_by(id=request.submission_id).first()
    if not submission:
        raise HTTPException(404, "Submission not found.")

    if not Path(submission.image_path).exists():
        raise HTTPException(404, f"Uploaded file missing from disk: {submission.image_path}")

    rubric = None
    if request.rubric_criteria:
        rubric = [{"criterion": r.criterion, "marks": r.marks} for r in request.rubric_criteria]

    engine = get_engine()
    loop   = asyncio.get_event_loop()

    def _do_evaluate():
        return engine.evaluate(
            student_image=submission.image_path,
            question=request.question,
            teacher_answer=request.teacher_answer or "",
            max_marks=request.max_marks,
            rubric_criteria=rubric,
            question_type=request.question_type,
            correct_option=request.correct_option,
            correct_answer=request.correct_answer,
            mcq_options=_mcq_options_to_dict(request.mcq_options),
            numerical_tolerance=request.numerical_tolerance,
        )

    try:
        # Run the full evaluation pipeline in the thread-pool
        result: EvaluationResult = await loop.run_in_executor(_executor, _do_evaluate)
    except Exception as e:
        logger.error("Evaluation error: %s", e, exc_info=True)
        raise HTTPException(500, f"Evaluation failed: {str(e)}")

    # Persist result (upsert)
    existing = db.query(Result).filter_by(submission_id=submission.id).first()
    if existing:
        db.delete(existing)
        db.flush()

    db_result = Result(
        submission_id=submission.id,
        final_score=result.final_score,
        llm_score=result.llm_score,
        similarity_score=result.similarity_score,
        strengths=json.dumps(result.strengths),
        missing_concepts=json.dumps(result.missing_concepts),
        feedback=result.feedback,
        confidence=result.confidence,
        evaluation_time_sec=result.evaluation_time_sec,
        question_type=result.question_type,
        mcq_correct=result.is_correct,
        mcq_detected_answer=result.detected_answer,
        mcq_correct_answer=result.correct_answer,
    )
    db.add(db_result)
    submission.extracted_text = result.extracted_text
    submission.ocr_confidence = result.ocr_confidence
    db.commit()
    db.refresh(db_result)

    background_tasks.add_task(_recompute_metrics_background, SessionLocal)

    return EvaluationResponse(
        submission_id=submission.id,
        question_type=result.question_type,
        final_score=result.final_score,
        max_marks=result.max_marks,
        llm_score=result.llm_score,
        similarity_score=result.similarity_score,
        mcq_correct=result.mcq_correct,
        mcq_detected_answer=result.mcq_detected_answer or None,
        mcq_correct_answer=result.mcq_correct_answer or None,
        ocr_text=result.extracted_text,
        ocr_confidence=result.ocr_confidence,
        strengths=result.strengths,
        missing_concepts=result.missing_concepts,
        feedback=result.feedback,
        confidence=result.confidence,
        evaluation_time_sec=result.evaluation_time_sec,
        rubric_details=result.rubric_details,
    )


# ── Get result ────────────────────────────────────────────

@app.get("/result/{submission_id}", summary="Get evaluation result for a submission")
async def get_result(submission_id: int, db: Session = Depends(get_db)):
    submission = db.query(Submission).filter_by(id=submission_id).first()
    if not submission:
        raise HTTPException(404, "Submission not found.")

    result = db.query(Result).filter_by(submission_id=submission_id).first()
    if not result:
        raise HTTPException(404, "Result not yet available. Run /evaluate first.")

    return {
        "submission_id": submission_id,
        "question_type": result.question_type,
        "final_score": result.final_score,
        "llm_score": result.llm_score,
        "similarity_score": result.similarity_score,
        "mcq_correct": result.mcq_correct,
        "mcq_detected_answer": result.mcq_detected_answer,
        "mcq_correct_answer": result.mcq_correct_answer,
        "strengths": json.loads(result.strengths or "[]"),
        "missing_concepts": json.loads(result.missing_concepts or "[]"),
        "feedback": result.feedback,
        "confidence": result.confidence,
        "evaluation_time_sec": result.evaluation_time_sec,
        "created_at": result.created_at.isoformat(),
    }


# ── Rubric upload ─────────────────────────────────────────

@app.post("/rubric", summary="Upload rubric criteria for a question")
async def upload_rubric(request: RubricUploadRequest, db: Session = Depends(get_db)):
    question = db.query(Question).filter_by(id=request.question_id).first()
    if not question:
        raise HTTPException(404, "Question not found.")
    db.query(Rubric).filter_by(question_id=request.question_id).delete()
    for item in request.criteria:
        db.add(Rubric(question_id=request.question_id, criterion=item.criterion, marks=item.marks))
    db.commit()
    return {"message": f"Rubric updated with {len(request.criteria)} criteria."}


# ── Stats ─────────────────────────────────────────────────

@app.get("/stats", summary="Get overall system statistics")
async def get_stats(db: Session = Depends(get_db)):
    total_submissions = db.query(Submission).count()
    evaluated = db.query(Result).count()
    results   = db.query(Result).all()

    avg_score = avg_time = 0.0
    mcq_count = open_count = 0
    if results:
        avg_score  = sum(r.final_score for r in results) / len(results)
        timed      = [r.evaluation_time_sec for r in results if r.evaluation_time_sec]
        avg_time   = sum(timed) / len(timed) if timed else 0.0
        mcq_count  = sum(1 for r in results if r.question_type == "mcq")
        open_count = sum(1 for r in results if r.question_type == "open_ended")

    return {
        "total_submissions": total_submissions,
        "evaluated": evaluated,
        "mcq_evaluated": mcq_count,
        "open_ended_evaluated": open_count,
        "average_score": round(avg_score, 2),
        "average_evaluation_time_sec": round(avg_time, 2),
    }


# ── Metrics: cached background report ────────────────────

@app.get("/metrics", summary="AI scoring accuracy metrics")
async def get_metrics(refresh: bool = False):
    if refresh or _metrics_cache["last_updated"] is None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(_executor, _recompute_metrics_background, SessionLocal)

    return {
        "last_updated": _metrics_cache.get("last_updated"),
        "total_evaluated": _metrics_cache.get("total_evaluated", 0),
        "open_ended": _metrics_cache.get("open_ended"),
        "mcq": _metrics_cache.get("mcq"),
        "note": (
            "Open-ended MAE/Kappa requires teacher ground-truth scores. "
            "Add a `teacher_score` column to Result and populate it after "
            "manual teacher review for accurate comparison."
        ),
    }


# ── Metrics: ad-hoc compute ────────────────────────────────

@app.get("/metrics/compute", summary="Compute metrics from provided score lists")
async def compute_metrics_adhoc(
    ai_scores: str,
    teacher_scores: str,
    max_marks: float = 10.0,
    question_type: str = "open_ended",
):
    try:
        ai = [float(x.strip()) for x in ai_scores.split(",") if x.strip()]
        gt = [float(x.strip()) for x in teacher_scores.split(",") if x.strip()]
    except ValueError:
        raise HTTPException(400, "Scores must be comma-separated floats, e.g. 7.5,8.0,6.0")

    if len(ai) != len(gt):
        raise HTTPException(400, f"Length mismatch: ai_scores has {len(ai)}, teacher_scores has {len(gt)}")
    if len(ai) < 2:
        raise HTTPException(400, "At least 2 score pairs are required.")

    report = compute_metrics(ai, gt, max_marks=max_marks, question_type=question_type)

    response = {
        "question_type": report.question_type,
        "n_samples": report.n_samples,
    }

    if question_type == "mcq":
        response["mcq"] = {
            "accuracy": report.mcq_accuracy,
            "accuracy_pct": f"{report.mcq_accuracy * 100:.1f}%",
            "n_correct": report.mcq_n_correct,
            "n_wrong": report.mcq_n_wrong,
        }
    else:
        response["open_ended"] = {
            "mae": report.mae,
            "pearson_r": report.pearson_r,
            "cohen_kappa": report.cohen_kappa,
            "accuracy_within_1_mark": report.accuracy_within_1,
            "accuracy_within_1_mark_pct": f"{report.accuracy_within_1 * 100:.1f}%",
            "accuracy_within_0_5_mark": report.accuracy_within_0_5,
            "accuracy_within_0_5_mark_pct": f"{report.accuracy_within_0_5 * 100:.1f}%",
            "mean_ai_score": report.mean_ai_score,
            "mean_teacher_score": report.mean_teacher_score,
        }

    return response


# ── List submissions ──────────────────────────────────────

@app.get("/submissions", summary="List all submissions with their results")
async def list_submissions(db: Session = Depends(get_db)):
    submissions = db.query(Submission).all()
    out = []
    for sub in submissions:
        entry = {
            "submission_id": sub.id,
            "student_code": sub.student.student_code if sub.student else None,
            "image_path": sub.image_path,
            "submitted_at": sub.submitted_at.isoformat() if sub.submitted_at else None,
            "ocr_confidence": sub.ocr_confidence,
            "result": None,
        }
        if sub.result:
            entry["result"] = {
                "question_type": sub.result.question_type,
                "final_score": sub.result.final_score,
                "mcq_correct": sub.result.mcq_correct,
                "feedback": sub.result.feedback,
            }
        out.append(entry)
    return out


# ─────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_excludes=["tests/*", "*.pyc"],
    )