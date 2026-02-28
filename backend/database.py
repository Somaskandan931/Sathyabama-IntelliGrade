"""
IntelliGrade-H - Database Models
SQLAlchemy ORM models for the application.

v2 additions:
  • MetricsSnapshot table — persists computed metrics across server restarts.
    Written by _recompute_metrics_background() in api.py after every /evaluate.
    Read back on startup so GET /metrics never returns null after a restart.
"""

import json
from sqlalchemy import (
    Column, Integer, String, Float, Text, DateTime, ForeignKey, Boolean
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

Base = declarative_base()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./intelligrade.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# ─────────────────────────────────────────────────────────
# Existing Models (unchanged)
# ─────────────────────────────────────────────────────────

class Student(Base):
    __tablename__ = "students"

    id           = Column(Integer, primary_key=True, index=True)
    student_code = Column(String(50), unique=True, index=True)
    created_at   = Column(DateTime, default=datetime.utcnow)

    submissions = relationship("Submission", back_populates="student")


class Question(Base):
    __tablename__ = "questions"

    id              = Column(Integer, primary_key=True, index=True)
    exam_name       = Column(String(200))
    question_number = Column(Integer)
    question_text   = Column(Text)
    question_type   = Column(String(20), default="open_ended")
    teacher_answer  = Column(Text)
    correct_option  = Column(String(1))
    mcq_options     = Column(Text)
    max_marks       = Column(Float, default=10.0)
    created_at      = Column(DateTime, default=datetime.utcnow)

    rubrics     = relationship("Rubric", back_populates="question")
    submissions = relationship("Submission", back_populates="question")


class Rubric(Base):
    __tablename__ = "rubrics"

    id          = Column(Integer, primary_key=True, index=True)
    question_id = Column(Integer, ForeignKey("questions.id"))
    criterion   = Column(String(200))
    marks       = Column(Float, default=1.0)
    is_required = Column(Boolean, default=False)

    question = relationship("Question", back_populates="rubrics")


class Submission(Base):
    __tablename__ = "submissions"

    id             = Column(Integer, primary_key=True, index=True)
    student_id     = Column(Integer, ForeignKey("students.id"))
    question_id    = Column(Integer, ForeignKey("questions.id"))
    image_path     = Column(String(500))
    extracted_text = Column(Text)
    ocr_confidence = Column(Float)
    submitted_at   = Column(DateTime, default=datetime.utcnow)

    student  = relationship("Student", back_populates="submissions")
    question = relationship("Question", back_populates="submissions")
    result   = relationship("Result", back_populates="submission", uselist=False)


class Result(Base):
    __tablename__ = "results"

    id                  = Column(Integer, primary_key=True, index=True)
    submission_id       = Column(Integer, ForeignKey("submissions.id"), unique=True)

    question_type       = Column(String(20), default="open_ended")

    final_score         = Column(Float)
    confidence          = Column(Float)
    evaluation_time_sec = Column(Float)
    feedback            = Column(Text)
    created_at          = Column(DateTime, default=datetime.utcnow)

    llm_score           = Column(Float, default=0.0)
    similarity_score    = Column(Float, default=0.0)
    strengths           = Column(Text)
    missing_concepts    = Column(Text)

    mcq_correct         = Column(Boolean, nullable=True)
    mcq_detected_answer = Column(String(1), nullable=True)
    mcq_correct_answer  = Column(String(1), nullable=True)

    submission = relationship("Submission", back_populates="result")


# ─────────────────────────────────────────────────────────
# NEW: MetricsSnapshot — persists computed metrics to DB
# ─────────────────────────────────────────────────────────

class MetricsSnapshot(Base):
    """
    Stores the latest computed metrics as a JSON blob.

    There is always exactly ONE row in this table (id=1).
    _recompute_metrics_background() does an upsert on id=1 after
    every /evaluate call, so the row is always current.

    On server startup, api.py reads this row to pre-populate
    _metrics_cache, so GET /metrics never returns null after a restart.

    Columns
    -------
    open_ended_json : JSON-serialised open-ended metrics dict (or NULL)
    mcq_json        : JSON-serialised MCQ metrics dict (or NULL)
    total_evaluated : total number of Result rows when last computed
    computed_at     : UTC timestamp of last computation
    """
    __tablename__ = "metrics_snapshots"

    id              = Column(Integer, primary_key=True, default=1)
    open_ended_json = Column(Text, nullable=True)   # JSON string or NULL
    mcq_json        = Column(Text, nullable=True)   # JSON string or NULL
    total_evaluated = Column(Integer, default=0)
    computed_at     = Column(DateTime, default=datetime.utcnow)

    # ── Convenience helpers ───────────────────────────────

    @property
    def open_ended(self):
        """Return the open-ended metrics as a dict, or None."""
        return json.loads(self.open_ended_json) if self.open_ended_json else None

    @property
    def mcq(self):
        """Return the MCQ metrics as a dict, or None."""
        return json.loads(self.mcq_json) if self.mcq_json else None

    @classmethod
    def upsert(cls, db, open_ended_dict, mcq_dict, total_evaluated: int):
        """
        Insert or update the single metrics snapshot row (id=1).
        Called from _recompute_metrics_background() in api.py.
        """
        snapshot = db.query(cls).filter_by(id=1).first()
        if snapshot is None:
            snapshot = cls(id=1)
            db.add(snapshot)

        snapshot.open_ended_json = json.dumps(open_ended_dict) if open_ended_dict else None
        snapshot.mcq_json        = json.dumps(mcq_dict) if mcq_dict else None
        snapshot.total_evaluated = total_evaluated
        snapshot.computed_at     = datetime.utcnow()
        db.commit()
        return snapshot

    @classmethod
    def load(cls, db):
        """
        Load the current snapshot from the DB.
        Returns the MetricsSnapshot row, or None if it doesn't exist yet.
        """
        return db.query(cls).filter_by(id=1).first()


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created.")