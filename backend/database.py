"""
IntelliGrade-H - Database Models
SQLAlchemy ORM models for the application.

v3 additions (dynamic question paper support):
  • ExamPaper   — stores parsed exam metadata (course, marks, date …)
  • ExamPart    — stores parts within a paper (Part-A 2 marks each, Part-B 12 marks each …)
  • ExamQuestion — stores individual questions with marks extracted from the paper PDF

v4.1 additions:
  • StudentBooklet    — one row per uploaded student answer booklet PDF
  • StudentAnswerText — one row per (booklet, question_number) answer

v2 additions:
  • MetricsSnapshot — persists computed metrics across server restarts.

FIXES:
  • ExamQuestion.paper_id_fk used consistently everywhere (removed paper_id alias confusion)
  • Student model includes 'name' column (used by booklet upload)
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
# v3: Dynamic Exam Paper Structure
# ─────────────────────────────────────────────────────────

class ExamPaper(Base):
    """
    Stores the parsed structure of an uploaded question paper PDF.

    paper_id     : unique slug e.g. "S11BLH41_CAE1_Set-A"
    total_marks  : extracted from the paper PDF, never hardcoded
    """
    __tablename__ = "exam_papers"

    id              = Column(Integer, primary_key=True, index=True)
    paper_id        = Column(String(200), unique=True, index=True)
    pdf_path        = Column(String(500), nullable=True)
    course_code     = Column(String(50),  index=True)
    course_name     = Column(String(200))
    exam_name       = Column(String(200))
    total_marks     = Column(Float, default=0.0)
    duration_hours  = Column(Float, default=2.0)
    exam_date       = Column(String(30))
    batch           = Column(String(30))
    programme       = Column(String(100))
    semester        = Column(String(20))
    set_name        = Column(String(20), default="")
    raw_text        = Column(Text, nullable=True)
    parsed_at       = Column(DateTime, default=datetime.utcnow)

    parts       = relationship("ExamPart",     back_populates="paper",   cascade="all, delete-orphan")
    questions   = relationship("ExamQuestion", back_populates="paper",   cascade="all, delete-orphan")
    submissions = relationship("Submission",   back_populates="exam_paper")


class ExamPart(Base):
    """
    One part of an exam paper (e.g. Part-A, Part-B).
    marks_per_question is extracted from the paper header.
    """
    __tablename__ = "exam_parts"

    id                  = Column(Integer, primary_key=True, index=True)
    paper_id_fk         = Column(Integer, ForeignKey("exam_papers.id"), index=True)
    part_name           = Column(String(50))
    marks_per_question  = Column(Float, default=2.0)
    num_questions       = Column(Integer, default=0)
    instructions        = Column(String(200), default="Answer ALL questions")

    paper = relationship("ExamPaper", back_populates="parts")


class ExamQuestion(Base):
    """
    One question from an exam paper.

    marks      : extracted from the paper PDF — NEVER hardcoded
    paper_id_fk: FK to ExamPaper.id  (note: NOT paper_id string slug)

    For OR-alternative questions: is_or_option=True marks the alternative.
    """
    __tablename__ = "exam_questions"

    id              = Column(Integer, primary_key=True, index=True)
    paper_id_fk     = Column(Integer, ForeignKey("exam_papers.id"), index=True)
    question_number = Column(Integer)
    question_text   = Column(Text)
    marks           = Column(Float)           # from paper PDF, never hardcoded
    part_name       = Column(String(50))
    question_type   = Column(String(30), default="open_ended")
    is_or_option    = Column(Boolean, default=False)
    teacher_answer  = Column(Text, nullable=True)
    correct_option  = Column(String(5), nullable=True)
    created_at      = Column(DateTime, default=datetime.utcnow)

    paper   = relationship("ExamPaper",  back_populates="questions")
    rubrics = relationship("Rubric",     back_populates="exam_question")


# ─────────────────────────────────────────────────────────
# Core Models
# ─────────────────────────────────────────────────────────

class Student(Base):
    __tablename__ = "students"

    id           = Column(Integer, primary_key=True, index=True)
    student_code = Column(String(50), unique=True, index=True)
    name         = Column(String(200), nullable=True, default="")
    created_at   = Column(DateTime, default=datetime.utcnow)

    submissions = relationship("Submission", back_populates="student")


class Question(Base):
    """Legacy question model — kept for backward compatibility."""
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

    rubrics     = relationship("Rubric",      back_populates="question")
    submissions = relationship("Submission",  back_populates="question")


class Rubric(Base):
    __tablename__ = "rubrics"

    id               = Column(Integer, primary_key=True, index=True)
    question_id      = Column(Integer, ForeignKey("questions.id"),      nullable=True)
    exam_question_id = Column(Integer, ForeignKey("exam_questions.id"), nullable=True)
    criterion        = Column(String(200))
    marks            = Column(Float, default=1.0)
    is_required      = Column(Boolean, default=False)

    question      = relationship("Question",     back_populates="rubrics")
    exam_question = relationship("ExamQuestion", back_populates="rubrics")


class Submission(Base):
    __tablename__ = "submissions"

    id               = Column(Integer, primary_key=True, index=True)
    student_id       = Column(Integer, ForeignKey("students.id"))
    question_id      = Column(Integer, ForeignKey("questions.id"),      nullable=True)
    exam_paper_id    = Column(Integer, ForeignKey("exam_papers.id"),    nullable=True)
    exam_question_id = Column(Integer, ForeignKey("exam_questions.id"), nullable=True)
    image_path       = Column(String(500))
    extracted_text   = Column(Text)
    ocr_confidence   = Column(Float)
    submitted_at     = Column(DateTime, default=datetime.utcnow)

    student    = relationship("Student",      back_populates="submissions")
    question   = relationship("Question",     back_populates="submissions")
    exam_paper = relationship("ExamPaper",    back_populates="submissions")
    result     = relationship("Result",       back_populates="submission", uselist=False,
                              cascade="all, delete-orphan")


class Result(Base):
    __tablename__ = "results"

    id                  = Column(Integer, primary_key=True, index=True)
    submission_id       = Column(Integer, ForeignKey("submissions.id"), unique=True)
    question_type       = Column(String(20), default="open_ended")
    final_score         = Column(Float)
    max_marks           = Column(Float, default=10.0)
    confidence          = Column(Float)
    evaluation_time_sec = Column(Float)
    feedback            = Column(Text)
    created_at          = Column(DateTime, default=datetime.utcnow)

    llm_score        = Column(Float, default=0.0)
    similarity_score = Column(Float, default=0.0)
    strengths        = Column(Text)
    missing_concepts = Column(Text)

    mcq_correct         = Column(Boolean,  nullable=True)
    mcq_detected_answer = Column(String(1), nullable=True)
    mcq_correct_answer  = Column(String(1), nullable=True)

    submission = relationship("Submission", back_populates="result")


# ─────────────────────────────────────────────────────────
# MetricsSnapshot (v2)
# ─────────────────────────────────────────────────────────

class MetricsSnapshot(Base):
    """
    Stores the latest computed metrics as a JSON blob.
    Always exactly ONE row in this table (id=1).
    """
    __tablename__ = "metrics_snapshots"

    id              = Column(Integer, primary_key=True, default=1)
    open_ended_json = Column(Text, nullable=True)
    mcq_json        = Column(Text, nullable=True)
    total_evaluated = Column(Integer, default=0)
    computed_at     = Column(DateTime, default=datetime.utcnow)

    @property
    def open_ended(self):
        return json.loads(self.open_ended_json) if self.open_ended_json else None

    @property
    def mcq(self):
        return json.loads(self.mcq_json) if self.mcq_json else None

    @classmethod
    def upsert(cls, db, open_ended_dict, mcq_dict, total_evaluated: int):
        snapshot = db.query(cls).filter_by(id=1).first()
        if snapshot is None:
            snapshot = cls(id=1)
            db.add(snapshot)
        snapshot.open_ended_json = json.dumps(open_ended_dict) if open_ended_dict else None
        snapshot.mcq_json        = json.dumps(mcq_dict)        if mcq_dict        else None
        snapshot.total_evaluated = total_evaluated
        snapshot.computed_at     = datetime.utcnow()
        db.commit()
        return snapshot

    @classmethod
    def load(cls, db):
        return db.query(cls).filter_by(id=1).first()


# ─────────────────────────────────────────────────────────
# v4.1: Student Booklet Tables
# ─────────────────────────────────────────────────────────

class StudentBooklet(Base):
    """
    One row per uploaded student answer booklet PDF.
    Links to a Student and an ExamPaper.
    """
    __tablename__ = "student_booklets"

    id            = Column(Integer, primary_key=True, index=True)
    student_id    = Column(Integer, ForeignKey("students.id"),     nullable=True)
    exam_paper_id = Column(Integer, ForeignKey("exam_papers.id"),  nullable=True)

    pdf_path    = Column(String(500))
    total_pages = Column(Integer, default=0)

    # Cover metadata (extracted by LLM from cover page OCR)
    roll_number     = Column(String(50),  default="")
    register_number = Column(String(50),  default="")
    set_name        = Column(String(20),  default="")
    course_code     = Column(String(20),  default="")
    course_name     = Column(String(200), default="")
    exam_name       = Column(String(200), default="")
    programme       = Column(String(100), default="")
    semester        = Column(String(10),  default="")
    date_of_exam    = Column(String(20),  default="")

    raw_full_ocr = Column(Text,     default="")
    parsed_at    = Column(DateTime, default=datetime.utcnow)

    student    = relationship("Student",           backref="booklets")
    exam_paper = relationship("ExamPaper",         backref="student_booklets")
    answers    = relationship("StudentAnswerText", back_populates="booklet",
                              cascade="all, delete-orphan")


class StudentAnswerText(Base):
    """
    One row per (booklet, question_number) — the student's answer text for one question.
    Feeds into the Submission/Result pipeline for evaluation.
    """
    __tablename__ = "student_answer_texts"

    id               = Column(Integer, primary_key=True, index=True)
    booklet_id       = Column(Integer, ForeignKey("student_booklets.id"), index=True)
    exam_question_id = Column(Integer, ForeignKey("exam_questions.id"),   nullable=True)

    question_number = Column(Integer)
    part_name       = Column(String(20),  default="")
    is_or_option    = Column(Boolean,     default=False)
    answer_text     = Column(Text,        default="")
    raw_ocr_text    = Column(Text,        default="")

    # Set after evaluation is triggered
    submission_id = Column(Integer, ForeignKey("submissions.id"), nullable=True)
    evaluated     = Column(Boolean, default=False)

    booklet       = relationship("StudentBooklet",  back_populates="answers")
    exam_question = relationship("ExamQuestion",    backref="student_answers")
    submission    = relationship("Submission",      backref="student_answer_text")


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
    _migrate_db()
    print("✅ Database tables created / verified.")


def _migrate_db():
    """
    Apply lightweight ALTER TABLE migrations for columns added after initial release.
    Safe to run on every startup — idempotent (uses try-except on column-already-exists).
    """
    migrations = [
        # v2.3
        "ALTER TABLE results ADD COLUMN max_marks REAL DEFAULT 10.0",
        # v3.0 — new columns on submissions for exam paper linking
        "ALTER TABLE submissions ADD COLUMN exam_paper_id INTEGER REFERENCES exam_papers(id)",
        "ALTER TABLE submissions ADD COLUMN exam_question_id INTEGER REFERENCES exam_questions(id)",
        # v3.0 — rubric exam_question link
        "ALTER TABLE rubrics ADD COLUMN exam_question_id INTEGER REFERENCES exam_questions(id)",
        # v3.0 — LLM score breakdown on results
        "ALTER TABLE results ADD COLUMN llm_score REAL DEFAULT 0.0",
        "ALTER TABLE results ADD COLUMN similarity_score REAL DEFAULT 0.0",
        "ALTER TABLE results ADD COLUMN question_type TEXT DEFAULT 'open_ended'",
        "ALTER TABLE results ADD COLUMN mcq_correct INTEGER",
        "ALTER TABLE results ADD COLUMN mcq_detected_answer TEXT",
        "ALTER TABLE results ADD COLUMN mcq_correct_answer TEXT",
        # v3.0 — OCR confidence on submissions
        "ALTER TABLE submissions ADD COLUMN ocr_confidence REAL",
        "ALTER TABLE submissions ADD COLUMN extracted_text TEXT",
        # v4.1 — student booklet
        "ALTER TABLE student_answer_texts ADD COLUMN submission_id INTEGER REFERENCES submissions(id)",
        "ALTER TABLE student_answer_texts ADD COLUMN evaluated INTEGER DEFAULT 0",
        # v5.0 — student name column
        "ALTER TABLE students ADD COLUMN name TEXT DEFAULT ''",
    ]

    raw_conn = engine.raw_connection()
    try:
        cursor = raw_conn.cursor()
        for sql in migrations:
            try:
                cursor.execute(sql)
                raw_conn.commit()
            except Exception:
                raw_conn.rollback()   # column already exists — safe to ignore
    finally:
        raw_conn.close()


import logging as _logging
logger = _logging.getLogger(__name__)