"""
IntelliGrade-H - Database Models
SQLAlchemy ORM models for the application.
"""

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
# Models
# ─────────────────────────────────────────────────────────

class Student(Base):
    __tablename__ = "students"

    id            = Column(Integer, primary_key=True, index=True)
    student_code  = Column(String(50), unique=True, index=True)   # anonymized ID
    created_at    = Column(DateTime, default=datetime.utcnow)

    submissions = relationship("Submission", back_populates="student")


class Question(Base):
    __tablename__ = "questions"

    id              = Column(Integer, primary_key=True, index=True)
    exam_name       = Column(String(200))
    question_number = Column(Integer)
    question_text   = Column(Text)
    teacher_answer  = Column(Text)
    max_marks       = Column(Float, default=10.0)
    created_at      = Column(DateTime, default=datetime.utcnow)

    rubrics     = relationship("Rubric", back_populates="question")
    submissions = relationship("Submission", back_populates="question")


class Rubric(Base):
    __tablename__ = "rubrics"

    id           = Column(Integer, primary_key=True, index=True)
    question_id  = Column(Integer, ForeignKey("questions.id"))
    criterion    = Column(String(200))   # e.g. "Definition", "Example"
    marks        = Column(Float, default=1.0)
    is_required  = Column(Boolean, default=False)

    question = relationship("Question", back_populates="rubrics")


class Submission(Base):
    __tablename__ = "submissions"

    id               = Column(Integer, primary_key=True, index=True)
    student_id       = Column(Integer, ForeignKey("students.id"))
    question_id      = Column(Integer, ForeignKey("questions.id"))
    image_path       = Column(String(500))
    extracted_text   = Column(Text)
    ocr_confidence   = Column(Float)
    submitted_at     = Column(DateTime, default=datetime.utcnow)

    student  = relationship("Student", back_populates="submissions")
    question = relationship("Question", back_populates="submissions")
    result   = relationship("Result", back_populates="submission", uselist=False)


class Result(Base):
    __tablename__ = "results"

    id                  = Column(Integer, primary_key=True, index=True)
    submission_id       = Column(Integer, ForeignKey("submissions.id"), unique=True)
    final_score         = Column(Float)
    llm_score           = Column(Float)
    similarity_score    = Column(Float)
    strengths           = Column(Text)
    missing_concepts    = Column(Text)
    feedback            = Column(Text)
    confidence          = Column(Float)
    evaluation_time_sec = Column(Float)
    created_at          = Column(DateTime, default=datetime.utcnow)

    submission = relationship("Submission", back_populates="result")


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
