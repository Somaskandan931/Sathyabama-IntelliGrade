# IntelliGrade-H

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

**AI System for Automatic Evaluation of Handwritten Subjective Answers**

IntelliGrade-H is an advanced AI-powered grading system designed to automatically evaluate handwritten student answers. The platform combines modern **computer vision, natural language processing, and large language models** to analyze student responses and generate accurate marks with detailed feedback.

The system is designed for **universities, academic research, and scalable exam evaluation workflows**.

---

## Overview

Evaluating subjective answers is traditionally time-consuming, inconsistent between evaluators, and difficult to scale for large classes. IntelliGrade-H solves this using a **multi-stage AI pipeline** that converts handwritten answers into structured feedback and marks within seconds.

The system integrates:

- Handwriting Recognition (EasyOCR + TrOCR Ensemble)
- Semantic Answer Evaluation (Sentence-BERT)
- Rubric-Aware Grading (zero-shot NLI)
- AI Feedback Generation (Claude primary, Groq fallback)
- Teacher Analytics Dashboard (Streamlit)

---

## Key Features

- Automatic handwriting recognition from scanned answer sheets and PDFs
- Smart image preprocessing — auto-upscale, deskew, adaptive denoising, smart thresholding
- Ensemble OCR — EasyOCR fast path, TrOCR for low-confidence pages
- Semantic similarity scoring between student and teacher answers
- Rubric-aware grading using zero-shot AI models
- AI-generated feedback highlighting strengths and missing concepts
- Question type auto-classification (MCQ, open-ended, numerical, diagram, etc.)
- Batch grading for entire exam submissions
- Teacher dashboard for analytics and grading review
- REST API for LMS integration

---

## System Architecture

```
Answer Sheet (image / PDF)
         │
         ▼
Image Preprocessing              ← preprocessor.py
(upscale · denoise · deskew · CLAHE · threshold)
         │
         ▼
OCR Ensemble                     ← ocr_module.py
(EasyOCR → TrOCR on low confidence)
         │
         ▼
Text Processing                  ← text_processor.py
(clean · normalise · tokenise)
         │
         ▼
Question Classifier              ← question_classifier.py
(MCQ · open_ended · numerical · diagram · …)
         │
         ▼
Evaluation Engine                ← evaluator.py
 ├─ Semantic Similarity          ← similarity.py      (Sentence-BERT)
 ├─ Rubric Matcher               ← rubric_matcher.py  (zero-shot NLI)
 ├─ Keyword Coverage             ← evaluator.py
 └─ LLM Evaluator                ← llm_evaluator.py   (Claude → Groq)
         │
         ▼
Hybrid Scoring Engine
Final = 0.40×LLM + 0.25×Similarity + 0.20×Rubric + 0.10×Keyword + 0.05×Length
         │
         ▼
Teacher Dashboard                ← frontend/dashboard.py
```

<p align="center">
  <img src="assets/ChatGPT Image Mar 6, 2026, 11_19_15 AM.png" alt="IntelliGrade-H System Architecture Diagram" width="750"/>
</p>

---

## Grading Engine

IntelliGrade-H uses a **hybrid grading model** instead of relying on any single scoring method.

```
Final Score =
  0.40 × LLM Evaluation         (Claude / Groq professor-style assessment)
  0.25 × Semantic Similarity     (Sentence-BERT meaning-level comparison)
  0.20 × Rubric Coverage         (zero-shot NLI per rubric criterion)
  0.10 × Keyword Coverage        (key technical terms detected)
  0.05 × Length Normalisation    (penalises blank / trivially short answers)
```

Weights are configurable in `.env`.

---

## Technology Stack

### LLM Providers — Claude (Primary), Groq (Fallback)

**Claude** (`claude-haiku-4-5-20251001`) is the primary LLM provider. It produces nuanced, professor-style partial-credit scoring with explicit rationale. Set `ANTHROPIC_API_KEY` in `.env`.  
To use the higher-quality model, set `CLAUDE_MODEL=claude-sonnet-4-6` in `.env`.

**Groq** (`llama-3.3-70b-versatile`) is the fallback provider when Claude is unavailable. It offers fast inference suitable for real-time grading. Set `GROQ_API_KEY` in `.env`.

**Rule-based fallback** — if both cloud providers are unavailable, the system returns a safe partial score so grading is never fully blocked.

### OCR

**EasyOCR** is the primary fast-path OCR engine. It handles both printed and handwritten text with no system dependencies. Runs in ~1–3 s/page on CPU.

**TrOCR** (`microsoft/trocr-small-handwritten`) is invoked automatically when EasyOCR confidence falls below 65%. It is a transformer-based model specifically trained on handwritten text and gives the best accuracy for messy or cursive writing. Runs in ~8–20 s/page on CPU.

Set `OCR_ENGINE=ensemble` in `.env` to use the best-of-both strategy (recommended).

### Computer Vision

**OpenCV** performs all image preprocessing on scanned exam sheets: grayscale conversion, adaptive denoising, skew correction, CLAHE contrast enhancement, and smart binarisation. It also segments PDF pages into line crops for line-by-line OCR.

**PyTorch** is the deep learning runtime powering TrOCR, Sentence-BERT, and the rubric NLI model.

### NLP

**Sentence-BERT** (`all-MiniLM-L6-v2`) generates semantic embeddings of student and teacher answers, allowing meaning-based similarity scoring rather than simple keyword matching.

**spaCy** is used for sentence segmentation and tokenisation in the text processing pipeline.

### Backend

**FastAPI** provides the REST API layer that orchestrates the entire grading pipeline — file upload, OCR, evaluation, result storage, and metrics.

**Uvicorn** is the ASGI server that runs FastAPI.

**SQLAlchemy + SQLite** stores all grading results, uploaded files, and teacher-defined questions in a local database.

### Frontend

**Streamlit** provides the interactive teacher dashboard where instructors can upload exam sheets, review AI grades, analyse class performance, and export grading reports.

---

## Project Structure

```
IntelliGrade-H/
│
├── backend/
│   ├── api.py                  REST API — all HTTP endpoints, file upload, evaluation routing
│   ├── evaluator.py            Core evaluation engine — orchestrates OCR → NLP → LLM → score
│   ├── llm_provider.py         Multi-provider LLM client (Claude primary, Groq fallback)
│   ├── llm_evaluator.py        Builds LLM prompts and parses evaluation JSON responses
│   ├── ocr_module.py           OCR engines — EasyOCR, TrOCR, Ensemble; PDF page extraction
│   ├── preprocessor.py         Image preprocessing — upscale, denoise, deskew, threshold
│   ├── similarity.py           Sentence-BERT semantic similarity between answers
│   ├── rubric_matcher.py       Zero-shot NLI rubric criterion detection
│   ├── question_classifier.py  Auto-classifies question type (MCQ, open-ended, numerical…)
│   ├── text_processor.py       OCR output cleaning — spell correction, normalisation, tokenisation
│   ├── schemas.py              Pydantic v2 request/response schemas for the API
│   ├── database.py             SQLAlchemy models, DB init, and startup migration
│   ├── metrics.py              Grading accuracy metrics — MAE, Pearson r, Cohen's Kappa
│   └── config.py               All settings loaded from .env (API keys, weights, paths)
│
├── frontend/
│   └── dashboard.py            Streamlit teacher dashboard — upload, review, analytics, export
│
├── prompts/
│   └── evaluation_prompts.py   LLM prompt templates — standard, CS/engineering, rubric, strict
│
├── scripts/
│   ├── collect_dataset.py      Scrapes or organises labelled handwriting datasets for training
│   ├── train_trocr.py          Fine-tunes TrOCR on a custom handwriting dataset
│   ├── finetune_trocr.py       Alternative fine-tuning entry point with advanced options
│   ├── benchmark.py            Benchmarks OCR engine speed and accuracy on test images
│   ├── evaluate_metrics.py     Runs grading accuracy evaluation against teacher ground truth
│   └── create_db.py            One-time script to initialise the SQLite database from scratch
│
├── tests/
│   ├── test_all.py             End-to-end integration tests for the full grading pipeline
│   ├── test_metrics.py         Unit tests for scoring and metrics calculations
│   └── conftest.py             Pytest fixtures and shared test configuration
│
├── uploads/                    Uploaded exam sheets stored here (auto-created)
├── intelligrade.db             SQLite database (auto-created on first run)
├── .env                        Your API keys and settings (create from .env.example)
├── .env.example                Template showing all configurable settings
├── requirements.txt            Python package dependencies
├── run.py                      One-command launcher for API + dashboard together
├── docker-compose.yml          Docker deployment configuration
└── README.md                   This file
```


---

## Setup & Running

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Install the spaCy language model:

```bash
python -m spacy download en_core_web_sm
```

---

### 2. Configure environment

Copy the example file and fill in your API keys:

```bash
copy .env.example .env        # Windows
cp .env.example .env          # Mac / Linux
```

Minimum required settings in `.env`:

```env
# Primary LLM — Claude (best evaluation quality)
ANTHROPIC_API_KEY=sk-ant-...
CLAUDE_MODEL=claude-haiku-4-5-20251001

# Fallback LLM — Groq
GROQ_API_KEY=gsk_...
GROQ_MODEL=llama-3.3-70b-versatile

# OCR engine — ensemble gives best handwriting accuracy
OCR_ENGINE=ensemble

# Scoring weights (must sum to 1.0)
LLM_WEIGHT=0.40
SIMILARITY_WEIGHT=0.25
RUBRIC_WEIGHT=0.20
KEYWORD_WEIGHT=0.10
LENGTH_WEIGHT=0.05
```

> **Tip:** Set `CLAUDE_MODEL=claude-sonnet-4-6` for higher evaluation quality at slightly higher cost.

---

### 3. Run the system

**Recommended — start everything with one command:**

```bash
python run.py
```

This starts both the API and the dashboard together.

| Service | URL |
|---|---|
| Teacher Dashboard | http://localhost:8501 |
| API | http://localhost:8000 |
| API Docs (Swagger) | http://localhost:8000/docs |

---

**Or start them separately:**

```bash
# Terminal 1 — Backend API
uvicorn backend.api:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 — Teacher Dashboard
streamlit run frontend/dashboard.py
```

---

**Other run modes:**

```bash
python run.py api     # API only
python run.py ui      # Dashboard only
python run.py init    # Initialise database only (first run)
```

---

### 4. Docker (optional)

```bash
docker-compose up --build
```

---

## Evaluation Metrics

IntelliGrade-H compares AI scores against teacher-provided ground truth to validate grading accuracy.

| Metric | Target |
|---|---|
| Mean Absolute Error | < 0.8 |
| Pearson Correlation | > 0.85 |
| Cohen's Kappa | > 0.75 |
| Accuracy within ±1 mark | > 90% |

Run the metrics evaluation script against your own labelled data:

```bash
python scripts/evaluate_metrics.py
```

---

## Configuration Reference

All settings are loaded from `.env`. See `.env.example` for the full list.

| Key | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Claude API key (primary LLM) |
| `CLAUDE_MODEL` | `claude-haiku-4-5-20251001` | Claude model — swap to `claude-sonnet-4-6` for higher quality |
| `GROQ_API_KEY` | — | Groq API key (fallback LLM) |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model |
| `OCR_ENGINE` | `easyocr` | `easyocr` · `trocr` · `ensemble` (recommended) |
| `TROCR_MODEL_PATH` | `microsoft/trocr-small-handwritten` | TrOCR weights |
| `OCR_WORKERS` | `4` | Parallel OCR/evaluation threads |
| `LLM_MAX_TOKENS` | `1500` | Max tokens per LLM response |
| `LLM_TEMPERATURE` | `0.1` | LLM temperature (lower = more consistent) |
| `LLM_WEIGHT` | `0.40` | LLM score weight in hybrid formula |
| `SIMILARITY_WEIGHT` | `0.25` | Semantic similarity weight |
| `RUBRIC_WEIGHT` | `0.20` | Rubric coverage weight |
| `KEYWORD_WEIGHT` | `0.10` | Keyword coverage weight |
| `LENGTH_WEIGHT` | `0.05` | Length normalisation weight |
| `DATABASE_URL` | `sqlite:///intelligrade.db` | Database connection string |
| `MAX_FILE_SIZE_MB` | `20` | Maximum upload file size |

---

## Ethical Considerations

IntelliGrade-H is designed with responsible AI principles:

- Student identities are anonymised
- AI grading remains advisory and requires teacher review
- Feedback is transparent and explainable
- OCR errors are handled gracefully — spelling artifacts are never penalised

---

## Future Improvements

- Mathematical equation evaluation
- Multilingual grading support
- Diagram understanding with vision models
- LMS integrations (Moodle, Google Classroom)
- Mobile scanning application
- Continuous learning from teacher corrections