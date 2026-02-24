

# Sathyabama IntelliGrade

### AI System for Automatic Evaluation of Handwritten Subjective Answers

An intelligent grading system that reads handwritten student answers, understands them using NLP and large language models, and generates marks with detailed feedback.

Developed for academic research and evaluation workflows at
Sathyabama Institute of Science and Technology.

---

# Overview

Evaluating subjective answers is time-consuming, inconsistent, and difficult to scale. IntelliGrade-H automates this process using modern AI techniques:

• Computer Vision for handwriting recognition
• Natural Language Processing for answer understanding
• Large Language Models for professor-like evaluation
• Hybrid scoring algorithms for fair grading

The system converts handwritten answers into structured feedback and marks in seconds.

---

# Key Features

• Handwritten answer recognition
• AI-based grading using LLMs
• Semantic similarity scoring
• Rubric-aware evaluation
• Detailed student feedback
• Teacher dashboard for batch grading
• API-based architecture for LMS integration

---

# Project Structure

```
IntelliGrade-H/
│
├── backend/
│   ├── api.py
│   ├── preprocessor.py
│   ├── ocr_module.py
│   ├── text_processor.py
│   ├── similarity.py
│   ├── llm_evaluator.py
│   ├── rubric_matcher.py
│   ├── evaluator.py
│   ├── metrics.py
│   └── database.py
│
├── frontend/
│   └── dashboard.py
│
├── models/
│   └── train_trocr.py
│
├── datasets/
│   └── collect_dataset.py
│
├── prompts/
│   └── evaluation_prompts.py
│
├── tests/
│   └── test_all.py
│
├── requirements.txt
├── docker-compose.yml
└── README.md
```

---

# System Architecture

```
Student Answer Image
        │
        ▼
Image Preprocessing
(OpenCV pipeline)
        │
        ▼
Handwriting Recognition
(TrOCR / Tesseract)
        │
        ▼
Text Processing
(spaCy / NLTK)
        │
        ▼
AI Evaluation Engine
 ┌───────────────────────────────┐
 │ Semantic Similarity (SBERT)   │
 │ LLM Examiner (Gemini)         │
 │ Rubric Matcher (DeBERTa NLI)  │
 └───────────────────────────────┘
        │
        ▼
Hybrid Score Calculation
        │
        ▼
Results + Feedback Dashboard
```

---

# Components

## Image Preprocessing

File: `backend/preprocessor.py`

Pipeline:

• Grayscale conversion
• Noise removal
• Deskew correction
• CLAHE contrast enhancement
• Otsu thresholding
• Line segmentation

Improves OCR accuracy significantly.

---

# OCR Engine

File: `backend/ocr_module.py`

Supports two engines:

Primary
Microsoft TrOCR (Transformer-based)

Fallback
Tesseract OCR

Capabilities

• Handwriting recognition
• PDF support
• Confidence scoring
• Fine-tuning support

---

# Text Processing

File: `backend/text_processor.py`

Tasks performed:

• Spell correction with academic vocabulary
• Sentence segmentation
• Tokenization
• Normalization

Libraries used:

spaCy
NLTK

---

# Semantic Similarity

File: `backend/similarity.py`

Uses Sentence-BERT to measure how close a student answer is to the teacher answer.

Features:

• Cosine similarity scoring
• Sentence-level analysis
• Fine-tuning capability

Model used:

all-MiniLM-L6-v2.

---

# LLM Evaluation

File: `backend/llm_evaluator.py`

Uses:

Gemini API

The model evaluates answers like a university professor.

Outputs:

• Marks
• Strengths
• Missing concepts
• Improvement suggestions

Responses are parsed into structured JSON.

---

# Rubric Matching

File: `backend/rubric_matcher.py`

Uses zero-shot classification with DeBERTa NLI.

Benefits:

• No training required
• Detects rubric coverage
• Works for any subject

Example rubric:

Definition
Explanation
Example
Applications.

---

# Evaluation Engine

File: `backend/evaluator.py`

This is the core orchestration module.

Responsibilities:

• Runs the full pipeline
• Aggregates outputs from all models
• Calculates final score
• Generates feedback

Scoring formula

```
Final Score =
0.6 × LLM Score
+
0.4 × Similarity × Max Marks
```

---

# API Layer

File: `backend/api.py`

Built with FastAPI.

Endpoints

| Method | Endpoint     | Purpose               |
| ------ | ------------ | --------------------- |
| POST   | /upload      | Upload student answer |
| POST   | /ocr/{id}    | Run OCR               |
| POST   | /evaluate    | Full AI evaluation    |
| GET    | /result/{id} | Fetch result          |
| POST   | /rubric      | Upload grading rubric |
| GET    | /stats       | System analytics      |
| GET    | /health      | Service health check  |

Interactive docs available at `/docs`.

---

# Database

File: `backend/database.py`

ORM: SQLAlchemy.

Tables:

Students
Questions
Submissions
Results
Rubrics

Supports SQLite (development) and PostgreSQL (production).

---

# Teacher Dashboard

File: `frontend/dashboard.py`

Built using Streamlit.

Features:

Single evaluation
Batch evaluation
Result analytics
Settings configuration

---

# Dataset Tools

File: `datasets/collect_dataset.py`

Includes:

• Labeling interface
• Synthetic dataset generator
• Dataset statistics
• Export for OCR training

Useful for bootstrapping when real data is limited.

---

# Fine-Tuning TrOCR

```
python models/train_trocr.py train \
  --dataset datasets/training \
  --output models/trocr-finetuned \
  --epochs 5
```

Evaluation:

```
python models/train_trocr.py eval
```

Metric used:

Character Error Rate (CER).

---

# Installation

### Clone Repository

```
git clone <repo>
cd IntelliGrade-H
```

Install dependencies

```
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

# Environment Setup

Create `.env`

```
GEMINI_API_KEY=your_key
DATABASE_URL=sqlite:///./intelligrade.db
OCR_ENGINE=trocr
LLM_WEIGHT=0.6
SIMILARITY_WEIGHT=0.4
UPLOAD_DIR=./uploads
```

Get Gemini key from
[https://aistudio.google.com](https://aistudio.google.com)

---

# Run Backend

```
uvicorn backend.api:app --reload
```

Open:

```
http://localhost:8000/docs
```

---

# Run Dashboard

```
streamlit run frontend/dashboard.py
```

Open:

```
http://localhost:8501
```

---

# Running Tests

```
pytest tests/
```

Includes tests for:

OCR
Similarity
API endpoints
Evaluation pipeline.

---

# Docker Deployment

```
docker-compose up --build
```

Services started:

API server
Database
Dashboard.

---

# Evaluation Metrics

System accuracy is validated against teacher grading.

| Metric              | Target   |
| ------------------- | -------- |
| Mean Absolute Error | < 1 mark |
| Pearson Correlation | > 0.80   |
| Cohen’s Kappa       | > 0.70   |
| Accuracy ±1 mark    | > 85%    |

---

# Technology Stack

| Layer           | Technology       |
| --------------- | ---------------- |
| OCR             | TrOCR, Tesseract |
| NLP             | spaCy, NLTK      |
| Embeddings      | Sentence-BERT    |
| LLM             | Gemini           |
| Rubric Matching | DeBERTa          |
| Backend         | FastAPI          |
| Frontend        | Streamlit        |
| Database        | PostgreSQL       |
| Deployment      | Docker           |

---

# Ethical Considerations

Student identities are anonymized.

Data is processed locally except for LLM requests.

AI grades must be validated by instructors before official use.

---

# Future Improvements

• Diagram recognition
• Mathematical expression grading
• Multilingual support (Tamil / Hindi / English)
• Continuous learning from teacher corrections
• LMS integrations

---

# Research Potential

This project can evolve into:

• An academic publication
• A startup-grade grading platform
• A scalable assessment system for universities

---

# Team

Department of Computer Science and Engineering
Sathyabama Institute of Science and Technology
