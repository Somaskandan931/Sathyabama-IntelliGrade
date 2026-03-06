# Sathyabama IntelliGrade-H

> **AI System for Automatic Evaluation of Handwritten Subjective Answers**

An intelligent grading system that reads handwritten student answers, understands them using NLP and large language models, and generates marks with detailed feedback — built for academic research and evaluation workflows at **Sathyabama Institute of Science and Technology**.

---

## Overview

Evaluating subjective answers is time-consuming, inconsistent, and difficult to scale. **IntelliGrade-H** automates this process using modern AI techniques:

- **Computer Vision** for handwriting recognition (TrOCR)
- **Natural Language Processing** for answer understanding (spaCy, NLTK)
- **Large Language Models** for professor-like evaluation (Gemini)
- **Hybrid Scoring Algorithms** for fair, calibrated grading

The system converts handwritten answers into structured feedback and marks **in seconds**, supporting both MCQ and open-ended subjective questions.

---

## Key Features

| Feature | Description |
|---|---|
| Handwriting Recognition | TrOCR transformer + Tesseract fallback |
| LLM Grading | Professor-style evaluation via Gemini |
| Semantic Similarity | Sentence-BERT cosine scoring |
| Rubric-Aware Evaluation | Zero-shot DeBERTa NLI rubric checking |
| Detailed Feedback | Strengths, missing concepts, suggestions |
| Teacher Dashboard | Streamlit UI for batch grading and analytics |
| REST API | FastAPI backend for LMS integration |
| Metrics Tracking | MAE, Pearson r, Cohen's Kappa, accuracy reporting |
| Docker Ready | One-command deployment |

---

## Project Structure

```
IntelliGrade-H/
│
├── backend/
│   ├── api.py                  # FastAPI REST endpoints
│   ├── evaluator.py            # Core orchestration engine
│   ├── ocr_module.py           # TrOCR + Tesseract OCR
│   ├── preprocessor.py         # OpenCV image preprocessing
│   ├── text_processor.py       # Spell correction, tokenization
│   ├── similarity.py           # Sentence-BERT similarity
│   ├── llm_evaluator.py        # Gemini LLM evaluation
│   ├── rubric_matcher.py       # DeBERTa NLI rubric checking
│   ├── question_classifier.py  # Auto question-type detection
│   ├── metrics.py              # MAE, Kappa, Pearson metrics
│   ├── evaluation_prompts.py   # LLM prompt templates
│   └── database.py             # SQLAlchemy ORM models
│
├── frontend/
│   └── dashboard.py            # Streamlit teacher dashboard
│
├── models/
│   └── train_trocr.py          # TrOCR fine-tuning script
│
├── datasets/
│   └── collect_dataset.py      # Dataset collection & labeling tool
│
├── tests/
│   └── test_all.py             # Full test suite (pytest)
│
├── uploads/                    # Student answer images (auto-created)
├── .env                        # Environment variables (see Configuration)
├── requirements.txt
├── docker-compose.yml
└── README.md
```

---

## System Architecture

```
Student Answer Image
        │
        ▼
┌─────────────────────┐
│  Image Preprocessing │  ← grayscale, denoise, deskew, CLAHE, Otsu
│  (OpenCV pipeline)   │
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│ Handwriting  (OCR)  │  ← TrOCR (primary) / Tesseract (fallback)
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│   Text Processing   │  ← spell correction, normalization, tokenization
└─────────────────────┘
        │
        ▼
┌────────────────────────────────────────┐
│          AI Evaluation Engine          │
│  ┌──────────────────────────────────┐  │
│  │  Semantic Similarity (SBERT)     │  │
│  │  LLM Examiner (Gemini Flash)     │  │
│  │  Rubric Matcher (DeBERTa NLI)    │  │
│  └──────────────────────────────────┘  │
└────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│  Hybrid Score = 0.6 × LLM Score         │
│               + 0.4 × Similarity × Max  │
└─────────────────────────────────────────┘
        │
        ▼
Results + Feedback → Dashboard / API Response
```

---

## Components

### Image Preprocessing — `backend/preprocessor.py`

Prepares raw scanned images for accurate OCR:

1. Grayscale conversion
2. Noise removal (`fastNlMeansDenoising`)
3. Deskew correction (Hough line transform)
4. CLAHE contrast enhancement
5. Otsu binarization
6. Line segmentation for multi-line answers

### OCR Engine — `backend/ocr_module.py`

| Engine | Use Case |
|---|---|
| **TrOCR** (Microsoft) | Primary — transformer-based handwriting recognition |
| **Tesseract** | Fallback — rule-based OCR |

Supports JPEG, PNG, and multi-page PDF inputs. Confidence scores returned for every extraction.

### Text Processing — `backend/text_processor.py`

- Spell correction (pyspellchecker with academic vocabulary)
- Sentence segmentation via spaCy
- Stopword removal and lemmatization
- Normalization of OCR noise and punctuation artifacts

### Semantic Similarity — `backend/similarity.py`

Uses **Sentence-BERT** (`all-MiniLM-L6-v2`) to compute cosine similarity between student and teacher answers. Features:
- Overall answer similarity (0.0 – 1.0 scale)
- Sentence-level analysis with best-match highlighting
- Fine-tuning support on domain-specific QA pairs

### LLM Evaluation — `backend/llm_evaluator.py`

Routes evaluation to the correct prompt based on question type:

| Question Type | Evaluation Method |
|---|---|
| `open_ended` | Full STANDARD / CS / RUBRIC / STRICT prompt |
| `short_answer` | Concise factual accuracy check |
| `fill_blank` | Exact / near-exact match with OCR tolerance |
| `numerical` | Method + answer with configurable tolerance |
| `diagram` | OCR label extraction + description matching |
| `mcq` | Deterministic (no LLM call needed) |
| `true_false` | Deterministic (no LLM call needed) |

Output is structured JSON: `score`, `confidence`, `strengths`, `missing_concepts`, `feedback`.

### Rubric Matching — `backend/rubric_matcher.py`

Uses **DeBERTa NLI** zero-shot classification to check whether each rubric criterion is addressed in the student's answer — no training required. Works for any subject.

Example rubric criteria:
- "Definition of machine learning" (2 marks)
- "Supervised vs unsupervised example" (1.5 marks)
- "Real-world application" (1 mark)

### Question Classifier — `backend/question_classifier.py`

Auto-detects question type using LLM + regex fallback. Supported types:

`mcq` · `true_false` · `fill_blank` · `short_answer` · `open_ended` · `numerical` · `diagram`

Pass `question_type="auto"` to the API to enable this.

### Metrics — `backend/metrics.py`

Validates AI scoring accuracy against teacher ground truth:

| Metric | Applies To |
|---|---|
| MAE (Mean Absolute Error) | Open-ended |
| Pearson Correlation | Open-ended |
| Cohen's Kappa (linear weighted) | Open-ended |
| Accuracy within ±1 mark | Open-ended |
| Accuracy within ±0.5 marks | Open-ended |
| MCQ Accuracy % | MCQ |

Metrics are **recomputed in the background** after every `/evaluate` call and exposed at `GET /metrics`.

---

## API Reference

Built with **FastAPI**. Interactive docs available at `http://localhost:8000/docs`.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | API info and available endpoints |
| `GET` | `/health` | Service health check |
| `POST` | `/upload` | Upload student answer image or PDF |
| `POST` | `/ocr/{id}` | Run OCR only on a submission |
| `POST` | `/evaluate` | Full AI evaluation pipeline |
| `GET` | `/result/{id}` | Fetch evaluation result |
| `POST` | `/rubric` | Upload rubric criteria for a question |
| `GET` | `/stats` | System-wide statistics |
| `GET` | `/metrics` | AI scoring accuracy metrics (background-updated) |
| `GET` | `/metrics/compute` | Ad-hoc metric computation from score lists |
| `GET` | `/submissions` | List all submissions with results |

### Example: Evaluate a Submission

```bash
# Step 1 — Upload image
curl -X POST http://localhost:8000/upload \
  -F "file=@answer.jpg" \
  -F "student_code=STU001"

# Step 2 — Evaluate (returns JSON result instantly)
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "submission_id": 1,
    "question": "Explain gradient descent in machine learning.",
    "question_type": "open_ended",
    "teacher_answer": "Gradient descent is an optimization algorithm...",
    "max_marks": 10,
    "rubric_criteria": [
      {"criterion": "definition of gradient descent", "marks": 3},
      {"criterion": "role of learning rate", "marks": 3},
      {"criterion": "convergence explanation", "marks": 4}
    ]
  }'
```

### Example: MCQ Evaluation

```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "submission_id": 2,
    "question": "Which algorithm uses backpropagation?",
    "question_type": "mcq",
    "correct_option": "B",
    "max_marks": 1,
    "mcq_options": {
      "A": "K-Means",
      "B": "Neural Network",
      "C": "Decision Tree",
      "D": "Naive Bayes"
    }
  }'
```

### Example: Ad-hoc Metrics

```bash
# Compute metrics from score lists directly (no DB needed)
curl "http://localhost:8000/metrics/compute?ai_scores=7.5,8,6,9&teacher_scores=8,7.5,6.5,9&max_marks=10"
```

---

## Installation

### Prerequisites

- Python 3.9+
- Node.js 16+ (for docx generation, optional)
- Tesseract OCR (`apt install tesseract-ocr` on Linux)
- A Gemini API key from [https://aistudio.google.com](https://aistudio.google.com)

### Clone and Install

```bash
git clone https://github.com/your-org/IntelliGrade-H.git
cd IntelliGrade-H

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# Install Python dependencies
pip install -r requirements.txt

# Download spaCy language model
python -m spacy download en_core_web_sm
```

---

## Configuration

Create a `.env` file in the project root:

```env
# Required
GEMINI_API_KEY=your_gemini_api_key_here

# Database (SQLite for dev, PostgreSQL for prod)
DATABASE_URL=sqlite:///./intelligrade.db
# DATABASE_URL=postgresql://user:password@localhost:5432/intelligrade

# OCR Engine: "trocr" | "tesseract" | "ensemble"
OCR_ENGINE=trocr

# Scoring weights (must sum to 1.0)
LLM_WEIGHT=0.6
SIMILARITY_WEIGHT=0.4

# File upload settings
UPLOAD_DIR=./uploads
MAX_FILE_SIZE_MB=20
```

Get your free Gemini API key at: [https://aistudio.google.com](https://aistudio.google.com)

---

## Running the System

### Backend API

```bash
uvicorn backend.api:app --reload

# API running at:  http://localhost:8000
# Swagger UI at:   http://localhost:8000/docs
# ReDoc at:        http://localhost:8000/redoc
```

### Teacher Dashboard

```bash
streamlit run frontend/dashboard.py

# Dashboard running at: http://localhost:8501
```

### Both together (recommended)

```bash
# Terminal 1
uvicorn backend.api:app --reload

# Terminal 2
streamlit run frontend/dashboard.py
```

---

## Evaluation Metrics

The system continuously validates AI grading accuracy against teacher ground truth. Access results at `GET /metrics`.

| Metric | Target | Description |
|---|---|---|
| Mean Absolute Error | < 1.0 mark | Average scoring deviation |
| Pearson Correlation | > 0.80 | Linear correlation with teacher grades |
| Cohen's Kappa | > 0.70 | Inter-rater agreement (chance-corrected) |
| Accuracy ±1 mark | > 85% | % of grades within 1 mark of teacher |
| Accuracy ±0.5 marks | > 70% | % of grades within 0.5 marks of teacher |
| MCQ Accuracy | > 95% | Exact match for multiple-choice |

> **Note:** Open-ended MAE and Kappa metrics require teacher ground-truth scores stored in the database. For best results, have teachers review and confirm AI grades — their scores are then used for ongoing calibration.

---

## Fine-Tuning TrOCR

### Step 1 — Collect Dataset

```bash
# Add labeled samples
python datasets/collect_dataset.py add --image sample.jpg --text "The answer is 42"

# Generate synthetic samples for bootstrapping
python datasets/collect_dataset.py synthetic --n 500

# Check dataset statistics
python datasets/collect_dataset.py review

# Export train/val split (80/20)
python datasets/collect_dataset.py export
```

### Step 2 — Train

```bash
python models/train_trocr.py train \
  --dataset datasets/training \
  --output models/trocr-finetuned \
  --epochs 5 \
  --batch-size 8
```

### Step 3 — Evaluate

```bash
python models/train_trocr.py eval \
  --model models/trocr-finetuned \
  --test-dir datasets/training/val
```

Metric used: **Character Error Rate (CER)**. A CER below 0.05 (5%) is considered production-ready.

### Step 4 — Use Fine-Tuned Model

Update `.env`:

```env
OCR_ENGINE=trocr
TROCR_MODEL_PATH=./models/trocr-finetuned
```

---

## Docker Deployment

```bash
# Build and start all services
docker-compose up --build

# Services started:
#   API Server   →  http://localhost:8000
#   Dashboard    →  http://localhost:8501
#   Database     →  PostgreSQL (internal)
```

For production, update `docker-compose.yml` to set `GEMINI_API_KEY` and `DATABASE_URL` as environment secrets rather than in plain text.

---

## Running Tests

```bash
# Run full test suite
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=backend --cov-report=html

# Run specific test module
pytest tests/test_all.py::TestOCR -v
```

Tests cover: OCR pipeline · Text processing · Similarity scoring · LLM evaluation · API endpoints · Rubric matching · Metrics computation.

---

## Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| OCR | TrOCR (Microsoft), Tesseract | Handwriting recognition |
| Image Processing | OpenCV, Pillow | Preprocessing pipeline |
| NLP | spaCy, NLTK | Text processing |
| Embeddings | Sentence-BERT | Semantic similarity |
| LLM | Gemini 2.0 Flash | Answer evaluation |
| Rubric Matching | DeBERTa (cross-encoder NLI) | Zero-shot criterion detection |
| Question Classification | LLM + Regex | Auto question-type detection |
| Backend | FastAPI, SQLAlchemy | REST API + ORM |
| Frontend | Streamlit | Teacher dashboard |
| Database | SQLite (dev) / PostgreSQL (prod) | Data persistence |
| Deployment | Docker, docker-compose | Containerization |
| Testing | pytest | Test suite |

---

## Ethical Considerations

- **Student privacy:** Identities are anonymized using student codes — no personally identifiable information is stored.
- **Local processing:** All computation runs locally except LLM API calls to Gemini (which are stateless and not stored by the provider).
- **Human oversight:** AI grades are advisory. Results must be reviewed and confirmed by instructors before official use.
- **Fairness:** The system does not penalize OCR spelling errors — it evaluates conceptual understanding, not surface form.
- **Transparency:** Every score includes reasoning, strengths, and missing concepts so students understand their grade.

---

## Future Improvements

- Diagram and flowchart recognition (visual grading)
- Mathematical expression evaluation (LaTeX parsing)
- Multilingual support — Tamil, Hindi, and English
- Continuous learning from teacher corrections (active learning loop)
- Moodle / Google Classroom LMS integration
- Offline mode (local LLM via Ollama)
- Mobile app for scanning and submitting answers
- PDF batch upload with per-page question mapping

---

## Research Potential

IntelliGrade-H is designed to evolve into:

- An **academic publication** on automated subjective answer evaluation
- A **startup-grade grading platform** deployable across universities
- A **scalable assessment infrastructure** for national-level examinations
