# IntelliGrade-H

<p align="center">
  <b>AI-Powered Automatic Evaluation System for Handwritten Subjective Exam Answers</b><br/>
  Developed at Sathyabama Institute of Science and Technology
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue"/>
  <img src="https://img.shields.io/badge/FastAPI-Backend-green"/>
  <img src="https://img.shields.io/badge/PyTorch-DeepLearning-red"/>
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-orange"/>
  <img src="https://img.shields.io/badge/TrOCR--Large-Handwriting_OCR-blueviolet"/>
  <img src="https://img.shields.io/badge/Groq-LLaMA_3.3_70B-purple"/>
  <img src="https://img.shields.io/badge/Claude-Haiku_Fallback-lightblue"/>
  <img src="https://img.shields.io/badge/Sentence--BERT-NLP-yellow"/>
  <img src="https://img.shields.io/badge/License-MIT-lightgrey"/>
</p>

<p align="center">
  A production-grade AI grading system that evaluates handwritten exam booklets using<br/>
  <b>Computer Vision · OCR · NLP · LLM Reasoning</b>
</p>

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [How It Works — End-to-End Overview](#how-it-works--end-to-end-overview)
3. [Key Features](#key-features)
4. [System Architecture](#️-system-architecture)
5. [Image Preprocessing](#image-preprocessing)
6. [OCR Pipeline](#ocr-pipeline)
7. [Text Processing](#text-processing)
8. [Exam Parsers](#exam-parsers)
9. [Evaluation Engine](#evaluation-engine)
10. [Hybrid Scoring Formula](#hybrid-scoring-formula)
11. [LLM Integration](#llm-integration)
12. [Dashboard Features](#dashboard-features)
13. [Evaluation Metrics & Targets](#evaluation-metrics--targets)
14. [Technology Stack](#technology-stack)
15. [Project Structure](#project-structure)
16. [Quick Start](#quick-start)
17. [Installation](#installation)
18. [Environment Configuration](#environment-configuration)
19. [Running the System](#running-the-system)
20. [Fine-Tuning TrOCR on Your Exam Data](#-fine-tuning-trocr-on-your-exam-data)
21. [API Reference](#api-reference)
22. [Ethical Considerations](#ethical-considerations)
23. [Future Work](#future-work)
24. [License](#license)

---

## Problem Statement

Universities and colleges that conduct handwritten subjective examinations face a serious operational bottleneck at every evaluation cycle:

- **Time-consuming** — A single faculty member may need days to evaluate hundreds of answer booklets, delaying result publication.
- **Inconsistent** — Scores for the same answer can vary significantly depending on the evaluator, their mood, or fatigue. Research in educational assessment shows inter-rater disagreement of ±1–2 marks is common even among experienced faculty.
- **Unscalable** — As cohort sizes grow, the effort required grows linearly. There is no economies-of-scale benefit in manual grading.
- **Opaque** — Students rarely receive structured feedback explaining *why* they lost marks — only a number.

**IntelliGrade-H automates this entire pipeline end-to-end.** A teacher uploads scanned booklets and answer keys; the system returns marks, detailed feedback, and analytics — all within minutes. Every AI-generated grade is teacher-reviewable and editable before finalisation, keeping the human in control.

---

## How It Works — End-to-End Overview

Here is the complete journey of a single student booklet through the system:

**1. Upload** — Teacher uploads a scanned PDF or image of the student's handwritten booklet via the Streamlit dashboard or REST API.

**2. Preprocessing** — OpenCV cleans the scanned image: corrects page skew (deskew), removes noise, enhances contrast using CLAHE (Contrast Limited Adaptive Histogram Equalisation), and applies adaptive binarisation to make handwriting sharp and machine-readable.

**3. OCR** — The cleaned image passes through a 6-engine hybrid cascade. Cloud APIs (Google Vision, Mistral, Azure) are tried first for maximum accuracy. If unavailable or low-confidence, local engines (PaddleOCR, Tesseract, TrOCR-Large) take over. The best result is selected and returned.

**4. Text Processing** — The raw OCR output is cleaned with spaCy: tokenisation, stopword handling, spell correction (targeting domain-specific vocabulary like subject keywords), and text normalisation.

**5. Parsing** — The system identifies which text belongs to which question using the Student Answer Parser. This uses LLM-assisted segmentation to handle numbered answers, question parts (a, b, c), and even out-of-order answers.

**6. Evaluation** — Each answer is evaluated against the model answer from the Answer Key using five parallel methods: LLM reasoning, Sentence-BERT semantic similarity, zero-shot NLI rubric matching, keyword coverage, and length normalisation.

**7. Scoring** — A weighted hybrid score is computed and bounded to the maximum marks for that question.

**8. Feedback** — The LLM generates structured feedback: strengths identified, missing concepts, improvement suggestions, and a sentence-level similarity breakdown.

**9. Review** — The teacher sees all scores and feedback on the dashboard. They can adjust any mark before finalising.

**10. Export** — Final marks and feedback are exportable as CSV for uploading to the institution's LMS or mark register.

---

## Key Features

| Feature | Description |
|---|---|
| 6-Engine Hybrid OCR | Google Vision → Mistral → Azure → PaddleOCR → Tesseract → TrOCR cascade with automatic best-result selection |
| Fine-Tunable TrOCR | `trocr-large-handwritten` is auto-replaced by your domain-trained model when placed in `models/trocr-finetuned/` |
| Dual LLM Support | Groq LLaMA 3.3-70B (primary, fast, free) + Anthropic Claude Haiku (automatic fallback when API key is set) |
| Semantic Similarity | Sentence-BERT cosine similarity with sentence-level breakdown showing exactly which parts of the answer matched |
| Rubric Matching | Zero-shot NLI rubric coverage via `cross-encoder/nli-deberta-v3-small` — no rubric training data required |
| Answer Key Parser | Auto-extracts model answers from teacher PDF (both typed and scanned), supports Set-A / Set-B variants |
| Question Paper Parser | Detects question parts, marks, OR alternatives, and 7 question types automatically from PDF |
| 7 Question Types | Auto-classifies: MCQ, Short Answer, Long Answer, Descriptive, Code, Diagram, Mathematical |
| Bulk Grading | Evaluate an entire class in one upload with full CSV export of marks and feedback |
| Analytics Dashboard | MAE, Pearson r, Cohen's Kappa, accuracy within ±1 and ±0.5 marks, score distribution charts |
| Diagram Detection | YOLOv8n detects whether student drew required diagrams — awarded bonus consideration |
| Cover Page Extraction | LLM extracts roll number, course code, semester, set from booklet cover page automatically |
| Docker Ready | One-command deployment with PostgreSQL support for production |
| Teacher Override | Every AI grade is editable — IntelliGrade-H is a grading assistant, not a replacement |

---

## System Architecture

<p align="center">
  <img src="assets/intelligrade-architecture.png" width="900">
</p>

```
Handwritten Booklet (PDF / Image)
           │
           ▼
   Image Preprocessing
   (OpenCV — deskew, denoise, CLAHE, smart threshold)
           │
           ▼
      Hybrid OCR Pipeline (6 engines)
  Google Vision → Mistral OCR → Azure AI Vision
    → PaddleOCR → Tesseract → TrOCR-Large
           │
           ▼
     Text Processing
   (spaCy + spell correction + normalisation)
           │
           ▼
      Exam Parsers
  ┌────────────────────┐
  │ Question Paper     │
  │ Answer Key         │
  │ Student Booklet    │
  └────────────────────┘
           │
           ▼
    Evaluation Engine
  ┌────────────────────┐
  │ LLM Evaluator      │  ← Groq / Claude
  │ Sentence-BERT      │
  │ Rubric Matcher     │
  │ Keyword Coverage   │
  │ Diagram Detector   │  ← YOLOv8
  └────────────────────┘
           │
           ▼
   Hybrid Scoring Engine
           │
           ▼
   Teacher Dashboard (Streamlit)
```

---

## Image Preprocessing

Raw scanned booklets are rarely clean. Students scan pages with their phones, pages are skewed, lighting is uneven, and ink strokes vary in thickness. The `preprocessor.py` module applies a multi-stage pipeline before any OCR is attempted:

**Deskew** — Hough transform detects the dominant angle of text lines and rotates the page to correct misalignment. Skewed text dramatically reduces OCR accuracy.

**Denoising** — Gaussian and median filters remove scanner artifacts and paper grain without blurring ink strokes.

**CLAHE (Contrast Limited Adaptive Histogram Equalisation)** — Improves local contrast in regions where handwriting is faint or the background is uneven (common with pencil-written answers or low-quality scans).

**Adaptive Binarisation** — Converts to binary (black/white) using locally-computed thresholds rather than a global threshold. This handles pages where some regions are darker than others — for example, shadows near the spine of a booklet.

**Line Segmentation** — The `segment_lines()` method uses horizontal projection profiles to identify individual lines of text and crop them into separate images. This is required for TrOCR, which operates on single-line images, and also for the fine-tuning dataset preparation workflow.

---

## OCR Pipeline

The OCR system uses a **6-engine cascade** ordered by accuracy. The system tries each engine in sequence; cloud APIs return as soon as they produce a high-confidence result, avoiding unnecessary API calls. Local engines all run and the best result is selected.

```
1. Google Cloud Vision API   ← Best general handwriting accuracy (~3–8% CER)
2. Mistral OCR               ← Document-optimised, 1000 pages/month free tier
3. Azure AI Vision           ← 5000 pages/month free, no expiry on free quota
4. PaddleOCR                 ← Best local engine for mixed text/diagram layouts
5. Tesseract (PSM 11)        ← Solid layout-aware fallback, fully offline
6. TrOCR-Large               ← Fine-tunable handwriting transformer (HuggingFace)
```

**How the cascade works in practice:**

- If `GOOGLE_VISION_API_KEY` is set and returns a result with confidence above threshold, the pipeline stops there and returns immediately.
- If Google Vision is unavailable or confidence is low, Mistral OCR is tried next.
- If no cloud API key is configured, the system falls through directly to local engines.
- Local engines (PaddleOCR, Tesseract, TrOCR) all run in parallel (configurable via `OCR_WORKERS`), and their outputs are compared using a confidence scoring heuristic — the result with the best character-level confidence wins.

**Typed PDFs bypass OCR entirely.** When a teacher uploads a typed question paper or answer key PDF, the system detects that the PDF contains selectable text and extracts it directly via pdfplumber or PyMuPDF. This is instant and 100% accurate — no OCR is involved.

**Fine-tuned model auto-detection:** If `models/trocr-finetuned/config.json` exists on disk, TrOCR-Large is automatically replaced with your domain-trained model. The system logs `Fine-tuned TrOCR model found at models/trocr-finetuned — using it.` at startup. No configuration change needed.

**OCR confidence scoring** is computed per-engine using character-level probability outputs where available (Google Vision, TrOCR) or word-level confidence scores (Tesseract). For engines that provide no confidence signal (Mistral), a heuristic based on dictionary word hit rate is used.

---

## Text Processing

After OCR, raw extracted text goes through the `text_processor.py` pipeline before evaluation:

**Tokenisation** — spaCy `en_core_web_sm` tokenises the text into sentences and words with part-of-speech tagging. This enables sentence-level similarity analysis downstream.

**Spell Correction** — A domain-aware spell corrector (pyspellchecker with a custom vocabulary of CS/engineering terms) corrects common OCR misreads. For example: `"algoriThm"` → `"algorithm"`, `"datahase"` → `"database"`. Domain terms like `"DBMS"`, `"TCP/IP"`, `"SQL"` are whitelisted to avoid being "corrected" to common English words.

**Text Normalisation** — Converts to lowercase, removes excess whitespace, normalises punctuation, expands common abbreviations (e.g. `"w.r.t"` → `"with respect to"`), and strips page headers/footers that OCR may have picked up.

**Important:** OCR noise does **not** penalise students. All LLM evaluation prompts explicitly instruct the model to interpret unclear or garbled text charitably and focus on the conceptual content of the answer.

---

## Exam Parsers

IntelliGrade-H contains three specialised parsers that understand the structure of Indian university exam documents:

### Question Paper Parser (`question_paper_parser.py`)

Processes the teacher's question paper PDF and extracts a structured representation. Handles:

- **Multi-part questions** — e.g. "Q3. (a) Define normalisation. [5 marks] (b) Explain 3NF with example. [5 marks]"
- **OR alternatives** — e.g. "Q5. Either (a) ... OR (b) ..." — both alternatives are stored and the system evaluates whichever the student answered
- **Mark extraction** — Detects marks in parentheses, brackets, or inline text and associates them with each question/part
- **7 question types** auto-classified by the `question_classifier.py` module:

| Type | Detection Cues |
|---|---|
| MCQ | Option labels (A/B/C/D), "choose the correct", "which of the following" |
| Short Answer | Low mark value (1–3 marks), "define", "state", "list" |
| Long Answer | High mark value (8–16 marks), "explain in detail", "describe" |
| Descriptive | "discuss", "compare and contrast", "critically analyse" |
| Code | "write a program", "implement", language keywords detected |
| Diagram | "draw", "sketch", "illustrate with diagram" |
| Mathematical | "derive", "prove", "calculate", equation symbols detected |

### Answer Key Parser (`answer_key_parser.py`)

Processes the teacher's model answer PDF. Handles both typed PDFs (direct extraction) and scanned answer keys (OCR + structure extraction). Supports:

- Multiple sets (Set-A, Set-B) — answers are stored per-set and matched to the student's booklet set extracted from the cover page
- Partial model answers (bullet points, keywords, formulae) — the system uses these as evaluation anchors rather than requiring verbatim match
- Rubric attachments — if the teacher marks criteria inline (e.g. "1 mark for definition, 2 marks for example"), these are extracted and used by the Rubric Matcher

### Student Answer Parser (`student_answer_parser.py`)

This is the most complex parser. It segments a student's handwritten booklet into individual answers and maps each to the correct question. Challenges handled:

- **Out-of-order answers** — Students sometimes answer Q5 before Q3. The parser uses question number detection (written by the student) to map answers correctly.
- **Multi-page answers** — Answers that span page boundaries are detected and merged.
- **Answer continuation markers** — "Contd..." and similar are handled.
- **LLM-assisted segmentation** — For ambiguous cases where the question number is unclear, the LLM infers which question an answer segment belongs to based on its content.
- **Cover page extraction** — Roll number, name, course code, semester, set (A/B), and date are extracted from the cover page using a dedicated LLM prompt.

---

## Evaluation Engine

Each student answer is evaluated by five parallel components, whose scores are then combined by the Hybrid Scoring Engine.

### 1. LLM Evaluator (`llm_evaluator.py`)

The LLM receives the student answer, the model answer, the question text, the maximum marks, and the question type. It returns:

- A **numerical score** (bounded to max marks)
- **Strengths** — what the student correctly covered
- **Missing concepts** — what was absent or incorrect
- **Improvement suggestions** — specific guidance for the student
- **Score rationale** — explanation of why this score was awarded

The prompt template is selected automatically based on question type (see [LLM Integration](#llm-integration)). The LLM is explicitly told to ignore OCR noise in the student's answer.

### 2. Sentence-BERT Semantic Similarity (`similarity.py`)

`sentence-transformers/all-MiniLM-L6-v2` computes a cosine similarity between the student answer and model answer embeddings. This catches cases where the student used different but correct terminology (e.g. "heap data structure" vs "priority queue implementation") that keyword matching would miss.

Additionally, the module performs **sentence-level breakdown**: each sentence of the student answer is individually compared against each sentence of the model answer. This produces a per-sentence similarity matrix displayed in the dashboard, showing exactly which concepts the student covered and which were absent.

### 3. Rubric Matcher (`rubric_matcher.py`)

Uses `cross-encoder/nli-deberta-v3-small` for zero-shot Natural Language Inference. Each rubric criterion (e.g. "Student correctly defines primary key") is treated as a hypothesis, and the student answer is the premise. The NLI model determines whether the criterion is `entailed`, `neutral`, or `contradicted` — without any task-specific training data required.

This enables rubric-based marking even when the teacher has not manually created structured rubrics — the system infers rubric criteria from the model answer itself.

### 4. Keyword Coverage

Extracts domain keywords from the model answer using TF-IDF weighting and checks what fraction of them appear in the student answer. Keywords are matched after lemmatisation (e.g. "normalised" matches "normalisation") and synonyms are resolved using a domain-specific vocabulary. This component acts as a fast, interpretable sanity check alongside the more sophisticated semantic and LLM methods.

### 5. Length Normalisation

A mild penalty/bonus applied based on answer length relative to expected length for the question type and marks. Very short answers (under 20% of expected length) receive a small penalty. This prevents the LLM from being overly generous with one-line answers to 10-mark questions. The weight is deliberately low (5%) to avoid punishing concise but correct answers.

### 6. Diagram Detector (`diagram_detector.py`)

YOLOv8n runs on each page to detect drawn figures, flowcharts, circuit diagrams, and other visual content. For questions that require a diagram (detected by the Question Classifier), the presence of a diagram contributes to the student's score. This also flags cases where a student drew something but the OCR missed it entirely — the evaluator is notified to review manually.

---

## Hybrid Scoring Formula

```
Final Score =
  0.40 × LLM Evaluation Score        (Groq LLaMA 3.3-70B / Claude Haiku)
  0.25 × Semantic Similarity          (Sentence-BERT cosine similarity)
  0.20 × Rubric Coverage              (Zero-Shot NLI)
  0.10 × Keyword Coverage
  0.05 × Length Normalisation
```

The final score is **clamped to [0, max_marks]** for that question — it can never exceed the allocated marks or go below zero.

**Why these weights?** The LLM gets the highest weight (40%) because it provides the most holistic evaluation — it understands context, can assess reasoning quality, and handles domain-specific language. Semantic similarity (25%) catches correct ideas expressed differently. Rubric coverage (20%) provides structured, criterion-based fairness. Keyword coverage (10%) provides a fast, transparent check. Length normalisation (5%) is a guard against trivially short answers.

**Weights are fully configurable** via `.env` (see `LLM_WEIGHT`, `SIMILARITY_WEIGHT`, etc.). The system validates at startup that weights sum to 1.0 and raises a warning if they do not — preventing silent grade inflation or deflation.

---

## LLM Integration

### Primary: Groq — `llama-3.3-70b-versatile`

Groq provides extremely fast inference (typically under 2 seconds per evaluation) at no cost up to generous rate limits. It handles all answer evaluations, answer key extraction, booklet segmentation, cover page extraction, and MCQ disambiguation.

### Fallback: Anthropic Claude — `claude-haiku-4-5-20251001`

Claude Haiku activates automatically when:
- `ANTHROPIC_API_KEY` is set in `.env`, **and**
- Groq returns an error, rate limit, or timeout

No manual switching is required. The fallback is transparent to the teacher.

### Prompt Strategies

The system selects the appropriate prompt template automatically based on question type:

| Prompt | Used for | What it emphasises |
|---|---|---|
| `STANDARD_PROMPT` | General open-ended answers | Conceptual accuracy, completeness, clarity |
| `CS_ENGINEERING_PROMPT` | DBMS, algorithms, OS, Networks, code | Technical precision, correct terminology, algorithmic correctness |
| `RUBRIC_PROMPT` | Questions with explicit rubric criteria | Per-criterion mark allocation, structured breakdown |
| `STRICT_PROMPT` | Board-exam style marking | Exact keyword coverage, penalises vague answers |
| `MCQ_VALIDATION_PROMPT` | MCQ when OCR confidence < 0.5 | Disambiguates likely intended option from noisy OCR |

### LLM Output Format

All evaluation LLM calls return structured JSON, validated by Pydantic v2 schemas in `schemas.py`:

```json
{
  "score": 7.5,
  "max_marks": 10,
  "breakdown": {
    "llm_score": 7.5,
    "similarity_score": 0.74,
    "rubric_score": 0.60,
    "keyword_score": 0.55,
    "length_score": 0.90
  },
  "feedback": {
    "strengths": ["Correctly defined normalisation", "Gave relevant example of 2NF"],
    "missing_concepts": ["Did not mention BCNF", "No discussion of anomalies"],
    "suggestions": ["Revise BCNF definition", "Include update/delete anomaly examples"],
    "rationale": "Student demonstrated understanding of 1NF and 2NF but missed higher normal forms."
  }
}
```

---

## Dashboard Features

The teacher dashboard (`frontend/dashboard.py`) is built with Streamlit and provides a complete grading workflow in a browser interface requiring no technical knowledge to use.

### Paper Manager

Upload a question paper PDF → the system auto-extracts every question with its parts, marks, type, and OR alternatives. The teacher reviews the extracted structure and can correct any parsing errors before proceeding. Extracted papers are stored and reused across multiple evaluation sessions.

### Answer Key Manager

Upload the teacher's model answer PDF → auto-extracts model answers per question. Supports Set-A / Set-B variants. If the PDF is typed, extraction is instant. If scanned, OCR is applied. The teacher can edit extracted answers directly in the dashboard.

### Student Booklets

Upload a single scanned booklet (PDF or image):

1. Cover page metadata is extracted (roll number, name, course, set)
2. The booklet is OCR'd and segmented into individual answers
3. Each answer is evaluated against the corresponding model answer
4. A structured result view shows: score, max marks, strengths, missing concepts, suggestions, and a sentence-level similarity heatmap

### Bulk Upload

Upload an entire class's booklets at once. The system processes all booklets in parallel (configurable via `OCR_WORKERS`). Results are shown in a class-wide table. Export to CSV includes: roll number, question-wise marks, total marks, and feedback per question.

### Analytics

After bulk evaluation, the Analytics page shows:

- **MAE** (Mean Absolute Error) — average deviation of AI scores from teacher-corrected scores
- **Pearson Correlation** — linear agreement between AI and teacher scores
- **Cohen's Kappa** — inter-rater agreement corrected for chance
- **Accuracy within ±1 mark** and **±0.5 marks**
- **Score distribution chart** — histogram of AI scores vs teacher scores
- **Per-question breakdown** — which questions showed the highest AI-teacher disagreement

---

## Evaluation Metrics & Targets

These are the system's design targets for evaluation quality, validated against a held-out set of manually graded booklets:

| Metric | Target | What it measures |
|---|---|---|
| Mean Absolute Error (MAE) | < 0.8 marks | Average absolute difference between AI score and expert score |
| Pearson Correlation | > 0.85 | Linear agreement trend between AI and expert scores |
| Cohen's Kappa | > 0.75 | Agreement adjusted for chance — 0.75+ is "substantial agreement" |
| Accuracy within ±1 mark | > 90% | Fraction of answers scored within 1 mark of expert |
| Accuracy within ±0.5 marks | > 70% | Stricter threshold — near-exact agreement rate |

---

## Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| OCR (cloud) | Google Vision API, Mistral OCR, Azure AI Vision | High-accuracy cloud OCR with free tiers |
| OCR (local) | PaddleOCR, Tesseract (PSM 11), TrOCR-Large | Fully offline fallback engines |
| Image Processing | OpenCV, PIL | Deskew, CLAHE, adaptive threshold, line segmentation |
| Spell Correction | pyspellchecker + custom domain vocab | Post-OCR text cleaning |
| NLP | spaCy `en_core_web_sm` | Tokenisation, POS tagging, sentence splitting |
| Semantic Similarity | Sentence-BERT `all-MiniLM-L6-v2` | Cosine similarity + sentence-level breakdown |
| Rubric Matching | `cross-encoder/nli-deberta-v3-small` | Zero-shot NLI rubric coverage |
| Diagram Detection | YOLOv8n (Ultralytics) | Detects drawn diagrams and figures in scanned pages |
| LLM (primary) | Groq — `llama-3.3-70b-versatile` | Fast, free, high-quality answer evaluation |
| LLM (fallback) | Anthropic — `claude-haiku-4-5-20251001` | Automatic fallback when Groq is unavailable |
| Deep Learning | PyTorch, HuggingFace Transformers | TrOCR inference and fine-tuning |
| Backend | FastAPI, SQLAlchemy | REST API with 20+ endpoints, async support |
| Database | SQLite (dev) / PostgreSQL (prod) | Persistent storage of booklets, results, papers |
| Frontend | Streamlit | Teacher dashboard — no frontend code required |
| Containerisation | Docker, docker-compose | One-command deployment with service orchestration |

---

## Project Structure

```
IntelliGrade-H/
│
├── backend/
│   ├── api.py                    # FastAPI routes (20+ endpoints)
│   ├── evaluator.py              # Hybrid scoring engine — combines all sub-scores
│   ├── llm_provider.py           # Groq + Claude multi-provider client with fallback logic
│   ├── llm_evaluator.py          # LLM evaluation calls and prompt routing by question type
│   ├── evaluation_prompts.py     # All prompt templates (STANDARD, CS, RUBRIC, STRICT, MCQ)
│   ├── ocr_module.py             # 6-engine hybrid OCR cascade
│   ├── preprocessor.py           # Image preprocessing (deskew, CLAHE, binarise, segment)
│   ├── similarity.py             # Sentence-BERT cosine + sentence-level breakdown matrix
│   ├── rubric_matcher.py         # Zero-shot NLI rubric coverage (DeBERTa cross-encoder)
│   ├── question_classifier.py    # Auto question type detection (7 types)
│   ├── question_paper_parser.py  # Question paper PDF parser — extracts questions, marks, parts
│   ├── answer_key_parser.py      # Answer key PDF parser — supports Set-A/B, typed + scanned
│   ├── student_answer_parser.py  # Student booklet parser, segmenter, cover page extractor
│   ├── diagram_detector.py       # YOLOv8 + heuristic diagram detection
│   ├── text_processor.py         # NLP cleaning, spell correction, normalisation
│   ├── metrics.py                # MAE, Pearson r, Cohen's Kappa computation
│   ├── database.py               # SQLAlchemy models + auto-migration
│   ├── schemas.py                # Pydantic v2 request/response schemas (validates LLM output)
│   └── config.py                 # Environment configuration with startup validation
│
├── frontend/
│   └── dashboard.py              # Streamlit teacher dashboard (all pages)
│
├── models/
│   └── trocr-finetuned/          # Drop your fine-tuned model here — auto-detected at startup
│
├── uploads/                      # Uploaded PDFs stored here (auto-created on first run)
├── assets/                       # Images for README and dashboard
├── Dockerfile.backend            # FastAPI container
├── Dockerfile.frontend           # Streamlit container
├── docker-compose.yml            # Orchestrates backend + frontend + PostgreSQL
├── requirements.txt              # All Python dependencies pinned
├── IntelliGrade_TrOCR_Finetune.ipynb  # Google Colab fine-tuning notebook
├── run.py                        # Unified launcher (api / ui / both / init)
└── README.md
```

---

## Quick Start

If you want the system running in under 5 minutes with minimum configuration:

```bash
# 1. Clone the repo
git clone https://github.com/your-repo/IntelliGrade-H.git
cd IntelliGrade-H

# 2. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# 3. Get a free Groq API key at https://console.groq.com
#    Then create your .env:
echo "GROQ_API_KEY=gsk_your_key_here" > .env

# 4. Start
python run.py
```

Open `http://localhost:8501` — the teacher dashboard is ready.

The system works with **only a Groq API key**. All cloud OCR engines are optional — TrOCR and PaddleOCR handle OCR locally if no cloud keys are provided.

---

## Installation

### Local Installation (Full)

```bash
# 1. Clone
git clone https://github.com/your-repo/IntelliGrade-H.git
cd IntelliGrade-H

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Download spaCy and NLTK models
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# 4. Install Tesseract (system dependency)
# Linux:
sudo apt install tesseract-ocr poppler-utils
# macOS:
brew install tesseract poppler
# Windows:
# Download installer: https://github.com/UB-Mannheim/tesseract/wiki
# Then set TESSERACT_CMD in .env to the full path

# 5. Configure and run
cp .env.example .env
# Edit .env — add GROQ_API_KEY at minimum
python run.py
```

### Docker Installation (Recommended for Production)

Docker handles all system dependencies automatically, including Tesseract, PaddleOCR, and PostgreSQL.

```bash
# 1. Clone and configure
git clone https://github.com/your-repo/IntelliGrade-H.git
cd IntelliGrade-H
cp .env.example .env
# Edit .env — add GROQ_API_KEY

# 2. Build and start all services
docker compose up --build
```

Docker Compose starts three services: `backend` (FastAPI on port 8000), `frontend` (Streamlit on port 8501), and `db` (PostgreSQL on port 5432). Data is persisted in a named Docker volume so results survive container restarts.

### System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| Python | 3.10 | 3.11 |
| RAM | 4 GB | 8 GB+ |
| Disk | 5 GB (models) | 10 GB+ |
| GPU | Not required | CUDA GPU for faster TrOCR |
| OS | Linux / macOS / Windows | Ubuntu 22.04 LTS |

---

## Environment Configuration

Copy `.env.example` to `.env` and fill in the values. Only `GROQ_API_KEY` is required.

```env
# ── LLM (at least one key required) ──────────────────────────────────────────
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_...                       # Free at console.groq.com — required
GROQ_MODEL=llama-3.3-70b-versatile
ANTHROPIC_API_KEY=sk-ant-...               # Optional — Claude auto-activates as fallback
CLAUDE_MODEL=claude-haiku-4-5-20251001

# ── OCR Cloud APIs (each one improves accuracy — all optional) ────────────────
GOOGLE_VISION_API_KEY=                     # Best accuracy — free 1000 units/month
MISTRAL_API_KEY=                           # 1000 pages/month free
AZURE_VISION_KEY=                          # 5000 pages/month free, no expiry
AZURE_VISION_ENDPOINT=https://your-resource.cognitiveservices.azure.com

# ── OCR Local Engines ─────────────────────────────────────────────────────────
TESSERACT_CMD=tesseract                    # Full path on Windows
OCR_DPI=400                               # 300–400 DPI recommended for handwriting
OCR_WORKERS=2                             # Parallel workers for bulk processing
PADDLEOCR_LANG=en

# ── TrOCR (fine-tuned model auto-detected — no change needed after deploy) ────
TROCR_FINETUNED_PATH=models/trocr-finetuned
TROCR_MODEL_PATH=microsoft/trocr-large-handwritten

# ── Diagram Detection ─────────────────────────────────────────────────────────
YOLO_MODEL_PATH=yolov8n.pt
DIAGRAM_CONF_THRESHOLD=0.35

# ── Semantic Similarity ───────────────────────────────────────────────────────
SBERT_MODEL=sentence-transformers/all-MiniLM-L6-v2

# ── Hybrid Scoring Weights (must sum to 1.0 — validated at startup) ───────────
LLM_WEIGHT=0.40
SIMILARITY_WEIGHT=0.25
RUBRIC_WEIGHT=0.20
KEYWORD_WEIGHT=0.10
LENGTH_WEIGHT=0.05

# ── Database ──────────────────────────────────────────────────────────────────
DATABASE_URL=sqlite:///./intelligrade.db
# Production: DATABASE_URL=postgresql://user:pass@localhost:5432/intelligrade

# ── Upload / API ──────────────────────────────────────────────────────────────
MAX_FILE_SIZE_MB=20
UPLOAD_DIR=./uploads
API_HOST=0.0.0.0
API_PORT=8000
```

### Getting API Keys

| Key | Where to get it | Free tier |
|---|---|---|
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) | Generous free tier — required |
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com) | Paid, optional fallback |
| `GOOGLE_VISION_API_KEY` | [console.cloud.google.com](https://console.cloud.google.com) | 1000 units/month free |
| `MISTRAL_API_KEY` | [console.mistral.ai](https://console.mistral.ai) | 1000 pages/month free |
| `AZURE_VISION_KEY` | [portal.azure.com](https://portal.azure.com) | 5000 pages/month, no expiry |

---

## Running the System

```bash
python run.py           # Start both API server and Streamlit dashboard
python run.py api       # Start API server only (port 8000)
python run.py ui        # Start dashboard only (port 8501)
python run.py init      # Initialise/migrate database only
```

| Service | URL |
|---|---|
| Teacher Dashboard | http://localhost:8501 |
| REST API | http://localhost:8000 |
| API Documentation (Swagger) | http://localhost:8000/docs |
| API Documentation (Redoc) | http://localhost:8000/redoc |
| Metrics Debug | http://localhost:8000/metrics/print |

---

## 🔬 Fine-Tuning TrOCR on Your Exam Data

Fine-tuning on handwriting samples from your own students is the **single highest-impact improvement** you can make to OCR accuracy. A fine-tuned model learns your students' specific handwriting style, subject vocabulary, and common abbreviations. The system auto-detects and uses your model — no configuration change required after deployment.

### Why Fine-Tune? Model Comparison

| Model | CER on exam handwriting | VRAM needed |
|---|---|---|
| trocr-small, no fine-tuning | ~20–30% | ~2 GB |
| trocr-small, fine-tuned 1000 samples | ~15–20% | ~2 GB |
| trocr-large, no fine-tuning | ~15–22% | ~8 GB |
| **trocr-large, fine-tuned 500 samples** | **~10–15%** | **~8 GB** |
| **trocr-large, fine-tuned 1000+ samples** | **~6–11%** | **~8 GB** |
| Google Vision API (reference) | ~3–8% | Paid per page |

`trocr-large` has 4× more parameters than `trocr-small`. On variable, messy exam handwriting, this difference is decisive. The Colab free T4 GPU has 16 GB VRAM — `large` fits comfortably at batch size 8 with gradient checkpointing.

On domain-specific vocabulary (DBMS terms, algorithm names, circuit labels), a fine-tuned model can **match or exceed Google Vision** because it is trained on your students' exact writing style — whereas Google's model is general-purpose.

### Fine-Tuning Workflow

**Step 1 — Scan booklets** at 300–400 DPI (PNG). Anonymise student names before labelling.

**Step 2 — Crop into line images.** Each image must contain exactly one line of handwriting. Use `preprocessor.py`'s `segment_lines()` method for automation, or crop manually.

**Step 3 — Label the images.** Create a `labels.txt` file in each split folder (tab-separated):

```
0001.png	The mitochondria is the powerhouse of the cell
0002.png	Newton second law states F equals ma
```

Use [Label Studio](https://labelstud.io/) (free, runs locally) for a visual annotation interface. Two people can comfortably label 1000 samples in about one hour.

**Step 4 — Organise your dataset:**

```
My Drive/Intelligrade/datasets/handwriting/
├── train/    images/ + labels.txt    (~80% of samples)
├── val/      images/ + labels.txt    (~10% of samples)
└── test/     images/ + labels.txt    (~10% of samples)
```

**Step 5 — Run the Colab notebook:**

Open `IntelliGrade_TrOCR_Finetune.ipynb` → `Runtime → Change runtime type → T4 GPU` → Run all cells.

**Step 6 — Deploy your model:**

```
1. Download trocr-finetuned/ folder from Google Drive
2. Extract to: IntelliGrade-H/models/trocr-finetuned/
   (the folder must contain config.json — its presence triggers auto-detection)
3. Restart: python run.py

System logs at startup:
   Fine-tuned TrOCR model found at models/trocr-finetuned — using it.
```

No `.env` changes needed.

### Dataset Format

```
datasets/handwriting/
├── train/
│   ├── images/
│   │   ├── 0001.png
│   │   ├── 0002.png
│   │   └── ...
│   └── labels.txt
│
├── val/
│   ├── images/
│   │   ├── 1001.png
│   │   └── ...
│   └── labels.txt
│
└── test/
    ├── images/
    │   ├── 2001.png
    │   └── ...
    └── labels.txt
```

Each `labels.txt` line must follow:

```
image_filename<TAB>transcribed_text
```

**Important rules:**

- Use **tab (`\t`)** as separator — not spaces
- Images must be **PNG or JPG**
- Text must be **UTF-8 encoded**
- Each image must contain exactly **one line of handwriting**
- Filename must match exactly (case-sensitive on Linux)

### Training Configuration

| Parameter | Value | Notes |
|---|---|---|
| Base model | microsoft/trocr-large-handwritten | Better than small for exam handwriting |
| Epochs | 15 | Sufficient for 500–5000 samples |
| Batch Size | 8 | Fits in 16 GB VRAM with FP16 |
| Gradient Accumulation | 4 | Effective batch = 32 |
| Learning Rate | 5e-5 | Safe starting point for fine-tuning |
| Warmup Steps | 300 | Prevents early training instability |
| Mixed Precision | FP16 | Halves memory usage, same accuracy |
| Data Augmentation | Enabled | Random brightness, contrast, rotation ±5° |
| Evaluation Strategy | Per epoch | Saves best checkpoint automatically |

### Expected Training Time

| Dataset Size | Time on Colab T4 |
|---|---|
| 500 samples | ~20 minutes |
| 1000 samples | ~40 minutes |
| 5000 samples | ~2–3 hours |
| 10000 samples | ~4–5 hours |

GPU is strongly recommended. Training on CPU takes **10–20× longer**.

### Expected OCR Accuracy After Fine-Tuning

| Training Data | Character Error Rate (CER) |
|---|---|
| No fine-tuning (baseline) | ~15–22% |
| 500 samples | ~10–15% |
| 1000 samples | ~8–12% |
| 5000 samples | ~5–8% |

### Using the Fine-Tuned Model for Inference

```python
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

processor = TrOCRProcessor.from_pretrained("models/trocr-finetuned")
model = VisionEncoderDecoderModel.from_pretrained("models/trocr-finetuned")

image = Image.open("handwriting_line.png").convert("RGB")
pixel_values = processor(image, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values)

text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(text)
```

### Fine-Tuning Troubleshooting

**CUDA not detected**

```
AssertionError: Torch not compiled with CUDA enabled
```

Go to `Runtime → Change runtime type → GPU (T4)` in Colab and re-run the notebook from the start.

---

**Out of memory (OOM)**

Reduce batch size in the notebook:

```python
BATCH_SIZE = 4  # down from 8
```

Also ensure `gradient_checkpointing=True` is set in `TrainingArguments`.

---

**Training loss not decreasing**

Check for:
- Spaces instead of tabs in `labels.txt` — verify with `cat -A labels.txt | head -5` (tabs show as `^I`)
- Corrupted image files — test with `python -c "from PIL import Image; Image.open('0001.png')"`
- Labels over 100 characters — these destabilise training; split long lines

---

**Garbled output after deployment**

Ensure you copied the entire `trocr-finetuned/` folder including `preprocessor_config.json` and `tokenizer_config.json` — not just the model weights. These files are required for correct image preprocessing and text decoding.

---

## API Reference

The REST API is built with FastAPI and provides interactive documentation at `http://localhost:8000/docs`.

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/paper/upload` | Upload and parse a question paper PDF — returns structured questions, marks, types |
| `GET` | `/papers` | List all stored exam papers |
| `GET` | `/paper/{paper_id}` | Get full paper with all questions and their details |
| `POST` | `/answer-key/upload` | Upload and extract answer key from teacher PDF |
| `POST` | `/booklet/upload` | Upload a student booklet PDF — stores it, returns booklet ID |
| `POST` | `/booklet/{id}/evaluate` | Run OCR + full evaluation pipeline on a stored booklet |
| `POST` | `/evaluate` | Evaluate a single answer inline (no storage) |
| `POST` | `/evaluate/paper` | Evaluate a student answer against a question from a stored paper |
| `GET` | `/submissions` | List all evaluated student submissions with scores |
| `GET` | `/submission/{id}` | Get full evaluation result for a single submission |
| `PUT` | `/submission/{id}/score` | Teacher override — update a score after review |
| `GET` | `/stats` | System statistics (total booklets, questions, average scores) |
| `GET` | `/metrics` | AI accuracy metrics (MAE, Pearson r, Kappa) |
| `GET` | `/metrics/print` | Print metrics to server log (debug) |
| `POST` | `/rubric` | Upload rubric criteria for a specific question |
| `POST` | `/bulk/evaluate` | Batch evaluate multiple booklets — returns job ID |
| `GET` | `/bulk/{job_id}` | Poll bulk evaluation job status and retrieve results |
| `DELETE` | `/booklet/{id}` | Delete a booklet and all its evaluation results |

### Example: Single Answer Evaluation

```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Explain the concept of normalisation in DBMS.",
    "student_answer": "Normalisation is the process of organising data in a database to reduce redundancy. It involves dividing large tables into smaller ones and defining relationships.",
    "model_answer": "Normalisation is a technique to organise a relational database to reduce data redundancy and improve data integrity by decomposing tables into well-structured tables following normal forms (1NF, 2NF, 3NF, BCNF).",
    "max_marks": 5,
    "question_type": "short_answer"
  }'
```

### Example Response

```json
{
  "score": 3.5,
  "max_marks": 5,
  "breakdown": {
    "llm_score": 3.8,
    "similarity_score": 0.74,
    "rubric_score": 0.60,
    "keyword_score": 0.55,
    "length_score": 0.90
  },
  "feedback": {
    "strengths": [
      "Correctly defined normalisation as reducing redundancy",
      "Mentioned decomposition of tables"
    ],
    "missing_concepts": [
      "No mention of normal forms (1NF, 2NF, 3NF, BCNF)",
      "Data integrity not discussed"
    ],
    "suggestions": [
      "Study the specific normal forms and their conditions",
      "Include examples of anomalies that normalisation prevents"
    ],
    "rationale": "Student shows basic understanding but lacks knowledge of formal normal forms."
  }
}
```

---

## Ethical Considerations

IntelliGrade-H is designed as a **grading assistant** that augments teacher judgment — not a replacement for it. Several design decisions reflect this principle:

**Teacher control** — Every AI-generated grade is explicitly provisional until the teacher reviews and approves it. The dashboard makes it one click to adjust any mark. Bulk finalisation requires a deliberate teacher action.

**Student privacy** — Student names and personally identifiable information are used only for matching and are not included in evaluation API calls. No student identity data is sent to LLM providers — only the text of their answer.

**OCR noise does not penalise students** — All LLM evaluation prompts explicitly instruct the model to interpret unclear or garbled text charitably. A student who wrote a correct answer that OCR misread should not lose marks because of a scanning error.

**Transparent reasoning** — Every evaluation stores the score rationale, strengths, missing concepts, and sentence-level similarity breakdown. Students and teachers can see exactly why a score was awarded. This is exportable per submission.

**No bias amplification** — Evaluation prompts are designed to assess conceptual correctness only, not writing style, grammar, or language fluency — which could disadvantage non-native English speakers.

**Scoring weight integrity** — The system validates at startup that scoring weights sum to 1.0 and logs a warning if they do not, preventing silent grade inflation or deflation from misconfiguration.

---

## Future Work

- **Mathematical equation evaluation** — LaTeX-aware scoring for mathematics, physics, and engineering formula derivations
- **Diagram understanding** — Vision-language models to evaluate drawn circuit diagrams, flowcharts, and architectural diagrams against expected structures
- **Multilingual grading** — Support for Tamil, Hindi, and other regional languages in mixed-language answer booklets
- **LMS integrations** — Direct grade push to Moodle, Google Classroom, and other LMS platforms
- **Mobile scanning app** — iOS/Android app for scanning booklets with automatic deskew and upload quality check
- **Continual learning** — Feedback loop from teacher corrections to improve evaluation accuracy over time without full retraining
- **Plagiarism detection** — Cross-student similarity analysis to flag suspiciously similar answers in bulk evaluations
- **Offline mode** — Fully air-gapped deployment using a local LLM (Ollama + Mistral 7B) for institutions with strict data policies

---

## License

MIT License — free for academic and research use.

You are free to use, modify, and distribute this software for academic, research, and institutional purposes. Commercial use is permitted under the same MIT terms.