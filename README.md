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
9. [Question Classifier](#question-classifier)
10. [Evaluation Engine](#evaluation-engine)
11. [Hybrid Scoring Formula](#hybrid-scoring-formula)
12. [LLM Integration](#llm-integration)
13. [Dashboard Features](#dashboard-features)
14. [Evaluation Metrics & Targets](#evaluation-metrics--targets)
15. [Database Schema](#database-schema)
16. [Technology Stack](#technology-stack)
17. [Project Structure](#project-structure)
18. [Quick Start](#quick-start)
19. [Installation](#installation)
20. [Environment Configuration](#environment-configuration)
21. [Running the System](#running-the-system)
22. [Fine-Tuning TrOCR on Your Exam Data](#-fine-tuning-trocr-on-your-exam-data)
23. [API Reference](#api-reference)
24. [Ethical Considerations](#ethical-considerations)
25. [Future Work](#future-work)
26. [License](#license)

---

## Problem Statement

Universities and colleges that conduct handwritten subjective examinations face a serious operational bottleneck at every evaluation cycle:

- **Time-consuming** — A single faculty member may need days to evaluate hundreds of answer booklets, delaying result publication.
- **Inconsistent** — Scores for the same answer can vary significantly depending on the evaluator, their mood, or fatigue. Inter-rater disagreement of ±1–2 marks is common even among experienced faculty.
- **Unscalable** — As cohort sizes grow, the effort required grows linearly with no economies-of-scale benefit.
- **Opaque** — Students rarely receive structured feedback explaining *why* they lost marks — only a number.

**IntelliGrade-H automates this entire pipeline end-to-end.** A teacher uploads scanned booklets and answer keys; the system returns marks, detailed feedback, and analytics — all within minutes. Every AI-generated grade is teacher-reviewable and editable before finalisation.

---

## How It Works — End-to-End Overview

Here is the complete journey of a single student booklet through the system:

**1. Upload** — Teacher uploads a scanned PDF or image of the student's handwritten booklet via the Streamlit dashboard or `POST /booklet/upload`.

**2. Preprocessing** — OpenCV cleans the image: upscales if the page is narrower than 1200 px (phone photos), corrects skew, denoises adaptively, applies CLAHE contrast enhancement, and binarises using smart thresholding.

**3. OCR** — The cleaned image passes through a 6-engine hybrid cascade. Cloud APIs are tried first. Local engines compete and the winner is selected by `real_word_count × 0.7 + confidence × 0.3`. Typed PDFs bypass OCR entirely — text is extracted directly if pdfplumber finds more than 50 characters.

**4. Text Processing** — Raw OCR output is spell-corrected (domain-aware, CS/engineering vocabulary), normalised, and tokenised with spaCy.

**5. Parsing** — The Student Answer Parser segments the booklet into individual answers, maps each to its question number (handling out-of-order answers and multi-page continuations), and extracts cover page metadata via LLM.

**6. Classification** — Each question is classified into one of 7 types by the Question Classifier (LLM first, rule-based fallback).

**7. Evaluation** — Deterministic question types (MCQ, True/False, Numerical) are graded exactly. Open-ended types use the full 5-component hybrid pipeline: LLM + Sentence-BERT + Rubric NLI + Keyword Coverage + Length Normalisation.

**8. Scoring** — A weighted hybrid score is computed, clamped to [0, max_marks].

**9. Feedback** — The LLM returns strengths, missing concepts, improvement suggestions, a confidence score, and a score rationale.

**10. Review & Export** — Teachers review and optionally adjust scores on the dashboard. Final marks are exportable as CSV.

---

## Key Features

| Feature | Description |
|---|---|
| 6-Engine Hybrid OCR | Google Vision → Mistral → Azure → PaddleOCR → Tesseract → TrOCR cascade; winner chosen by real-word + confidence heuristic |
| Smart Typed-PDF Detection | If pdfplumber extracts >50 chars, OCR is skipped entirely — instant, 100% accurate for typed documents |
| Fine-Tunable TrOCR | `trocr-large-handwritten` auto-replaced by domain-trained model when `models/trocr-finetuned/config.json` exists |
| Dual LLM Support | Groq LLaMA 3.3-70B (primary) + Anthropic Claude Haiku (auto-fallback); configurable with `LLM_PROVIDER` |
| LLM Retry with Backoff | 3 attempts for Groq, 2 for Claude; exponential backoff on rate-limit errors |
| Semantic Similarity | Sentence-BERT cosine similarity + `compute_sentence_level()` per-sentence breakdown matrix |
| Rubric Matching | Zero-shot NLI via `cross-encoder/nli-deberta-v3-small`; optional BERT fine-tuning; skipped for MCQ/True-False/Numerical |
| 7 Question Types | MCQ, True/False, Fill-in-Blank, Short Answer, Open-Ended, Numerical, Diagram — classified automatically |
| Deterministic Grading | MCQ and True/False: binary exact match. Numerical: extracted number vs expected. No LLM needed. |
| MCQ LLM Fallback | When OCR confidence < 0.5, LLM reads the options and infers the selected answer |
| 5 Prompt Templates | STANDARD, CS_ENGINEERING, RUBRIC, STRICT, MCQ_VALIDATION — auto-selected by question type |
| Dynamic Exam Paper | Papers stored in DB with marks-per-question from PDF — no hardcoding needed |
| OR-Alternative Questions | Both alternatives stored; system evaluates whichever the student answered |
| Answer Key Parser | Supports Set-A / Set-B, typed + scanned, inline rubric extraction |
| Bulk Grading | `evaluate_batch()` processes a full class; `POST /booklet/{id}/evaluate` runs all answers in one call |
| Analytics | MAE, Pearson r, Cohen's Kappa (linear-weighted), accuracy within ±1 and ±0.5 marks |
| MetricsSnapshot | Persistent DB table caches metric history; recomputed in background after each batch |
| Image Upscaling | Phone-camera photos narrower than 1200 px are 2× upscaled before processing |
| Adaptive Denoising | `fastNlMeans` for noisy images (std > 20), `bilateral` filter for clean scans |
| Teacher Override | `PATCH /booklet/{id}/answer/{q_num}` and `PATCH /paper/{paper_id}/question/{q_num}` for manual corrections |
| Docker Ready | One-command deployment with PostgreSQL support |

---

## System Architecture

<p align="center">
  <img src="assets/intelligrade-architecture.png" width="900">
</p>

```
Handwritten Booklet (PDF / Image)
           │
           ▼
   Image Preprocessing (preprocessor.py)
   upscale(<1200px) → grayscale → adaptive-denoise
   → deskew (Hough) → CLAHE(3.0) → smart-threshold
           │
           ▼
      Hybrid OCR Pipeline (ocr_module.py)
  ┌── Typed PDF? pdfplumber/PyMuPDF ──────────────┐
  │   (>50 chars → return immediately)            │
  └── Scanned/Handwritten: ───────────────────────┘
      Cloud: Google Vision → Mistral → Azure
      Local: PaddleOCR + Tesseract + TrOCR-Large
      Winner: real_words×0.7 + confidence×0.3
           │
           ▼
     Text Processing (text_processor.py)
   spell-correct → normalise → tokenise (spaCy)
           │
           ▼
      Exam Parsers
  ┌────────────────────────┐
  │ Question Paper Parser  │ → ExamPaper / ExamQuestion (DB)
  │ Answer Key Parser      │ → teacher answers per question
  │ Student Booklet Parser │ → StudentBooklet / StudentAnswerText (DB)
  └────────────────────────┘
           │
           ▼
   Question Classifier (question_classifier.py)
   LLM → rule-based fallback → 7 types
           │
           ├── Deterministic (MCQ / True-False / Numerical)
           │       exact match / number extraction
           │
           └── Open-Ended (Short / Long / Diagram / Fill-Blank)
                   Evaluation Engine (evaluator.py)
                ┌────────────────────────┐
                │ LLM Evaluator          │ ← Groq / Claude
                │ Sentence-BERT          │ ← cosine + sentence-level
                │ Rubric Matcher (NLI)   │ ← DeBERTa cross-encoder
                │ Keyword Coverage       │
                │ Length Normalisation   │
                │ Diagram Detector       │ ← YOLOv8
                └────────────────────────┘
           │
           ▼
   Hybrid Scoring Engine
   (LLM×0.40 + Similarity×0.25 + Rubric×0.20
    + Keyword×0.10 + Length×0.05) clamped to max_marks
           │
           ▼
   Teacher Dashboard (Streamlit) / REST API (FastAPI)
```

---

## Image Preprocessing

`preprocessor.py` — `ImagePreprocessor` class applies a 6-stage pipeline before any OCR is attempted.

**Stage 1 — Upscale small images** (`_upscale_small`): Pages narrower than 1200 px (e.g. phone camera photos) are 2× upscaled using Lanczos4 interpolation. This dramatically improves OCR accuracy on low-resolution scans.

**Stage 2 — Greyscale** (`_to_grayscale`): Converts BGR/RGB to greyscale for all subsequent processing.

**Stage 3 — Adaptive Denoising** (`_denoise`): Chooses algorithm based on image noise level:
- If `np.std(img) > 20` (noisy, e.g. phone photo): `fastNlMeansDenoising(h=10)` — aggressive noise removal
- Otherwise (clean scanner): `bilateralFilter(d=5)` — preserves edges, ~3 ms

**Stage 4 — Deskew** (`_deskew`): Runs Hough line detection, computes the median angle of text lines, and rotates the image to correct skew. Skew correction is only applied if the angle exceeds 0.5°.

**Stage 5 — CLAHE** (`_enhance_contrast`): Contrast Limited Adaptive Histogram Equalisation with `clipLimit=3.0` and `tileGridSize=(8,8)`. Corrects uneven illumination — important for pencil-written answers and shadows near the booklet spine.

**Stage 6 — Smart Threshold** (`_threshold`): Otsu's threshold is computed first. If `otsu_t < 50` (faint or light ink), switches to adaptive Gaussian thresholding with `blockSize=31, C=15`. This handles mixed-darkness pages.

**Line Segmentation** (`segment_lines`): Uses horizontal projection profiles to find line boundaries. Each line is padded by 4 px top/bottom before being cropped into a PIL Image. This output feeds TrOCR (which expects single-line images) and is also used for fine-tuning dataset preparation.

---

## OCR Pipeline

`ocr_module.py` implements a **6-engine hybrid cascade** with two tiers.

### Typed PDF Fast-Path

If `pdfplumber` extracts more than 50 characters from a PDF, the system returns that text immediately without running any OCR. This gives 100% accuracy on typed question papers and answer keys.

### Scanned / Handwritten Pipeline

```
Priority  Engine              Notes
─────────────────────────────────────────────────────────────────────
1         Google Cloud Vision Best general accuracy (~3–8% CER)
2         Mistral OCR          Document-optimised; 1000 pages/month free
3         Azure AI Vision      5000 pages/month free, no expiry
4         PaddleOCR            Best local engine for mixed layouts
5         Tesseract (PSM 11)   Layout-aware; tries multiple configs, picks best
6         TrOCR-Large          Handwriting-trained transformer; line-by-line
```

**Cloud engine behaviour:** As soon as a cloud engine returns a result with more than 10 characters, it is returned immediately and the remaining engines are skipped.

**Local engine scoring:** All three local engines run (concurrently via `OCR_WORKERS`). The winner is selected by:

```
score = real_word_count × 0.7 + confidence × 0.3
```

`real_word_count` counts tokens containing at least one alphabetic character. This prevents an engine that returns a high-confidence but short/garbled output from winning over one with more real text.

**Tesseract multi-config:** Tries multiple PSM modes internally and selects the one with the highest `real_words + confidence` score.

**Fine-tuned TrOCR auto-detection:** At startup, `_resolve_trocr_model()` checks if `models/trocr-finetuned/config.json` exists. If so, that model is used instead of the default HuggingFace model. The system logs: `Fine-tuned TrOCR model found at 'models/trocr-finetuned' — using it.`

**Confidence values by engine:**
- Google Vision: page-level block confidence from API
- Tesseract: average word confidence from `--psm` output
- TrOCR: token-level generation probabilities
- Mistral: fixed at 0.88 (API does not return confidence); a real-word hit-rate heuristic is computed separately

---

## Text Processing

`text_processor.py` — `TextProcessor` class cleans OCR output before evaluation.

**Normalisation** (`_normalize`): Removes non-printable characters, converts line breaks to sentence separators (but avoids double-punctuation when the line already ends with `.!?`), fixes missing space after periods, and normalises Unicode quotes.

**Spell Correction** (`_spellcheck`): Uses `pyspellchecker` with a custom domain vocabulary pre-loaded on init including: algorithm, preprocessing, tokenization, backpropagation, gradient, sigmoid, relu, convolution, transformer, embedding, cosine, classifier, regression, hyperparameter, overfitting, underfitting, epoch, batch, lstm, attention, bert, scalability, bandwidth, latency, throughput, synchronous, asynchronous, microservices, polymorphism, encapsulation, inheritance.

Words are skipped (left uncorrected) if they: contain digits, are ALL CAPS (abbreviations), or are ≤ 2 characters. Original casing is preserved after correction.

**Sentence Segmentation** (`_segment_sentences`): Uses spaCy `en_core_web_sm` sentence splitter. Falls back to regex on `.!?` if spaCy is unavailable.

**Tokenisation** (`_tokenize`): spaCy lemmatisation with stopword and punctuation removal. Falls back to NLTK if spaCy is unavailable.

**Important:** OCR noise does **not** penalise students. All LLM evaluation prompts explicitly instruct the model to ignore OCR-introduced typos and evaluate conceptual correctness only.

---

## Exam Parsers

### Question Paper Parser (`question_paper_parser.py`)

Processes the teacher's question paper PDF and builds an `ExamPaper` structure stored in the database. Handles:

- **Multi-part questions** — e.g. "Q3. (a) Define normalisation. [5 marks] (b) Explain 3NF. [5 marks]"
- **OR alternatives** — e.g. "Q5. Either (a) ... OR (b) ..." — `is_or_option=True` is set on the alternative; the system evaluates whichever the student answered
- **Mark extraction** — Detects marks in parentheses `(5)`, square brackets `[5]`, or inline text `5 marks`. Always reads from PDF — never hardcoded.
- **Paper ID generation** — Builds a unique slug: `course_code + "_" + exam_name + "_" + set_name` (e.g. `S11BLH41_CAE1_Set-A`)

### Answer Key Parser (`answer_key_parser.py`)

Processes the teacher's model answer PDF. Supports:

- Typed PDFs (direct extraction via pdfplumber) and scanned answer keys (full OCR pipeline)
- Set-A / Set-B variants — answers stored per-set and matched to student's booklet set from cover page
- Inline rubric extraction — e.g. "1 mark for definition, 2 marks for example" is parsed into `Rubric` rows in the DB
- Partial model answers (bullet points, keywords, formulae) used as evaluation anchors

### Student Answer Parser (`student_answer_parser.py`)

The most complex parser — segments a handwritten booklet into per-question answers. Challenges handled:

- **Out-of-order answers** — Question number written by the student is detected; answers are mapped to the correct question regardless of order
- **Multi-page answers** — Page boundary detection and merge via continuation marker detection ("Contd...", "P.T.O.")
- **LLM-assisted segmentation** — When question number is ambiguous or missing, the LLM infers which question the answer segment belongs to based on content
- **Cover page extraction** — A dedicated LLM prompt extracts: roll number, student name, course code, course name, semester, exam set (A/B), date, programme, and batch

Each extracted answer is stored as a `StudentAnswerText` row linked to its `StudentBooklet`.

---

## Question Classifier

`question_classifier.py` — `QuestionClassifier` class. Two-stage classification:

**Stage 1 — LLM:** Sends the question text to the LLM with a structured JSON prompt. Returns type, confidence (0–1), and one-sentence reasoning.

**Stage 2 — Rule-based fallback** (used when LLM fails or is unavailable):

| Type | Key detection patterns |
|---|---|
| `mcq` | Lettered options `A) B) C) D)`, "which of the following", "choose the correct" |
| `true_false` | "true or false", "state whether", "T/F" |
| `fill_blank` | Underscores `____`, brackets `[...]`, "fill in", "complete the" |
| `numerical` | "calculate", "compute", "find", "solve" + result-type words ("value", "area", "force") |
| `diagram` | "draw", "sketch", "label", "illustrate", "flowchart" |
| `short_answer` | "define", "state", "list", "name", "what is" and question < 30 words |
| `open_ended` | "explain", "describe", "discuss", "analyze", "compare", "evaluate" |

Default when no pattern matches: `open_ended` with confidence 0.50.

**Routing impact:** The classifier determines the grading path:
- `mcq`, `true_false`, `fill_blank`, `numerical` → **Deterministic grading** (no LLM, no similarity)
- `open_ended`, `short_answer`, `diagram` → **Full LLM + similarity pipeline**

---

## Evaluation Engine

`evaluator.py` — `EvaluationEngine` class. Routes each question to the appropriate grading path.

### Deterministic Question Types

**MCQ (`_evaluate_mcq`):** Extracts the selected option letter (A–E) from OCR text using regex patterns (circled letter, standalone letter, underline). If OCR confidence < 0.5 and the full option texts are available, the `MCQ_VALIDATION_PROMPT` is sent to the LLM to infer the intended option. Score = `max_marks` if correct, 0 otherwise.

**True/False (`_evaluate_true_false`):** Extracts True/False/T/F from text using regex. Score = `max_marks` if correct, 0 otherwise.

**Numerical (`_evaluate_numerical_or_llm`):** Extracts a number from OCR text using regex. If it matches the expected answer within a small tolerance, full marks are awarded. If extraction fails, falls back to LLM evaluation.

### Open-Ended Questions — 5-Component Pipeline

**Component 1: LLM Evaluator (`llm_evaluator.py`)**

The LLM receives: question text, model answer, student answer (OCR-extracted), max marks, question type, and optionally rubric criteria. It returns structured JSON:

```json
{
  "score": 7.5,
  "confidence": 0.85,
  "strengths": ["..."],
  "missing_concepts": ["..."],
  "feedback": "...",
  "explanation": "..."
}
```

The `explanation` field is a one-sentence rationale for the exact score awarded — useful for teacher review and student transparency.

**Component 2: Sentence-BERT Semantic Similarity (`similarity.py`)**

`SemanticSimilarityModel` wraps `sentence-transformers/all-MiniLM-L6-v2`. Two methods:

- `compute_similarity(student, teacher)` → overall cosine similarity [0, 1]
- `compute_sentence_level(student, teacher)` → per-sentence matrix: each student sentence is matched to its most similar teacher sentence, returning `student_sentence`, `best_match_teacher`, and `similarity` score. Used for the heatmap in the dashboard.

Both inputs are normalised embeddings; the score is clamped to [0, 1].

The model can also be fine-tuned on institution-specific QA pairs via `fine_tune(training_data, output_dir)` using `CosineSimilarityLoss`.

**Component 3: Rubric Matcher (`rubric_matcher.py`)**

`RubricMatcher` uses `cross-encoder/nli-deberta-v3-small` for zero-shot NLI classification. For each rubric criterion, the student answer is the NLI premise and the criterion is the hypothesis. Criteria with entailment probability ≥ threshold (default 0.5) are marked as covered.

**Rubric matching is automatically skipped** for `mcq`, `true_false`, and `numerical` question types — it returns an empty `RubricResult` with zero scores for these.

Optional BERT fine-tuning: `train_finetuned(training_data, output_dir)` fine-tunes `bert-base-uncased` as a binary sequence classifier on labelled `(answer, criterion, present: 0/1)` pairs. This improves rubric detection precision for institutions that build a correction dataset over time.

**Component 4: Keyword Coverage**

TF-IDF-style keyword extraction from the model answer. Lemmatised tokens from the student answer are checked against the keyword set. The coverage ratio (0–1) reflects the fraction of key domain terms present in the student's response.

**Component 5: Length Normalisation**

Compares student answer word count to model answer word count. Returns a score ≤ 1.0. Very short answers receive a mild penalty. Deliberately low weight (5%) to avoid punishing concise but correct answers.

### Diagram Detection (`diagram_detector.py`)

YOLOv8n runs on each page to detect drawn figures, flowcharts, tables, and circuit diagrams. For questions classified as `diagram` type, the presence of a detected figure contributes to the evaluation. The detection confidence threshold is configurable via `DIAGRAM_CONF_THRESHOLD` (default 0.35).

### Batch Evaluation

`evaluate_batch(answers_list)` evaluates multiple answers in one call, returning a list of `EvaluationResult` objects. Used internally by `POST /booklet/{id}/evaluate` to process all answers in a student booklet in a single API request.

---

## Hybrid Scoring Formula

```
Final Score =
  LLM_WEIGHT        × llm_score           (default: 0.40)
  + SIMILARITY_WEIGHT × similarity_score  (default: 0.25)
  + RUBRIC_WEIGHT     × rubric_coverage   (default: 0.20)
  + KEYWORD_WEIGHT    × keyword_coverage  (default: 0.10)
  + LENGTH_WEIGHT     × length_norm       (default: 0.05)
```

The result is **clamped to [0, max_marks]**.

All five weights are configurable via `.env`. `config.py` validates at startup that they sum to 1.0 (tolerance ±0.01) and raises a `UserWarning` if they do not, explicitly stating whether scores will be inflated or deflated.

---

## LLM Integration

`llm_provider.py` — `LLMClient` class. A singleton accessed via `get_llm_client()`.

### Provider Selection

Set `LLM_PROVIDER` in `.env`:
- `LLM_PROVIDER=groq` (default) — Groq primary, Claude fallback
- `LLM_PROVIDER=claude` — Claude primary, Groq fallback

The `_ordered_providers()` method returns the list in the correct order. Offline heuristic is always the final fallback.

### Retry Logic

- **Groq:** 3 attempts. On rate-limit errors (HTTP 429) or "rate" in error message: exponential backoff `2^attempt` seconds. Other errors: linear backoff `attempt` seconds.
- **Claude:** 2 attempts. On overload errors (HTTP 529): exponential backoff.

### Token Cost Logging

After each Groq call, the logger records: `groq/llama-3.3-70b-versatile in 1234ms | 856 tokens | ~$0.000504`. Cost is estimated at `tokens × $0.00000059` (llama-3.3-70b approximate rate).

### API Methods

| Method | Description |
|---|---|
| `generate(prompt)` | Single-prompt call, returns `LLMResponse(text, provider, model, latency_ms)` |
| `generate_json(prompt)` | `generate()` + JSON parsing with fallback to safe zero-score dict |
| `complete(system, user)` | Structured system+user call, returns raw text string |

### Prompt Templates (`evaluation_prompts.py`)

| Template | Used when | Key characteristics |
|---|---|---|
| `STANDARD_PROMPT` | General open-ended | 6-level scoring ladder, OCR tolerance, `explanation` field |
| `CS_ENGINEERING_PROMPT` | DBMS, algorithms, OS, networks, ML | Technical criteria checklist, code OCR tolerance, `{rubric_section}` slot |
| `RUBRIC_PROMPT` | Questions with explicit rubric | `rubric_breakdown` per criterion with `awarded/max/reason` |
| `STRICT_PROMPT` | Board-exam marking | 5-level scheme, explicitly penalises vague/padded answers |
| `MCQ_VALIDATION_PROMPT` | MCQ, OCR confidence < 0.5 | Returns `detected_option` (A/B/C/D or null) + `confidence` + `reasoning` |

All prompts include the hard constraint: `"Your awarded score MUST be between 0 and {max_marks} inclusive"` and explicit OCR tolerance instructions.

---

## Dashboard Features

The teacher dashboard (`frontend/dashboard.py`) is built with Streamlit.

### Paper Manager

Upload a question paper PDF → auto-extract every question with parts, marks, type, and OR alternatives → teacher reviews and corrects any parsing errors → paper stored in DB as `ExamPaper` + `ExamQuestion` rows, reused across sessions.

### Answer Key Manager

Upload teacher's model answer PDF → auto-extract model answers per question → supports Set-A / Set-B → teacher can edit extracted answers directly → stored in DB as `teacher_answer` on each `ExamQuestion`.

### Student Booklets

Upload a single scanned booklet:
1. Cover page metadata extracted (roll number, name, course code, set, semester)
2. Booklet OCR'd and segmented → stored as `StudentBooklet` + `StudentAnswerText` rows
3. All answers evaluated: `POST /booklet/{id}/evaluate` runs `evaluate_batch()` internally
4. Result view: score, max marks, strengths, missing concepts, feedback, sentence-level similarity heatmap, `explanation` field, OCR confidence

### Bulk Upload

Upload full class booklets. Processed in parallel via `OCR_WORKERS`. Results in class-wide table. CSV export includes: roll number, question-wise marks, total, and per-question feedback.

### Analytics

After bulk evaluation:
- MAE, Pearson r, Cohen's Kappa (linear-weighted), accuracy within ±1 and ±0.5 marks
- Score distribution histogram (AI vs teacher)
- Per-question breakdown showing highest disagreement questions
- `MetricsSnapshot` persists metrics to DB so history is retained across sessions

---

## Evaluation Metrics & Targets

`metrics.py` — `compute_metrics()` and `compute_mcq_metrics()`.

The module supports three modes, auto-routed by `question_type`:

**Open-ended / Mixed:**

| Metric | Target | Description |
|---|---|---|
| MAE | < 0.8 marks | Mean absolute error vs expert |
| Pearson r | > 0.85 | Linear agreement |
| Cohen's Kappa (linear-weighted) | > 0.75 | Agreement corrected for chance |
| Accuracy within ±1 mark | > 90% | |
| Accuracy within ±0.5 marks | > 70% | Near-exact agreement |

**MCQ (via `compute_mcq_metrics`):** Takes raw option letters (`predicted_options`, `correct_options`) and returns accuracy, `mcq_n_correct`, `mcq_n_wrong`. Also available as a binary-score path in `compute_metrics()`.

`MetricsSnapshot` is a database table that caches the latest metrics and is recomputed in a background thread after each bulk evaluation, so the `/metrics` endpoint always returns fast without recomputing from scratch.

---

## Database Schema

`database.py` — SQLAlchemy models. SQLite (dev) or PostgreSQL (prod) via `DATABASE_URL`.

### Core Tables

**`ExamPaper`** — One row per uploaded question paper PDF.
- `paper_id`: unique slug (e.g. `S11BLH41_CAE1_Set-A`)
- Metadata: `course_code`, `course_name`, `exam_name`, `total_marks`, `duration_hours`, `exam_date`, `batch`, `programme`, `semester`, `set_name`
- Relationships: → `ExamPart[]`, → `ExamQuestion[]`, → `Submission[]`

**`ExamPart`** — One row per part within a paper (e.g. Part-A, Part-B).
- `marks_per_question`, `num_questions`, `instructions`

**`ExamQuestion`** — One row per question extracted from the paper.
- `question_number`, `question_text`, `marks` (always from PDF), `question_type`, `is_or_option`, `teacher_answer`, `correct_option` (MCQ)
- Relationships: → `Rubric[]`

**`Student`** — One row per student.
- `student_code` (unique), `name`

**`StudentBooklet`** — One row per uploaded student booklet PDF.
- Links to `Student` and `ExamPaper`
- Stores: `file_path`, `roll_number`, `student_name`, `exam_set`, `course_code`, `semester`
- Relationships: → `StudentAnswerText[]`

**`StudentAnswerText`** — One row per (booklet, question_number) pair.
- Stores the segmented OCR text for each answer
- Links to `ExamQuestion` for evaluation

**`Submission`** — One evaluation request (legacy single-question path).
- Links to `Student`, `Question` or `ExamPaper` + `ExamQuestion`

**`Result`** — One row per evaluated submission.
- Stores all five component scores, final score, max marks, feedback JSON (strengths, missing_concepts), OCR confidence, question type, evaluation time

**`Rubric`** — One row per rubric criterion.
- Can be linked to legacy `Question` or new `ExamQuestion`
- `element` (criterion text), `max_marks`

**`MetricsSnapshot`** — Cached metrics. Two rows max (open-ended, MCQ). Updated in background after bulk evaluations. `upsert()` creates or replaces on type key.

**Auto-migration:** `_migrate_db()` runs at startup and adds any missing columns to existing tables (e.g. adding `student_booklet` table columns on upgrade from older versions). No manual schema migration needed.

---

## Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| OCR (cloud) | Google Vision API, Mistral OCR, Azure AI Vision | High-accuracy cloud OCR with free tiers |
| OCR (local) | PaddleOCR, Tesseract (PSM 11), TrOCR-Large | Fully offline fallback engines |
| Image Processing | OpenCV, PIL | Upscale, deskew, CLAHE(3.0), adaptive threshold, line segmentation |
| Spell Correction | pyspellchecker + CS/engineering domain vocab | Post-OCR text cleaning |
| NLP | spaCy `en_core_web_sm` + NLTK fallback | Tokenisation, POS tagging, sentence splitting |
| Semantic Similarity | Sentence-BERT `all-MiniLM-L6-v2` | Cosine similarity + sentence-level breakdown |
| Rubric Matching | `cross-encoder/nli-deberta-v3-small` | Zero-shot NLI; optional BERT fine-tuning |
| Diagram Detection | YOLOv8n (Ultralytics) | Detects drawn diagrams in scanned pages |
| LLM (primary) | Groq — `llama-3.3-70b-versatile` | Fast, free inference with retry/backoff |
| LLM (fallback) | Anthropic — `claude-haiku-4-5-20251001` | Auto-activates when Groq fails |
| Deep Learning | PyTorch, HuggingFace Transformers | TrOCR inference and fine-tuning |
| Schemas | Pydantic v2 | Request/response validation; validates all LLM JSON output |
| Backend | FastAPI, SQLAlchemy | 25+ REST endpoints, async, auto-migration |
| Database | SQLite (dev) / PostgreSQL (prod) | 9 tables; MetricsSnapshot for cached analytics |
| Frontend | Streamlit | Teacher dashboard |
| Containerisation | Docker, docker-compose | One-command deployment |

---

## Project Structure

```
IntelliGrade-H/
│
├── backend/
│   ├── api.py                    # FastAPI app — 25+ routes, lifespan OCR warm-up
│   ├── evaluator.py              # EvaluationEngine — routes by question type, hybrid score
│   ├── llm_provider.py           # LLMClient — Groq + Claude, retry/backoff, cost logging
│   ├── llm_evaluator.py          # LLM evaluation calls, prompt selection by question type
│   ├── evaluation_prompts.py     # 5 prompt templates (STANDARD, CS, RUBRIC, STRICT, MCQ)
│   ├── ocr_module.py             # 6-engine hybrid OCR; typed-PDF fast-path; real-word scoring
│   ├── preprocessor.py           # ImagePreprocessor: upscale, denoise, deskew, CLAHE, threshold
│   ├── similarity.py             # SemanticSimilarityModel: cosine + sentence-level + fine_tune()
│   ├── rubric_matcher.py         # RubricMatcher: zero-shot NLI + optional BERT fine-tuning
│   ├── question_classifier.py    # QuestionClassifier: LLM → rule-based fallback, 7 types
│   ├── question_paper_parser.py  # Parses question paper PDF → ExamPaper + ExamQuestion
│   ├── answer_key_parser.py      # Parses answer key PDF → teacher answers, Set-A/B, rubrics
│   ├── student_answer_parser.py  # Segments booklet → StudentBooklet + StudentAnswerText
│   ├── diagram_detector.py       # YOLOv8n diagram detection with confidence threshold
│   ├── text_processor.py         # TextProcessor: normalise, spell-correct, tokenise
│   ├── metrics.py                # compute_metrics(), compute_mcq_metrics(), MetricsReport
│   ├── database.py               # 9 SQLAlchemy models + auto-migration + MetricsSnapshot
│   ├── schemas.py                # Pydantic v2 schemas — all request/response types
│   └── config.py                 # Settings dataclass, weight validation, active_llm property
│
├── frontend/
│   └── dashboard.py              # Streamlit teacher dashboard (all pages)
│
├── models/
│   └── trocr-finetuned/          # Place fine-tuned model here — auto-detected at startup
│
├── uploads/                      # Uploaded PDFs (auto-created)
├── assets/                       # Images for README and dashboard
├── Dockerfile.backend
├── Dockerfile.frontend
├── docker-compose.yml
├── requirements.txt
├── IntelliGrade_TrOCR_Finetune.ipynb
├── run.py
└── README.md
```

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/your-repo/IntelliGrade-H.git
cd IntelliGrade-H

# 2. Install
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# 3. Configure — only one line required
echo "GROQ_API_KEY=gsk_your_key_here" > .env

# 4. Run
python run.py
```

Open `http://localhost:8501` — the teacher dashboard is ready. The system runs fully offline for OCR (TrOCR + PaddleOCR + Tesseract) with only Groq needed for LLM evaluation.

---

## Installation

### Local

```bash
# 1. Clone
git clone https://github.com/your-repo/IntelliGrade-H.git
cd IntelliGrade-H

# 2. Python dependencies
pip install -r requirements.txt

# 3. NLP models
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# 4. Tesseract (system package)
# Linux:   sudo apt install tesseract-ocr poppler-utils
# macOS:   brew install tesseract poppler
# Windows: https://github.com/UB-Mannheim/tesseract/wiki
#          Set TESSERACT_CMD= in .env to the full .exe path

# 5. PaddleOCR (optional — skip if only using cloud OCR)
pip install paddlepaddle paddleocr

# 6. Configure and run
cp .env.example .env
# Edit .env — add GROQ_API_KEY at minimum
python run.py
```

### Docker (Recommended for Production)

```bash
cp .env.example .env   # add GROQ_API_KEY
docker compose up --build
```

Starts three containers: `backend` (port 8000), `frontend` (port 8501), `db` (PostgreSQL port 5432). Data persists in a named volume.

### System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| Python | 3.10 | 3.11 |
| RAM | 4 GB | 8 GB+ |
| Disk | 5 GB (models) | 10 GB+ |
| GPU | Not required | CUDA GPU for faster TrOCR |

---

## Environment Configuration

Only `GROQ_API_KEY` is required. All other settings have safe defaults.

```env
# ── LLM ────────────────────────────────────────────────────────────────────────
# LLM_PROVIDER=groq   → Groq primary, Claude fallback
# LLM_PROVIDER=claude → Claude primary, Groq fallback
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_...                       # Required — free at console.groq.com
GROQ_MODEL=llama-3.3-70b-versatile
ANTHROPIC_API_KEY=sk-ant-...               # Optional — Claude auto-activates as fallback
CLAUDE_MODEL=claude-haiku-4-5-20251001
LLM_MAX_TOKENS=6000                        # Max output tokens per LLM call
LLM_TEMPERATURE=0.1                        # Low = deterministic, consistent scoring

# ── OCR Cloud APIs (all optional — each one you add improves accuracy) ─────────
GOOGLE_VISION_API_KEY=                     # Free 1000 units/month
MISTRAL_API_KEY=                           # Free 1000 pages/month
AZURE_VISION_KEY=                          # Free 5000 pages/month, no expiry
AZURE_VISION_ENDPOINT=https://your-resource.cognitiveservices.azure.com

# ── OCR Local ─────────────────────────────────────────────────────────────────
TESSERACT_CMD=tesseract                    # Full path on Windows
OCR_DPI=400                               # 300–400 DPI recommended
OCR_WORKERS=2                             # Parallel local OCR threads
PADDLEOCR_LANG=en

# ── TrOCR ──────────────────────────────────────────────────────────────────────
TROCR_MODEL_PATH=microsoft/trocr-large-handwritten  # Default HuggingFace model
TROCR_FINETUNED_PATH=models/trocr-finetuned         # Auto-used when config.json found

# ── Diagram Detection ─────────────────────────────────────────────────────────
YOLO_MODEL_PATH=yolov8n.pt
DIAGRAM_CONF_THRESHOLD=0.35               # Lower = detect more (may increase FP)

# ── Semantic Similarity ───────────────────────────────────────────────────────
SBERT_MODEL=sentence-transformers/all-MiniLM-L6-v2

# ── Hybrid Scoring Weights (must sum to 1.0) ──────────────────────────────────
LLM_WEIGHT=0.40
SIMILARITY_WEIGHT=0.25
RUBRIC_WEIGHT=0.20
KEYWORD_WEIGHT=0.10
LENGTH_WEIGHT=0.05

# ── Database ──────────────────────────────────────────────────────────────────
DATABASE_URL=sqlite:///./intelligrade.db
# Prod: DATABASE_URL=postgresql://user:pass@localhost:5432/intelligrade

# ── Upload / API ──────────────────────────────────────────────────────────────
MAX_FILE_SIZE_MB=20
UPLOAD_DIR=./uploads
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
```

### `config.py` — Startup Validation

`get_settings()` (LRU-cached) runs two checks at startup:

1. **Weight sum check:** If `LLM_WEIGHT + SIMILARITY_WEIGHT + RUBRIC_WEIGHT + KEYWORD_WEIGHT + LENGTH_WEIGHT ≠ 1.0 (±0.01)`, a `UserWarning` is raised stating whether scores will be inflated or deflated.
2. **LLM key check:** If neither `GROQ_API_KEY` nor `ANTHROPIC_API_KEY` is set, a warning is raised that evaluation will use offline fallback scores.

The `Settings.active_llm` property returns which provider will actually be used (`"groq"`, `"claude"`, or `"offline"`) based on which keys are present and what `LLM_PROVIDER` is set to.

---

## Running the System

```bash
python run.py           # Both API + dashboard
python run.py api       # API only (port 8000)
python run.py ui        # Dashboard only (port 8501)
python run.py init      # Initialise / migrate database only
```

| Service | URL |
|---|---|
| Teacher Dashboard | http://localhost:8501 |
| REST API | http://localhost:8000 |
| API Docs (Swagger) | http://localhost:8000/docs |
| API Docs (Redoc) | http://localhost:8000/redoc |
| Health Check | http://localhost:8000/health |
| Metrics Debug | http://localhost:8000/metrics/print |

**Startup warm-up:** The API's `lifespan` handler runs `_warm_ocr()` in a background thread — this pre-loads TrOCR and PaddleOCR models so the first evaluation request doesn't incur model-loading latency.

---

## Fine-Tuning TrOCR on Your Exam Data

Fine-tuning on handwriting samples from your students is the **single highest-impact improvement** you can make to OCR accuracy. The system auto-detects and uses your model — no configuration change required.

### Why Fine-Tune? Model Comparison

| Model | CER on exam handwriting | VRAM needed |
|---|---|---|
| trocr-small, no fine-tuning | ~20–30% | ~2 GB |
| trocr-small, fine-tuned 1000 samples | ~15–20% | ~2 GB |
| trocr-large, no fine-tuning | ~15–22% | ~8 GB |
| **trocr-large, fine-tuned 500 samples** | **~10–15%** | **~8 GB** |
| **trocr-large, fine-tuned 1000+ samples** | **~6–11%** | **~8 GB** |
| Google Vision API (reference) | ~3–8% | Paid per page |

### Fine-Tuning Workflow

**Step 1 — Scan booklets** at 300–400 DPI (PNG). Anonymise student names.

**Step 2 — Crop into line images.** Each image = one line. Use `preprocessor.py`'s `segment_lines()` for automation.

**Step 3 — Label the images** (tab-separated `labels.txt`):

```
0001.png	The mitochondria is the powerhouse of the cell
0002.png	Newton second law states F equals ma
```

Use [Label Studio](https://labelstud.io/) for visual annotation. Two people can label 1000 samples in ~1 hour.

**Step 4 — Organise dataset:**

```
datasets/handwriting/
├── train/  images/ + labels.txt  (~80%)
├── val/    images/ + labels.txt  (~10%)
└── test/   images/ + labels.txt  (~10%)
```

**Step 5 — Open Colab:** `IntelliGrade_TrOCR_Finetune.ipynb` → T4 GPU → Run all cells.

**Step 6 — Deploy:**

```
1. Download trocr-finetuned/ from Google Drive
2. Place at: IntelliGrade-H/models/trocr-finetuned/
   (must contain config.json — triggers auto-detection)
3. Restart: python run.py
```

System logs: `Fine-tuned TrOCR model found at 'models/trocr-finetuned' — using it.`

### Dataset Format Rules

- Separator: **tab (`\t`)** — not spaces. Verify with `cat -A labels.txt | head -5` (tabs show as `^I`)
- Images: **PNG or JPG**
- Encoding: **UTF-8**
- Each image: **exactly one line of handwriting**

### Training Configuration

| Parameter | Value |
|---|---|
| Base model | microsoft/trocr-large-handwritten |
| Epochs | 15 |
| Batch Size | 8 |
| Gradient Accumulation | 4 (effective batch = 32) |
| Learning Rate | 5e-5 |
| Warmup Steps | 300 |
| Mixed Precision | FP16 |
| Data Augmentation | Enabled (brightness, contrast, rotation ±5°) |
| Evaluation | Per epoch — saves best checkpoint |

### Expected Training Time

| Dataset Size | Time on Colab T4 |
|---|---|
| 500 samples | ~20 minutes |
| 1000 samples | ~40 minutes |
| 5000 samples | ~2–3 hours |
| 10000 samples | ~4–5 hours |

### Inference After Fine-Tuning

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

**CUDA not detected:** `Runtime → Change runtime type → GPU (T4)` in Colab, then re-run from start.

**Out of memory:** Set `BATCH_SIZE = 4` and ensure `gradient_checkpointing=True` in `TrainingArguments`.

**Loss not decreasing:** Check for spaces instead of tabs in `labels.txt`; test image loading with `python -c "from PIL import Image; Image.open('0001.png')"`; split labels over 100 characters.

**Garbled output after deployment:** You must copy the entire `trocr-finetuned/` folder including `preprocessor_config.json` and `tokenizer_config.json` — not just the weight files.

---

## API Reference

Full interactive documentation at `http://localhost:8000/docs`. All endpoints listed below.

### Paper & Answer Key

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/paper/upload` | Upload question paper PDF — parses structure, stores as `ExamPaper` + `ExamQuestion` |
| `GET` | `/papers` | List all stored exam papers (lightweight, no questions) |
| `GET` | `/paper/{paper_id}` | Get full paper with all questions, marks, types, and teacher answers |
| `PATCH` | `/paper/{paper_id}/question/{q_num}` | Manually update a question's answer, type, or marks |
| `DELETE` | `/paper/{paper_id}` | Delete a paper and all its questions |
| `POST` | `/answer-key/upload` | Upload answer key PDF — links teacher answers to stored questions |

### Student Booklets (Primary Path)

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/booklet/upload` | Upload handwritten booklet PDF — OCR, segment, store as `StudentBooklet` |
| `POST` | `/booklet/{id}/evaluate` | Run full evaluation on all answers in a booklet (calls `evaluate_batch()`) |
| `GET` | `/booklets` | List all uploaded booklets |
| `GET` | `/booklet/{id}` | Get a booklet's parsed answers with their OCR text |
| `PATCH` | `/booklet/{id}/answer/{q_num}` | Manually correct a student's OCR-extracted answer text |
| `DELETE` | `/booklet/{id}` | Delete a booklet and all its answers and results |

### Single-Question Evaluation (Legacy & Flexible Path)

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/upload` | Upload a single answer sheet image or PDF — stores as `Submission` |
| `POST` | `/ocr/{submission_id}` | Run OCR on an uploaded submission |
| `POST` | `/evaluate` | Evaluate a submission with manually provided question/answer/marks |
| `POST` | `/evaluate/paper` | Evaluate against a stored `ExamQuestion` — marks and answer loaded from DB |
| `GET` | `/result/{submission_id}` | Get the evaluation result for a submission |

### Rubric, Metrics & System

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/rubric` | Upload rubric criteria for a question |
| `GET` | `/submissions` | List all submissions with their results |
| `GET` | `/stats` | System statistics (total booklets, submissions, average scores) |
| `GET` | `/metrics` | Cached AI accuracy metrics (MAE, Pearson r, Kappa) from `MetricsSnapshot` |
| `GET` | `/metrics/compute` | Ad-hoc metric recomputation from provided score lists |
| `GET` | `/metrics/print` | Print full metrics report to server log (debug) |
| `GET` | `/health` | Health check — returns status, version, and which models are loaded |
| `GET` | `/` | Root — returns API name and version |

### Example: Evaluate Against a Stored Paper Question

```bash
curl -X POST http://localhost:8000/evaluate/paper \
  -H "Content-Type: application/json" \
  -d '{
    "submission_id": "550e8400-e29b-41d4-a716-446655440000",
    "exam_paper_id": "S11BLH41_CAE1_Set-A",
    "question_number": 3
  }'
```

The system loads `max_marks`, `question_type`, and `teacher_answer` from the DB — nothing is hardcoded in the request.

### Example: Single Answer Evaluation (Inline)

```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "submission_id": "550e8400-e29b-41d4-a716-446655440001",
    "question": "Explain normalisation in DBMS.",
    "teacher_answer": "Normalisation reduces redundancy via 1NF, 2NF, 3NF, BCNF.",
    "max_marks": 5,
    "question_type": "open_ended"
  }'
```

### Example Response

```json
{
  "submission_id": "550e8400-...",
  "question_id": "...",
  "student_id": "...",
  "exam_paper_id": "S11BLH41_CAE1_Set-A",
  "question_number": 3,
  "llm_score": 3.8,
  "similarity_score": 0.74,
  "final_score": 3.5,
  "max_marks": 5.0,
  "strengths": ["Correctly defined normalisation", "Mentioned table decomposition"],
  "missing_concepts": ["Did not mention BCNF", "No discussion of anomalies"],
  "feedback": "Good basic understanding. Study higher normal forms.",
  "rubric_coverage": {"definition": 1, "example": 1, "bcnf": 0},
  "question_type": "open_ended",
  "ocr_text": "Normalisation is the process of organising data...",
  "ocr_confidence": 0.87,
  "confidence": 0.82,
  "word_count": 34,
  "evaluation_time_sec": 1.4,
  "percentage": 70.0
}
```

---

## Ethical Considerations

**Teacher control** — Every AI-generated grade is provisional until the teacher reviews and approves it. `PATCH` endpoints allow correction of both OCR errors and score errors before finalisation.

**Student privacy** — Student names and roll numbers are used only for matching within the institution's own infrastructure. No student identity data is included in LLM API calls — only the answer text.

**OCR noise does not penalise students** — All 5 prompt templates include an explicit instruction: *"Do NOT penalise OCR spelling errors — they are scanner artifacts, NOT student mistakes."*

**Transparent reasoning** — Every evaluation stores `strengths`, `missing_concepts`, `feedback`, and `explanation` (one-sentence score rationale). All fields are accessible via API and displayed in the dashboard.

**No bias amplification** — Prompts evaluate conceptual correctness only — not writing style, grammar, or language fluency — which could disadvantage non-native English speakers.

**Scoring weight integrity** — `config.py` raises a `UserWarning` at startup if weights do not sum to 1.0, preventing silent grade inflation/deflation.

**Rubric skipped for deterministic types** — The rubric matcher explicitly skips MCQ, True/False, and Numerical questions to avoid incorrect NLI-based partial credit on binary-graded items.

---

## Future Work

- **Mathematical equation evaluation** — LaTeX-aware scoring for formula derivations and proofs
- **Diagram understanding** — Vision-language models to evaluate drawn circuit diagrams and flowcharts structurally
- **Multilingual grading** — Tamil, Hindi, and other regional languages in mixed-language booklets
- **LMS integrations** — Direct grade push to Moodle, Google Classroom
- **Mobile scanning app** — iOS/Android app with automatic deskew and upload quality check
- **Continual learning** — Feedback loop from teacher corrections to improve accuracy over time
- **Plagiarism detection** — Cross-student similarity analysis to flag suspiciously similar answers
- **Offline LLM mode** — Fully air-gapped deployment using Ollama + Mistral 7B for strict data-policy environments
- **Similarity model fine-tuning** — Institution-specific `fine_tune()` on historical QA pairs to improve domain accuracy

---

## License

MIT License — free for academic and research use. Commercial use permitted under the same terms.