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

## Problem Statement

Manual evaluation of handwritten subjective answers at scale is:

- **Time-consuming** — days of effort for large cohorts
- **Inconsistent** — scores vary significantly between evaluators
- **Unscalable** — impractical for hundreds of students per subject

**IntelliGrade-H automates this end-to-end** — from a scanned booklet to final marks with structured, transparent feedback. All AI grades remain teacher-reviewable before finalisation.

---

## Key Features

| Feature           | Description |
|-------------------|---|
| 6-Engine Hybrid OCR | Google Vision → Mistral → Azure → PaddleOCR → Tesseract → TrOCR cascade |
| Fine-Tunable TrOCR | `trocr-large-handwritten` auto-replaced by your domain-trained model when present |
| Dual LLM Support  | Groq LLaMA 3.3-70B (primary) + Anthropic Claude Haiku (automatic fallback) |
| Semantic Similarity | Sentence-BERT cosine similarity + sentence-level breakdown |
| Rubric Matching   | Zero-shot NLI rubric coverage via cross-encoder/nli-deberta-v3-small |
| Answer Key Parser | Auto-extracts model answers from teacher PDF (typed or scanned) |
| Question Paper Parser | Detects parts, marks, OR alternatives, 7 question types automatically |
| Bulk Grading     | Evaluate an entire class in one upload with CSV export |
| Analytics Dashboard | MAE, Pearson r, Cohen's Kappa, score distribution charts |
| Docker Ready    | One-command deployment with PostgreSQL support |

---

## 🏗️ System Architecture

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

## Hybrid Scoring Formula

```
Final Score =
  0.40 × LLM Evaluation Score        (Groq LLaMA 3.3-70B / Claude Haiku)
  0.25 × Semantic Similarity          (Sentence-BERT cosine)
  0.20 × Rubric Coverage              (Zero-Shot NLI)
  0.10 × Keyword Coverage
  0.05 × Length Normalisation
```

Weights are configurable via `.env` and validated at startup — a warning is raised if they do not sum to 1.0.

---

## OCR Pipeline

The OCR system uses a **6-engine cascade** ordered by accuracy. Cloud APIs return immediately on a good result; local engines compete and the best result wins.

```
1. Google Cloud Vision API   ← Best general handwriting accuracy
2. Mistral OCR               ← Document-optimised, 1000 pages/month free
3. Azure AI Vision           ← 5000 pages/month free, no expiry
4. PaddleOCR                 ← Best local engine for mixed layouts
5. Tesseract (PSM 11)        ← Solid layout-aware fallback
6. TrOCR-Large               ← Fine-tunable handwriting transformer
```

**Typed PDFs** (question papers, answer keys) bypass the OCR pipeline entirely via pdfplumber/PyMuPDF — instant, 100% accurate.

**Fine-tuned model auto-detection:** When `models/trocr-finetuned/config.json` exists, the system uses your domain-trained model automatically. No configuration change needed.

---

## LLM Integration

**Primary:** Groq — `llama-3.3-70b-versatile`  
**Fallback:** Anthropic Claude — `claude-haiku-4-5-20251001` — auto-activates when `ANTHROPIC_API_KEY` is set

LLMs handle: answer evaluation with structured feedback, answer key extraction from teacher PDFs, student booklet segmentation (which answer belongs to which question), cover page metadata extraction (roll number, set, course, semester), and MCQ disambiguation when OCR confidence is low.

Evaluation prompt strategies (selected automatically by question type):

| Prompt | Used for |
|---|---|
| `STANDARD_PROMPT` | General open-ended answers |
| `CS_ENGINEERING_PROMPT` | DBMS, algorithms, OS, Networks, code-aware |
| `RUBRIC_PROMPT` | Per-criterion mark breakdown |
| `STRICT_PROMPT` | Board-exam style marking |
| `MCQ_VALIDATION_PROMPT` | MCQ when OCR confidence < 0.5 |

---

## Dashboard Features

**Paper Manager** — Upload question paper PDF → auto-extract questions, marks, parts, OR alternatives, question types

**Answer Key Manager** — Upload teacher answer key PDF → auto-extract model answers; supports Set-A / Set-B

**Student Booklets** — Upload scanned booklet → OCR → segment answers → evaluate → structured feedback with strengths, missing concepts, sentence-level similarity breakdown

**Bulk Upload** — Upload full class booklets → batch process → export CSV with all scores and feedback

**Analytics** — MAE, Pearson r, Cohen's Kappa, accuracy within ±1 and ±0.5 marks, score distribution chart

---

## Evaluation Metrics & Targets

| Metric | Target |
|---|---|
| Mean Absolute Error (MAE) | < 0.8 marks |
| Pearson Correlation | > 0.85 |
| Cohen's Kappa | > 0.75 |
| Accuracy within ±1 mark | > 90% |

---

## Technology Stack

| Layer | Technology |
|---|---|
| OCR (cloud) | Google Vision API, Mistral OCR, Azure AI Vision |
| OCR (local) | PaddleOCR, Tesseract, TrOCR-Large (HuggingFace) |
| Image Processing | OpenCV, PIL (CLAHE, deskew, adaptive threshold) |
| NLP | Sentence-BERT (all-MiniLM-L6-v2), spaCy en_core_web_sm |
| Rubric Matching | cross-encoder/nli-deberta-v3-small (zero-shot NLI) |
| Diagram Detection | YOLOv8n (Ultralytics) |
| LLM (primary) | Groq — llama-3.3-70b-versatile |
| LLM (fallback) | Anthropic — claude-haiku-4-5-20251001 |
| Backend | FastAPI, SQLAlchemy, SQLite (dev) / PostgreSQL (prod) |
| Frontend | Streamlit |
| Deep Learning | PyTorch, HuggingFace Transformers |

---

## Project Structure

```
IntelliGrade-H/
│
├── backend/
│   ├── api.py                    # FastAPI routes (20+ endpoints)
│   ├── evaluator.py              # Hybrid scoring engine
│   ├── llm_provider.py           # Groq + Claude multi-provider client
│   ├── llm_evaluator.py          # LLM evaluation and prompt routing
│   ├── evaluation_prompts.py     # Prompt templates
│   ├── ocr_module.py             # 6-engine hybrid OCR pipeline
│   ├── preprocessor.py           # Image preprocessing
│   ├── similarity.py             # Sentence-BERT + sentence-level breakdown
│   ├── rubric_matcher.py         # Zero-shot NLI rubric matching
│   ├── question_classifier.py    # Auto question type detection (7 types)
│   ├── question_paper_parser.py  # Question paper PDF parser
│   ├── answer_key_parser.py      # Answer key PDF parser
│   ├── student_answer_parser.py  # Student booklet parser and segmenter
│   ├── diagram_detector.py       # YOLOv8 + heuristic diagram detection
│   ├── text_processor.py         # NLP cleaning and spell correction
│   ├── metrics.py                # MAE, Pearson r, Cohen's Kappa
│   ├── database.py               # SQLAlchemy models + auto-migration
│   ├── schemas.py                # Pydantic v2 request/response schemas
│   └── config.py                 # Environment configuration with validation
│
├── frontend/
│   └── dashboard.py              # Streamlit teacher dashboard
│
├── models/
│   └── trocr-finetuned/          # Drop your fine-tuned model here (auto-detected)
│
├── uploads/                      # Uploaded PDFs (auto-created)
├── Dockerfile.backend
├── Dockerfile.frontend
├── docker-compose.yml
├── requirements.txt
├── IntelliGrade_TrOCR_Finetune.ipynb  # Google Colab fine-tuning notebook
├── run.py
└── README.md
```

---

## Installation

### Local

```bash
# 1. Clone
git clone https://github.com/your-repo/IntelliGrade-H.git
cd IntelliGrade-H

# 2. Install dependencies
pip install -r requirements.txt

# 3. Post-install
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# 4. Tesseract
# Linux:   sudo apt install tesseract-ocr poppler-utils
# macOS:   brew install tesseract poppler
# Windows: https://github.com/UB-Mannheim/tesseract/wiki

# 5. Configure and run
cp .env.example .env   # fill in GROQ_API_KEY at minimum
python run.py
```

### Docker

```bash
cp .env.example .env   # add GROQ_API_KEY
docker compose up --build
```

---

## Environment Configuration

```env
# ── LLM (at least one key required) ──────────────────────────────────────
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_...                       # free at console.groq.com
GROQ_MODEL=llama-3.3-70b-versatile
ANTHROPIC_API_KEY=sk-ant-...               # optional — Claude auto-activates as fallback
CLAUDE_MODEL=claude-haiku-4-5-20251001

# ── OCR Cloud APIs (each one improves accuracy, all optional) ─────────────
GOOGLE_VISION_API_KEY=
MISTRAL_API_KEY=                           # 1000 pages/month free
AZURE_VISION_KEY=                          # 5000 pages/month free, no expiry
AZURE_VISION_ENDPOINT=https://your-resource.cognitiveservices.azure.com

# ── OCR Local ─────────────────────────────────────────────────────────────
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
OCR_DPI=400
OCR_WORKERS=2
PADDLEOCR_LANG=en

# ── TrOCR (fine-tuned model auto-detected — no change needed after deploy) ─
TROCR_FINETUNED_PATH=models/trocr-finetuned
TROCR_MODEL_PATH=microsoft/trocr-large-handwritten

# ── Diagram Detection ─────────────────────────────────────────────────────
YOLO_MODEL_PATH=yolov8n.pt
DIAGRAM_CONF_THRESHOLD=0.35

# ── Semantic Similarity ───────────────────────────────────────────────────
SBERT_MODEL=sentence-transformers/all-MiniLM-L6-v2

# ── Hybrid Scoring Weights (must sum to 1.0) ──────────────────────────────
LLM_WEIGHT=0.40
SIMILARITY_WEIGHT=0.25
RUBRIC_WEIGHT=0.20
KEYWORD_WEIGHT=0.10
LENGTH_WEIGHT=0.05

# ── Database ──────────────────────────────────────────────────────────────
DATABASE_URL=sqlite:///./intelligrade.db
# Production: DATABASE_URL=postgresql://user:pass@localhost:5432/intelligrade

# ── Upload / API ──────────────────────────────────────────────────────────
MAX_FILE_SIZE_MB=20
UPLOAD_DIR=./uploads
API_HOST=0.0.0.0
API_PORT=8000
```

---

## Running the System

```bash
python run.py           # start both API + dashboard
python run.py api       # API only
python run.py ui        # dashboard only
python run.py init      # initialise database only
```

| Service | URL |
|---|---|
| Teacher Dashboard | http://localhost:8501 |
| REST API | http://localhost:8000 |
| API Documentation | http://localhost:8000/docs |
| Metrics Debug | http://localhost:8000/metrics/print |

---

## 🔬 Fine-Tuning TrOCR on Your Exam Data

Fine-tuning on handwriting samples from your own students is the highest-impact improvement you can make. The system auto-detects and uses your model — no configuration change required.

### Which model to fine-tune: `microsoft/trocr-large-handwritten`

The notebook has been updated from `trocr-small` to **`trocr-large-handwritten`**. Here is exactly why:

| Model | CER on exam handwriting | VRAM needed |
|---|---|---|
| trocr-small, no fine-tuning | ~20–30% | ~2 GB |
| trocr-small, fine-tuned 1000 samples | ~15–20% | ~2 GB |
| trocr-large, no fine-tuning | ~15–22% | ~8 GB |
| **trocr-large, fine-tuned 500 samples** | **~10–15%** | **~8 GB** |
| **trocr-large, fine-tuned 1000+ samples** | **~6–11%** | **~8 GB** |
| Google Vision API (reference point) | ~3–8% | Paid per page |

`trocr-large` has 4× more parameters. On variable, messy exam handwriting this difference is decisive. The Colab free T4 GPU has 16 GB VRAM — `large` fits comfortably at batch size 8 with gradient checkpointing.

On domain-specific vocabulary (DBMS, algorithm names, circuit diagrams) your fine-tuned model can match or exceed Google Vision because it is trained specifically on your students' handwriting style, while Google's model is general-purpose.

### Fine-tuning workflow

**Step 1 — Scan booklets** at 300–400 DPI (PNG). Anonymise student names.

**Step 2 — Crop into line images.** Each image = one line of handwriting. The `preprocessor.py` `segment_lines()` method can do this automatically, or crop manually.

**Step 3 — Create labels.txt** in each split folder (tab-separated):
```
0001.png	The mitochondria is the powerhouse of the cell
0002.png	Newton second law states F equals ma
```
Use [Label Studio](https://labelstud.io/) (free) for a visual annotation interface, or type directly into a spreadsheet. Two people can label 1000 samples in about one hour.

**Step 4 — Upload dataset to Google Drive:**
```
My Drive/Intelligrade/datasets/handwriting/
├── train/    images/ + labels.txt    (~80% of samples)
├── val/      images/ + labels.txt    (~10%)
└── test/     images/ + labels.txt    (~10%)
```

**Step 5 — Open the notebook in Colab:**

`IntelliGrade_TrOCR_Finetune.ipynb` → `Runtime → Change runtime type → T4 GPU` → Run all cells.

- 1000 samples, 15 epochs: ~35–45 min on T4
- 5000 samples, 15 epochs: ~2–3 hours on T4

**Step 6 — Deploy:**
```
1. Download trocr-finetuned/ from Google Drive
2. Extract to: IntelliGrade-H/models/trocr-finetuned/
   (folder must contain config.json — this triggers auto-detection)
3. Restart: python run.py

The system logs:
   Fine-tuned TrOCR model found at models/trocr-finetuned — using it.
```

No `.env` changes needed.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/paper/upload` | Upload and parse a question paper PDF |
| `GET` | `/papers` | List all exam papers |
| `GET` | `/paper/{paper_id}` | Get full paper with questions |
| `POST` | `/answer-key/upload` | Upload and extract answer key |
| `POST` | `/booklet/upload` | Upload a student booklet PDF |
| `POST` | `/booklet/{id}/evaluate` | OCR + evaluate all answers |
| `POST` | `/evaluate` | Single-question evaluation |
| `POST` | `/evaluate/paper` | Evaluate against a known exam paper |
| `GET` | `/submissions` | List all submissions |
| `GET` | `/stats` | System statistics |
| `GET` | `/metrics` | AI accuracy metrics |
| `GET` | `/metrics/print` | Print metrics to server log (debug) |
| `POST` | `/rubric` | Upload rubric criteria for a question |
| `POST` | `/bulk/evaluate` | Batch evaluate multiple booklets |
| `DELETE` | `/booklet/{id}` | Delete a booklet and its results |

Full interactive documentation: `http://localhost:8000/docs`

---

## Ethical Considerations

- Student identities are anonymised during processing
- All AI grades are teacher-reviewable before finalisation — IntelliGrade-H is a grading assistant, not a replacement for the teacher
- OCR artefacts do **not** penalise students — all evaluation prompts explicitly instruct the LLM to ignore OCR noise
- Evaluation reasoning (strengths, missing concepts, score rationale) is stored transparently and exportable per submission
- The system raises a startup warning if scoring weights do not sum to 1.0, preventing silent grade inflation or deflation

---

## Future Work

- Mathematical equation and formula evaluation
- Diagram understanding using vision-language models
- Multilingual answer grading (Tamil, Hindi)
- LMS integrations (Moodle, Google Classroom)
- Mobile app for scanning exam booklets
- Continual learning loop from teacher corrections

---

## License

MIT License — free for academic and research use.