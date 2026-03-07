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

# Overview

Evaluating subjective answers is traditionally:

* time-consuming
* inconsistent between evaluators
* difficult to scale for large classes

IntelliGrade-H solves this problem using a **multi-stage AI pipeline** that converts handwritten answers into structured feedback and marks within seconds.

The system integrates:

* **Handwriting Recognition**
* **Document Layout Understanding**
* **Semantic Answer Evaluation**
* **Rubric-Aware Grading**
* **AI Feedback Generation**
* **Teacher Analytics Dashboard**

This enables automated grading while still maintaining **transparent and explainable evaluation results**.

---

# Key Features

* Automatic handwriting recognition from scanned answer sheets
* Document layout understanding for complex answer pages
* Per-question answer segmentation
* Semantic similarity scoring between student and teacher answers
* Rubric-aware grading using zero-shot AI models
* AI-generated feedback highlighting strengths and missing concepts
* Diagram detection for visual answers
* Batch grading for entire exam submissions
* Teacher dashboard for analytics and grading review
* REST API for LMS integration

---

# System Architecture

```
Answer Sheet
     │
     ▼
Image Preprocessing
(OpenCV)
     │
     ▼
Layout Detection
(OpenCV)
     │
     ▼
Answer Segmentation
     │
     ▼
OCR Ensemble
(TrOCR + EasyOCR + Tesseract)
     │
     ▼
Diagram Detection
(YOLOv8)
     │
     ▼
Text Processing
(spaCy)
     │
     ▼
Evaluation Engine
 ├ Semantic Similarity (Sentence-BERT)
 ├ Rubric Matcher (DeBERTa)
 ├ Keyword Coverage
 └ LLM Examiner (Groq / Claude)
     │
     ▼
Hybrid Scoring Engine
     │
     ▼
Feedback Generator
     │
     ▼
Teacher Dashboard
(Streamlit)
```

<p align="center">
  <img src="assets/ChatGPT Image Mar 6, 2026, 11_19_15 AM.png" alt="System Architecture" width="800"/>
</p>

The IntelliGrade-H architecture processes scanned answer sheets through a multi-stage AI pipeline.

First, the system preprocesses exam images using OpenCV to improve readability. OpenCV connected-components analysis then performs document layout detection to identify question regions and answer blocks. These segmented answers are passed through an OCR ensemble consisting of TrOCR, EasyOCR, and Tesseract to extract handwritten text.

The extracted text is analyzed using NLP models. Sentence-BERT computes semantic similarity between student and reference answers, while DeBERTa evaluates rubric compliance. The LLM Examiner uses Groq (primary) or Claude (fallback) to generate professor-style feedback explaining strengths and missing concepts.

All evaluation signals are combined in a hybrid scoring engine that produces the final grade. Results are presented to instructors through an interactive Streamlit dashboard and accessible via a FastAPI backend.

---

# Grading Engine

Instead of relying on a single scoring method, IntelliGrade-H uses a **hybrid grading model**.

```
Final Score =
0.40 × LLM Evaluation
0.25 × Semantic Similarity
0.20 × Rubric Coverage
0.10 × Keyword Coverage
0.05 × Length Normalization
```

This multi-factor scoring system improves fairness and better approximates **human grading behavior**.

---

# Technology Stack

## AI / Machine Learning

### PyTorch

PyTorch is used as the primary deep learning framework powering the AI models used in handwriting recognition, semantic similarity, and rubric evaluation.

### Transformers

The Transformers library provides access to modern transformer-based models used throughout the system including TrOCR, Sentence-BERT, and DeBERTa.

### Sentence-BERT

Sentence-BERT generates semantic embeddings of student and teacher answers, allowing the system to compute **meaning-based similarity rather than simple keyword matching**.

### DeBERTa

DeBERTa is used for **rubric-based evaluation** by performing natural language inference between rubric criteria and student answers.

---

## LLM Providers

IntelliGrade-H uses a multi-provider LLM setup with automatic fallback. The active provider chain is:

**Groq → Claude → Rule-Based Fallback**

### Groq (Primary)

Groq runs `llama-3.3-70b-versatile` and serves as the primary LLM provider. It offers fast inference speeds suitable for real-time grading workflows. Set `GROQ_API_KEY` in your `.env` file.

### Claude (Fallback)

Anthropic's Claude (`claude-3-haiku-20240307`) acts as the quality fallback when Groq is unavailable. It generates detailed professor-style feedback with high accuracy. Set `ANTHROPIC_API_KEY` in your `.env` file.

### Rule-Based Fallback

If both cloud providers are unavailable, the system falls back to a deterministic rule-based evaluator to ensure grading is never fully blocked.

To pin a specific provider, set `LLM_PROVIDER=groq` or `LLM_PROVIDER=claude` in your `.env` file.

---

## Computer Vision

### OpenCV

OpenCV performs preprocessing on scanned exam sheets to improve OCR accuracy.

Operations include:

* grayscale conversion
* noise removal
* skew correction
* contrast enhancement
* binarization

### OpenCV — Layout Detection

In addition to image preprocessing, OpenCV performs **document layout detection** using connected-components analysis to identify answer blocks, question numbers, and structural elements within exam sheets.

This allows the system to separate answers for different questions without requiring any additional installation.

> **Note:** Detectron2 is not supported on Windows. The system uses OpenCV as the active layout detector (`LAYOUT_DETECTOR=opencv_fallback` in `.env`), which handles standard answer sheet layouts reliably.

### YOLOv8

YOLOv8 is used to detect **diagrams and visual elements** inside student answers, enabling grading of answers that include flowcharts, architectures, or labeled figures.

---

## Optical Character Recognition

To achieve reliable handwriting recognition, IntelliGrade-H uses **multiple OCR systems together**.

### TrOCR

TrOCR is a transformer-based handwriting recognition model developed by Microsoft. It serves as the **primary OCR engine** for extracting handwritten text.

### EasyOCR

EasyOCR provides strong performance on mixed text types including handwritten labels, numbers, and short tokens. It acts as a secondary OCR engine when TrOCR confidence is low.

### Tesseract

Tesseract serves as a fallback OCR system and performs well on printed text, exam instructions, and structured forms.

Using multiple OCR engines ensures **robust recognition across diverse handwriting styles**.

---

## Backend

### FastAPI

FastAPI provides the REST API layer responsible for orchestrating the entire grading pipeline.

The API supports:

* uploading answer sheets
* performing OCR
* running evaluation
* retrieving results
* computing grading metrics

### Uvicorn

Uvicorn is the high-performance ASGI server used to run the FastAPI application.

---

## Frontend

### Streamlit

Streamlit provides an interactive **teacher dashboard** where instructors can:

* upload exam sheets
* review AI grades
* analyze class performance
* export grading reports

---

## Deployment

### Docker

Docker packages the entire system and its dependencies into containers, enabling easy deployment across local machines, servers, and cloud environments.

---

# Project Structure

```
IntelliGrade-H
│
├── backend
│   ├── api.py
│   ├── evaluator.py
│   ├── llm_provider.py
│   ├── llm_evaluator.py
│   ├── llm_examiner.py
│   ├── ocr_module.py
│   ├── preprocessor.py
│   ├── similarity.py
│   ├── rubric_matcher.py
│   ├── question_classifier.py
│   ├── config.py
│   ├── metrics.py
│
├── frontend
│   └── dashboard.py
│
├── models
│   └── train_trocr.py
│
├── datasets
│   └── collect_dataset.py
│
├── tests
│   └── test_all.py
│
├── prompts
│   └── evaluation_prompt.txt
│
├── uploads
│
├── .env
├── requirements.txt
├── docker-compose.yml
└── README.md
```

---

# Configuration

All settings are loaded from a `.env` file in the project root. The minimum required keys are:

```env
# LLM Providers — at least one is required
LLM_PROVIDER=groq                          # groq | claude | auto
GROQ_API_KEY=gsk_...                       # https://console.groq.com
GROQ_MODEL=llama-3.3-70b-versatile

ANTHROPIC_API_KEY=sk-ant-...               # https://console.anthropic.com
CLAUDE_MODEL=claude-3-haiku-20240307

# Layout Detection — use opencv_fallback on Windows (Detectron2 not supported)
LAYOUT_DETECTOR=opencv_fallback            # opencv_fallback | auto | detectron2

# Hybrid Scoring Weights (must sum to 1.0)
LLM_WEIGHT=0.40
SIMILARITY_WEIGHT=0.25
RUBRIC_WEIGHT=0.20
KEYWORD_WEIGHT=0.10
LENGTH_WEIGHT=0.05
```

---

# Running the System

## Install Dependencies

```
pip install -r requirements.txt
```

Install spaCy language model:

```
python -m spacy download en_core_web_sm
```

---

## Start Backend API

```
uvicorn backend.api:app --reload
```

API will run at:

```
http://localhost:8000
```

Swagger API documentation:

```
http://localhost:8000/docs
```

---

## Launch Teacher Dashboard

```
streamlit run frontend/dashboard.py
```

Dashboard will run at:

```
http://localhost:8501
```

---

# Evaluation Metrics

To validate grading accuracy, IntelliGrade-H compares AI scores with teacher-provided ground truth.

Target metrics for high-quality evaluation:

| Metric              | Target |
| ------------------- | ------ |
| Mean Absolute Error | < 0.8  |
| Pearson Correlation | > 0.85 |
| Cohen Kappa         | > 0.75 |
| Accuracy ±1 mark    | > 90%  |

These metrics ensure the AI grading system aligns closely with human evaluation.

---

# Ethical Considerations

IntelliGrade-H is designed with responsible AI principles.

* Student identities are anonymized
* AI grading remains advisory and requires teacher review
* Feedback is transparent and explainable
* OCR errors are handled gracefully without unfair penalties

---

# Future Improvements

Potential extensions of the system include:

* mathematical equation evaluation
* multilingual grading support
* diagram understanding with vision models
* LMS integrations (Moodle, Google Classroom)
* mobile scanning application
* continuous learning from teacher corrections

---

# Research Potential

IntelliGrade-H has strong potential for academic publication and further development into a scalable educational platform.

Possible research directions include:

* automated subjective answer grading
* explainable AI in educational assessment
* multimodal document understanding
* human-AI collaborative grading systems