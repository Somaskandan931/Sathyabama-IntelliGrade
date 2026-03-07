"""
IntelliGrade-H - Application Configuration
Loads all settings from environment variables / .env file.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Base paths ─────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", str(BASE_DIR / "uploads")))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ── Application ────────────────────────────────────────────────────────────
APP_NAME    = os.getenv("APP_NAME", "IntelliGrade-H")
APP_VERSION = os.getenv("APP_VERSION", "2.0.0")
DEBUG       = os.getenv("DEBUG", "false").lower() == "true"
SECRET_KEY  = os.getenv("SECRET_KEY", "change-me-in-production")

# ── Database ───────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR}/intelligrade.db")

# ── LLM Providers (Groq primary, Claude fallback) ──────────────────────────
GROQ_API_KEY      = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL        = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL      = os.getenv("CLAUDE_MODEL", "claude-3-haiku-20240307")

# ── OCR ────────────────────────────────────────────────────────────────────
# Options: "easyocr" | "trocr" | "ensemble"
OCR_ENGINE       = os.getenv("OCR_ENGINE", "easyocr")
TROCR_MODEL_PATH = os.getenv("TROCR_MODEL_PATH", "microsoft/trocr-small-handwritten")

# ── Document Layout Detection ──────────────────────────────────────────────
# Options: "detectron2" | "opencv_fallback" | "auto"
LAYOUT_DETECTOR  = os.getenv("LAYOUT_DETECTOR", "auto")

# ── Diagram Detection ──────────────────────────────────────────────────────
# Path to custom YOLOv8 weights, or use default "yolov8n.pt"
YOLO_MODEL_PATH  = os.getenv("YOLO_MODEL_PATH", "yolov8n.pt")
DIAGRAM_CONF_THRESHOLD = float(os.getenv("DIAGRAM_CONF_THRESHOLD", "0.35"))

# ── Semantic Similarity ────────────────────────────────────────────────────
SBERT_MODEL = os.getenv("SBERT_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# ── Hybrid Scoring Weights (must sum to 1.0) ───────────────────────────────
# Formula: Final = LLM×0.40 + Similarity×0.25 + Rubric×0.20 + Keyword×0.10 + Length×0.05
LLM_WEIGHT        = float(os.getenv("LLM_WEIGHT",        "0.40"))
SIMILARITY_WEIGHT = float(os.getenv("SIMILARITY_WEIGHT", "0.25"))
RUBRIC_WEIGHT     = float(os.getenv("RUBRIC_WEIGHT",     "0.20"))
KEYWORD_WEIGHT    = float(os.getenv("KEYWORD_WEIGHT",    "0.10"))
LENGTH_WEIGHT     = float(os.getenv("LENGTH_WEIGHT",     "0.05"))

# ── File Upload ────────────────────────────────────────────────────────────
MAX_FILE_SIZE_MB    = int(os.getenv("MAX_FILE_SIZE_MB", "20"))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS  = {".jpg", ".jpeg", ".png", ".pdf", ".bmp", ".tiff"}

# ── API ────────────────────────────────────────────────────────────────────
API_HOST     = os.getenv("API_HOST", "0.0.0.0")
API_PORT     = int(os.getenv("API_PORT", "8000"))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# ── Logging ────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ── Settings dataclass (used by llm_examiner.py and other modules) ──────────

from dataclasses import dataclass
from functools import lru_cache


@dataclass
class Settings:
    groq_api_key: str = GROQ_API_KEY
    groq_model: str = GROQ_MODEL
    anthropic_api_key: str = ANTHROPIC_API_KEY
    claude_model: str = CLAUDE_MODEL
    ocr_engine: str = OCR_ENGINE
    trocr_model_path: str = TROCR_MODEL_PATH
    sbert_model: str = SBERT_MODEL
    layout_detector: str = LAYOUT_DETECTOR
    yolo_model_path: str = YOLO_MODEL_PATH
    diagram_conf_threshold: float = DIAGRAM_CONF_THRESHOLD
    llm_weight: float = LLM_WEIGHT
    similarity_weight: float = SIMILARITY_WEIGHT
    rubric_weight: float = RUBRIC_WEIGHT
    keyword_weight: float = KEYWORD_WEIGHT
    length_weight: float = LENGTH_WEIGHT
    database_url: str = DATABASE_URL
    upload_dir: str = str(UPLOAD_DIR)
    max_file_size_mb: int = MAX_FILE_SIZE_MB
    api_host: str = API_HOST
    api_port: int = API_PORT
    log_level: str = LOG_LEVEL


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance populated from environment variables."""
    return Settings()