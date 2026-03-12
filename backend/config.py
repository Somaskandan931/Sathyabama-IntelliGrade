"""
IntelliGrade-H — Application Configuration
============================================
Single source of truth for all settings. Every value is read from .env
(or environment variables). Secrets are never hard-coded here.

LLM: Groq primary, Anthropic Claude as fallback.
     Set LLM_PROVIDER=groq (default) or LLM_PROVIDER=claude in .env.
Database: SQLite for dev (default), PostgreSQL for prod.
          Set DATABASE_URL=postgresql://... in .env for prod.
"""

import os
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()

# ── Base paths ─────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", str(BASE_DIR / "uploads")))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ── Application ────────────────────────────────────────────────────────────────
APP_NAME    = os.getenv("APP_NAME", "IntelliGrade-H")
APP_VERSION = os.getenv("APP_VERSION", "2.0.0")
DEBUG       = os.getenv("DEBUG", "false").lower() == "true"
SECRET_KEY  = os.getenv("SECRET_KEY", "change-me-in-production")

# ── Database ───────────────────────────────────────────────────────────────────
# Dev:  DATABASE_URL=sqlite:///./intelligrade.db
# Prod: DATABASE_URL=postgresql://intelligrade:pass@localhost:5432/intelligrade
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR}/intelligrade.db")

# ── LLM Providers ─────────────────────────────────────────────────────────────
# LLM_PROVIDER=groq   → Groq primary, Claude fallback if key present
# LLM_PROVIDER=claude → Claude primary, Groq fallback if key present
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower().strip()

# Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Anthropic / Claude
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL      = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001")

# LLM generation params
LLM_MAX_TOKENS  = int(os.getenv("LLM_MAX_TOKENS", "6000"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))

# ── OCR ────────────────────────────────────────────────────────────────────────
# OCR_ENGINE: "trocr" | "easyocr" | "ensemble"
# TROCR_MODEL_PATH: local fine-tuned dir (e.g. models/trocr-finetuned)
#                   or HuggingFace id (e.g. microsoft/trocr-small-handwritten)
OCR_ENGINE       = os.getenv("OCR_ENGINE", "trocr")
TROCR_MODEL_PATH = os.getenv("TROCR_MODEL_PATH", "microsoft/trocr-small-handwritten")
OCR_DPI          = int(os.getenv("OCR_DPI", "400"))
OCR_WORKERS      = int(os.getenv("OCR_WORKERS", "2"))

# Tesseract binary location (Windows default)
TESSERACT_CMD = os.getenv(
    "TESSERACT_CMD",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
)

# Cloud OCR keys (optional — leave blank to disable)
GOOGLE_VISION_API_KEY = os.getenv("GOOGLE_VISION_API_KEY", "")
AZURE_VISION_KEY      = os.getenv("AZURE_VISION_KEY", "")
AZURE_VISION_ENDPOINT = os.getenv("AZURE_VISION_ENDPOINT", "")
MISTRAL_API_KEY       = os.getenv("MISTRAL_API_KEY", "")

# ── Document Layout Detection ──────────────────────────────────────────────────
# "auto" | "detectron2" | "opencv_fallback"
LAYOUT_DETECTOR = os.getenv("LAYOUT_DETECTOR", "auto")

# ── Diagram Detection (YOLOv8) ────────────────────────────────────────────────
YOLO_MODEL_PATH        = os.getenv("YOLO_MODEL_PATH", "yolov8n.pt")
DIAGRAM_CONF_THRESHOLD = float(os.getenv("DIAGRAM_CONF_THRESHOLD", "0.35"))

# ── Semantic Similarity ────────────────────────────────────────────────────────
SBERT_MODEL = os.getenv("SBERT_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# ── Hybrid Scoring Weights (must sum to 1.0) ──────────────────────────────────
# Final = LLM×0.40 + Similarity×0.25 + Rubric×0.20 + Keyword×0.10 + Length×0.05
LLM_WEIGHT        = float(os.getenv("LLM_WEIGHT",        "0.40"))
SIMILARITY_WEIGHT = float(os.getenv("SIMILARITY_WEIGHT", "0.25"))
RUBRIC_WEIGHT     = float(os.getenv("RUBRIC_WEIGHT",     "0.20"))
KEYWORD_WEIGHT    = float(os.getenv("KEYWORD_WEIGHT",    "0.10"))
LENGTH_WEIGHT     = float(os.getenv("LENGTH_WEIGHT",     "0.05"))

# ── File Upload ────────────────────────────────────────────────────────────────
MAX_FILE_SIZE_MB    = int(os.getenv("MAX_FILE_SIZE_MB", "20"))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS  = {".jpg", ".jpeg", ".png", ".pdf", ".bmp", ".tiff"}

# ── API ────────────────────────────────────────────────────────────────────────
API_HOST     = os.getenv("API_HOST", "0.0.0.0")
API_PORT     = int(os.getenv("API_PORT", "8000"))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# ── Frontend ───────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


# ── Settings dataclass ─────────────────────────────────────────────────────────

@dataclass
class Settings:
    # LLM
    llm_provider: str      = LLM_PROVIDER
    groq_api_key: str      = GROQ_API_KEY
    groq_model: str        = GROQ_MODEL
    anthropic_api_key: str = ANTHROPIC_API_KEY
    claude_model: str      = CLAUDE_MODEL
    llm_max_tokens: int    = LLM_MAX_TOKENS
    llm_temperature: float = LLM_TEMPERATURE

    # OCR
    ocr_engine: str            = OCR_ENGINE
    trocr_model_path: str      = TROCR_MODEL_PATH
    ocr_dpi: int               = OCR_DPI
    ocr_workers: int           = OCR_WORKERS
    tesseract_cmd: str         = TESSERACT_CMD
    google_vision_api_key: str = GOOGLE_VISION_API_KEY
    azure_vision_key: str      = AZURE_VISION_KEY
    azure_vision_endpoint: str = AZURE_VISION_ENDPOINT
    mistral_api_key: str       = MISTRAL_API_KEY

    # Models / detection
    sbert_model: str              = SBERT_MODEL
    layout_detector: str          = LAYOUT_DETECTOR
    yolo_model_path: str          = YOLO_MODEL_PATH
    diagram_conf_threshold: float = DIAGRAM_CONF_THRESHOLD

    # Scoring weights
    llm_weight: float        = LLM_WEIGHT
    similarity_weight: float = SIMILARITY_WEIGHT
    rubric_weight: float     = RUBRIC_WEIGHT
    keyword_weight: float    = KEYWORD_WEIGHT
    length_weight: float     = LENGTH_WEIGHT

    # Storage / API
    database_url: str     = DATABASE_URL
    upload_dir: str       = str(UPLOAD_DIR)
    max_file_size_mb: int = MAX_FILE_SIZE_MB
    api_host: str         = API_HOST
    api_port: int         = API_PORT
    api_base_url: str     = API_BASE_URL
    log_level: str        = LOG_LEVEL

    @property
    def active_llm(self) -> str:
        """Which LLM provider will actually be used given present API keys."""
        if self.llm_provider == "claude" and self.anthropic_api_key:
            return "claude"
        if self.groq_api_key:
            return "groq"
        if self.anthropic_api_key:
            return "claude"          # fallback if groq key missing
        return "offline"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance. Validates weights and warns on issues."""
    s = Settings()

    # ── Weight validation ──────────────────────────────────────────────────────
    weight_sum = (
        s.llm_weight + s.similarity_weight + s.rubric_weight
        + s.keyword_weight + s.length_weight
    )
    if abs(weight_sum - 1.0) > 0.01:
        warnings.warn(
            f"Scoring weights sum to {weight_sum:.4f} (expected 1.0). "
            f"Scores will be {'inflated' if weight_sum > 1.0 else 'deflated'}. "
            "Check LLM_WEIGHT, SIMILARITY_WEIGHT, RUBRIC_WEIGHT, "
            "KEYWORD_WEIGHT, LENGTH_WEIGHT in your .env",
            stacklevel=2,
        )

    # ── LLM key validation ─────────────────────────────────────────────────────
    if not s.groq_api_key and not s.anthropic_api_key:
        warnings.warn(
            "No LLM API key found. Set GROQ_API_KEY or ANTHROPIC_API_KEY "
            "in .env. Evaluation will use offline fallback scores.",
            stacklevel=2,
        )

    return s