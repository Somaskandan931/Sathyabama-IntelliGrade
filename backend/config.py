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

# ── LLM Providers ──────────────────────────────────────────────────────────
GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL      = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GROQ_API_KEY      = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL        = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL      = os.getenv("CLAUDE_MODEL", "claude-3-haiku-20240307")
OLLAMA_BASE_URL   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL      = os.getenv("OLLAMA_MODEL", "llama3.2")

# ── OCR ────────────────────────────────────────────────────────────────────
# Options: "easyocr" | "trocr" | "ensemble"
OCR_ENGINE       = os.getenv("OCR_ENGINE", "easyocr")
TROCR_MODEL_PATH = os.getenv("TROCR_MODEL_PATH", "microsoft/trocr-small-handwritten")

# ── Semantic Similarity ────────────────────────────────────────────────────
SBERT_MODEL = os.getenv("SBERT_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# ── Scoring Weights ────────────────────────────────────────────────────────
LLM_WEIGHT        = float(os.getenv("LLM_WEIGHT", "0.6"))
SIMILARITY_WEIGHT = float(os.getenv("SIMILARITY_WEIGHT", "0.4"))

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
