"""
backend/text_processing.py
Post-OCR text cleaning and NLP preprocessing pipeline.

Steps:
    1. Basic normalisation (whitespace, unicode)
    2. Spell correction  (autocorrect)
    3. Sentence segmentation (spaCy)
    4. Tokenisation
    5. Stop-word removal (NLTK)
    6. Return structured TextDoc
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache

import nltk
import spacy
from autocorrect import Speller

# Download NLTK data if not present
for _pkg in ("stopwords", "punkt"):
    try:
        nltk.data.find(f"tokenizers/{_pkg}" if _pkg == "punkt" else f"corpora/{_pkg}")
    except LookupError:
        nltk.download(_pkg, quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


@dataclass
class TextDoc:
    raw: str
    cleaned: str
    sentences: list[str] = field(default_factory=list)
    tokens: list[str] = field(default_factory=list)
    filtered_tokens: list[str] = field(default_factory=list)  # no stop-words
    word_count: int = 0


@lru_cache(maxsize=1)
def _load_spacy() -> spacy.language.Language:
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        # Auto-download small model
        from spacy.cli import download
        download("en_core_web_sm")
        return spacy.load("en_core_web_sm")


_spell = Speller(lang="en")
_stop_words = set(stopwords.words("english"))


def process_text(raw_text: str) -> TextDoc:
    """Full text-processing pipeline. Returns a TextDoc."""
    cleaned = _normalise(raw_text)
    cleaned = _spell_correct(cleaned)

    nlp = _load_spacy()
    doc = nlp(cleaned)

    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    tokens = word_tokenize(cleaned.lower())
    filtered = [t for t in tokens if t.isalpha() and t not in _stop_words]

    return TextDoc(
        raw=raw_text,
        cleaned=cleaned,
        sentences=sentences,
        tokens=tokens,
        filtered_tokens=filtered,
        word_count=len([t for t in tokens if t.isalpha()]),
    )


# ── Internal helpers ──────────────────────────────────────────────────────────
def _normalise(text: str) -> str:
    """Unicode normalisation, collapse whitespace, remove control chars."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)  # control chars
    text = re.sub(r"\s+", " ", text)               # collapse whitespace
    return text.strip()


def _spell_correct(text: str) -> str:
    """Word-level spell correction (preserves numbers and proper nouns)."""
    words = text.split()
    corrected: list[str] = []
    for w in words:
        # Skip numbers, all-caps acronyms, short words
        if w.isnumeric() or (w.isupper() and len(w) > 1) or len(w) <= 2:
            corrected.append(w)
        else:
            corrected.append(_spell(w))
    return " ".join(corrected)
