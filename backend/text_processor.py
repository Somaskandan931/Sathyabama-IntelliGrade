"""
IntelliGrade-H - Text Processing Module
Cleans and normalizes OCR output for downstream NLP.
"""

import re
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ProcessedText:
    raw: str
    cleaned: str
    sentences: list
    tokens: list


class TextProcessor:
    """
    Post-processes OCR text:
    1. Spell correction
    2. Whitespace / punctuation normalization
    3. Sentence segmentation
    4. Tokenization
    """

    def __init__(self, language: str = "en"):
        self.language = language
        self._spellchecker = None
        self._nlp = None

    # ─────────────────────────────────────────────────────────
    # Lazy loaders
    # ─────────────────────────────────────────────────────────

    def _get_spellchecker(self):
        if self._spellchecker is None:
            from spellchecker import SpellChecker
            self._spellchecker = SpellChecker(language=self.language)
            # Add common academic / technical words
            self._spellchecker.word_frequency.load_words([
                "algorithm", "preprocessing", "tokenization", "neuron",
                "backpropagation", "gradient", "sigmoid", "relu",
                "convolution", "transformer", "embedding", "cosine",
                "classifier", "regression", "hyperparameter", "dataset",
                "overfitting", "underfitting", "epoch", "batch",
                "learning", "training", "validation", "inference",
                "perceptron", "recurrent", "lstm", "attention", "bert",
                "scalability", "bandwidth", "latency", "throughput",
                "synchronous", "asynchronous", "microservices",
                "polymorphism", "encapsulation", "inheritance"
            ])
        return self._spellchecker

    def _get_nlp(self):
        if self._nlp is None:
            try:
                import spacy
                self._nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy model not found. Run: python -m spacy download en_core_web_sm")
                self._nlp = None
        return self._nlp

    # ─────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────

    def process(self, raw_text: str, apply_spellcheck: bool = True) -> ProcessedText:
        cleaned = self._normalize(raw_text)
        if apply_spellcheck:
            cleaned = self._spellcheck(cleaned)
        sentences = self._segment_sentences(cleaned)
        tokens = self._tokenize(cleaned)

        return ProcessedText(
            raw=raw_text,
            cleaned=cleaned,
            sentences=sentences,
            tokens=tokens
        )

    # ─────────────────────────────────────────────────────────
    # Private methods
    # ─────────────────────────────────────────────────────────

    def _normalize(self, text: str) -> str:
        """Fix whitespace, remove control characters, normalize quotes."""
        # remove non-printable characters except newline and space
        text = re.sub(r'[^\x20-\x7E\n]', ' ', text)
        # normalize multiple spaces/newlines
        text = re.sub(r'\n+', '. ', text)
        text = re.sub(r' {2,}', ' ', text)
        # fix missing space after period
        text = re.sub(r'\.([A-Z])', r'. \1', text)
        # normalize quotes
        text = text.replace('\u2019', "'").replace('\u201c', '"').replace('\u201d', '"')
        return text.strip()

    def _spellcheck(self, text: str) -> str:
        """Correct obvious OCR spelling errors word by word."""
        sc = self._get_spellchecker()
        words = text.split()
        corrected = []
        for word in words:
            # preserve words with numbers, punctuation-only tokens, or ALL CAPS
            clean_word = re.sub(r'[^a-zA-Z]', '', word)
            if (
                not clean_word
                or any(c.isdigit() for c in word)
                or clean_word.isupper()
                or len(clean_word) <= 2
            ):
                corrected.append(word)
            else:
                correction = sc.correction(clean_word.lower())
                if correction and correction != clean_word.lower():
                    # preserve original casing
                    if clean_word[0].isupper():
                        correction = correction.capitalize()
                    word = word.replace(clean_word, correction)
                corrected.append(word)

        return " ".join(corrected)

    def _segment_sentences(self, text: str) -> list:
        """Use spaCy for sentence segmentation; fallback to regex."""
        nlp = self._get_nlp()
        if nlp:
            doc = nlp(text)
            return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        else:
            # simple regex fallback
            sentences = re.split(r'(?<=[.!?])\s+', text)
            return [s.strip() for s in sentences if s.strip()]

    def _tokenize(self, text: str) -> list:
        """Tokenize and remove stopwords."""
        nlp = self._get_nlp()
        if nlp:
            doc = nlp(text.lower())
            return [
                token.lemma_ for token in doc
                if not token.is_stop and not token.is_punct and token.is_alpha
            ]
        else:
            import nltk
            try:
                from nltk.corpus import stopwords
                from nltk.tokenize import word_tokenize
                nltk.download("punkt", quiet=True)
                nltk.download("stopwords", quiet=True)
                stop_words = set(stopwords.words("english"))
                tokens = word_tokenize(text.lower())
                return [t for t in tokens if t.isalpha() and t not in stop_words]
            except Exception:
                return text.lower().split()
