"""
IntelliGrade-H - Question Type Classifier
Automatically detects question type from question text using an LLM.

Supported question types:
  - mcq           : Multiple choice (A/B/C/D options)
  - true_false    : True or False question
  - fill_blank    : Fill in the blank / complete the sentence
  - short_answer  : 1-3 sentence factual answer
  - open_ended    : Long descriptive / essay-style answer
  - numerical     : Compute a number / formula result
  - diagram       : Draw or label a diagram (OCR extracts any text description)

Falls back to rule-based classification if LLM is unavailable.
"""

import re
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# Result
# ─────────────────────────────────────────────────────────

@dataclass
class ClassificationResult:
    question_type: str        # one of the 7 types above
    confidence: float         # 0.0 – 1.0
    reasoning: str            # brief explanation
    method: str               # "llm" | "rule_based"


VALID_TYPES = {
    "mcq", "true_false", "fill_blank",
    "short_answer", "open_ended", "numerical", "diagram"
}

# Human-readable labels used in UI
TYPE_LABELS = {
    "mcq":          "Multiple Choice (MCQ)",
    "true_false":   "True / False",
    "fill_blank":   "Fill in the Blank",
    "short_answer": "Short Answer",
    "open_ended":   "Open-Ended / Essay",
    "numerical":    "Numerical / Calculation",
    "diagram":      "Diagram / Diagram Description",
}

# Which types use the full LLM+similarity pipeline vs. deterministic grading
LLM_PIPELINE_TYPES  = {"open_ended", "short_answer", "diagram"}
SIMILARITY_TYPES    = {"open_ended", "short_answer"}
DETERMINISTIC_TYPES = {"mcq", "true_false", "fill_blank", "numerical"}


# ─────────────────────────────────────────────────────────
# Classifier
# ─────────────────────────────────────────────────────────

class QuestionClassifier:
    """
    Two-stage classifier:
    1. Try LLM classification (fast, accurate).
    2. Fall back to rule-based regex if LLM fails or is unavailable.
    """

    CLASSIFY_PROMPT = """You are an academic question classifier.

Given a question, classify it into exactly one of these types:
- mcq          : Has options A, B, C, D (or similar lettered/numbered choices)
- true_false   : Answer is True or False only
- fill_blank   : Has a blank (___) to fill in, or says "complete the sentence"
- short_answer : Expects 1-3 sentence factual answer (define, state, list, name, what is)
- open_ended   : Expects a detailed paragraph/essay (explain, describe, discuss, analyze, compare, evaluate)
- numerical    : Expects a calculated number, formula, or mathematical result (calculate, find, compute, solve)
- diagram      : Asks to draw, label, or sketch a diagram or flowchart

QUESTION:
{question}

Respond ONLY with valid JSON (no markdown, no extra text):
{{
  "question_type": "<one of the 7 types above>",
  "confidence": <float 0.0 to 1.0>,
  "reasoning": "<one sentence explanation>"
}}"""

    def __init__(self, llm_evaluator=None):
        """
        llm_evaluator: an LLMEvaluator instance (optional).
        If None, falls back to rule-based only.
        """
        self._llm = llm_evaluator

    def classify(self, question: str) -> ClassificationResult:
        """
        Classify a question. Tries LLM first, then falls back to rule-based.
        """
        if not question or not question.strip():
            return ClassificationResult(
                question_type="open_ended",
                confidence=0.0,
                reasoning="Empty question — defaulting to open-ended.",
                method="rule_based",
            )

        # Try LLM classification first
        if self._llm is not None:
            try:
                result = self._classify_with_llm(question.strip())
                if result:
                    return result
            except Exception as e:
                logger.warning("LLM classification failed, using rule-based: %s", e)

        # Rule-based fallback
        return self._classify_rule_based(question.strip())

    # ─────────────────────────────────────────────────────
    # LLM classification
    # ─────────────────────────────────────────────────────

    def _classify_with_llm(self, question: str) -> Optional[ClassificationResult]:
        import json, re

        prompt = self.CLASSIFY_PROMPT.format(question=question)

        try:
            client = self._llm._get_client()
            response = client.generate(prompt)
            raw = response.text
        except Exception as e:
            logger.warning("LLM call failed: %s", e)
            return None

        # Parse JSON
        cleaned = re.sub(r"```json|```", "", raw).strip()
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            m = re.search(r'\{.*\}', cleaned, re.DOTALL)
            data = json.loads(m.group()) if m else {}

        qtype = data.get("question_type", "").lower().strip()
        if qtype not in VALID_TYPES:
            logger.warning("LLM returned unknown type '%s', falling back.", qtype)
            return None

        return ClassificationResult(
            question_type=qtype,
            confidence=float(data.get("confidence", 0.8)),
            reasoning=data.get("reasoning", ""),
            method="llm",
        )

    # ─────────────────────────────────────────────────────
    # Rule-based fallback
    # ─────────────────────────────────────────────────────

    def _classify_rule_based(self, question: str) -> ClassificationResult:
        q = question.lower()

        # MCQ — has lettered options
        if re.search(r'\b[a-eA-E]\s*[\)\.]\s+\w', question) or \
           re.search(r'\(a\)|\(b\)|\(c\)|\(d\)', q) or \
           re.search(r'\bwhich\s+of\s+the\s+following\b', q) or \
           re.search(r'\bselect\s+(the\s+)?(correct|best|right)\b', q) or \
           re.search(r'\bchoose\s+(the\s+)?(correct|best|right|one)\b', q):
            return ClassificationResult("mcq", 0.92,
                "Found lettered options or 'which of the following' pattern.", "rule_based")

        # True/False
        if re.search(r'\b(true\s+or\s+false|state\s+whether|is\s+it\s+true|t\s*[\/|]\s*f)\b', q):
            return ClassificationResult("true_false", 0.95,
                "Found 'true or false' pattern.", "rule_based")

        # Fill in the blank
        if re.search(r'_{2,}|\[.*?\]|\(\s*\.\.\.\s*\)|\bfill\s+in\b|\bcomplete\s+the\b', q):
            return ClassificationResult("fill_blank", 0.93,
                "Found blank or fill-in-the-blank pattern.", "rule_based")

        # Numerical
        if re.search(r'\b(calculate|compute|find|solve|determine|evaluate|what\s+is\s+the\s+value|how\s+many|how\s+much)\b', q) and \
           re.search(r'\b(value|result|answer|total|sum|difference|product|quotient|area|volume|speed|force|energy|mass|resistance|current)\b', q):
            return ClassificationResult("numerical", 0.85,
                "Found numerical computation keywords.", "rule_based")

        # Diagram
        if re.search(r'\b(draw|sketch|label|illustrate|diagram|flowchart|draw\s+and\s+explain)\b', q):
            return ClassificationResult("diagram", 0.90,
                "Found diagram/draw keyword.", "rule_based")

        # Short answer — define, state, list, name, what is
        if re.search(r'\b(define|state|list|name|what\s+is|what\s+are|give\s+(an?\s+)?example|mention|identify|who\s+is|when\s+was|where\s+is)\b', q) and \
           len(question.split()) < 30:
            return ClassificationResult("short_answer", 0.80,
                "Short factual question with define/state/list pattern.", "rule_based")

        # Open-ended — explain, describe, discuss, compare, analyze
        if re.search(r'\b(explain|describe|discuss|analyze|analyse|compare|contrast|evaluate|elaborate|justify|examine|assess|critically|in\s+detail)\b', q):
            return ClassificationResult("open_ended", 0.88,
                "Found essay/descriptive keyword.", "rule_based")

        # Default
        return ClassificationResult("open_ended", 0.50,
            "No clear pattern found — defaulting to open-ended.", "rule_based")