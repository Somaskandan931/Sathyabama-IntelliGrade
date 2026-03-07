"""
backend/llm_examiner.py
LLM-based answer evaluation using Groq (primary) or Claude (fallback).

Loads the prompt template from prompts/evaluation_prompt.txt, sends it
to the LLM via llm_provider, and parses the structured JSON response.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from backend.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "evaluation_prompt.txt"

DEFAULT_PROMPT_TEMPLATE = (
    "You are an expert university professor evaluating a student's handwritten answer.\n\n"
    "QUESTION:\n{question}\n\n"
    "MODEL ANSWER (written by teacher):\n{teacher_answer}\n\n"
    "STUDENT ANSWER:\n{student_answer}\n\n"
    "MAXIMUM MARKS: {max_marks}\n\n"
    "RUBRIC:\n{rubric}\n\n"
    "Respond ONLY with valid JSON (no markdown):\n"
    '{"llm_score": <float 0-max_marks>, "confidence": <float 0-1>, '
    '"strengths": [<strings>], "missing_concepts": [<strings>], "feedback": "<string>"}'
)


@dataclass
class LLMEvaluation:
    llm_score: float
    strengths: list[str] = field(default_factory=list)
    missing_concepts: list[str] = field(default_factory=list)
    feedback: str = ""
    confidence: float = 0.8
    rubric_coverage: dict[str, float] = field(default_factory=dict)
    raw_response: str = ""
    latency_sec: float = 0.0


class LLMExaminer:
    """Evaluates student answers using Groq (primary) or Claude (fallback)."""

    def __init__(self):
        if not settings.groq_api_key and not settings.anthropic_api_key:
            raise ValueError(
                "No LLM API key set. Add GROQ_API_KEY or ANTHROPIC_API_KEY to your .env file."
            )
        from backend.llm_provider import LLMClient
        self._client = LLMClient.from_env()
        logger.info("LLMExaminer ready — active provider: %s", self._client.active_provider)

        # Load prompt template, fall back to inline default if file missing
        if PROMPT_PATH.exists():
            self._prompt_template = PROMPT_PATH.read_text(encoding="utf-8")
        else:
            logger.warning("Prompt file not found at %s — using inline default.", PROMPT_PATH)
            self._prompt_template = DEFAULT_PROMPT_TEMPLATE

    # ── Public API ─────────────────────────────────────────────────────────────
    def evaluate(
        self,
        question: str,
        teacher_answer: str,
        student_answer: str,
        rubric: dict[str, float] | None = None,
        max_marks: float = 10.0,
    ) -> LLMEvaluation:
        """
        Send the evaluation prompt to the LLM and parse the structured response.

        Args:
            question: Exam question text.
            teacher_answer: Model answer text.
            student_answer: OCR-extracted student answer text.
            rubric: Dict of {element: max_marks}, e.g. {"Definition": 2, "Example": 2}.
            max_marks: Total marks for this question.

        Returns:
            LLMEvaluation dataclass.
        """
        rubric = rubric or {}
        rubric_str = json.dumps(rubric, indent=2) if rubric else "No specific rubric provided."

        prompt = self._prompt_template.format(
            question=question,
            teacher_answer=teacher_answer,
            student_answer=student_answer,
            rubric=rubric_str,
            max_marks=max_marks,
        )

        t0 = time.perf_counter()
        try:
            response = self._client.generate(prompt)
            raw = response.text
        except Exception as exc:
            logger.error("LLM API error: %s", exc)
            raise

        latency = time.perf_counter() - t0
        return self._parse(raw, max_marks, latency)

    # ── Internal ───────────────────────────────────────────────────────────────
    @staticmethod
    def _parse(raw: str, max_marks: float, latency: float) -> LLMEvaluation:
        """Extract JSON from raw LLM output and populate LLMEvaluation."""
        cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("Could not parse LLM JSON; attempting regex extraction.")
            data = _extract_json_fallback(cleaned)

        score = float(data.get("llm_score", 0))
        score = max(0.0, min(max_marks, score))  # clamp

        return LLMEvaluation(
            llm_score=score,
            strengths=data.get("strengths", []),
            missing_concepts=data.get("missing_concepts", []),
            feedback=data.get("feedback", ""),
            confidence=float(data.get("confidence", 0.7)),
            rubric_coverage=data.get("rubric_coverage", {}),
            raw_response=raw,
            latency_sec=round(latency, 3),
        )


def _extract_json_fallback(text: str) -> dict:
    """Last-resort: grab first {...} block from text."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


# ── Module-level singleton ─────────────────────────────────────────────────────
_examiner: Optional[LLMExaminer] = None


def get_llm_examiner() -> LLMExaminer:
    global _examiner
    if _examiner is None:
        _examiner = LLMExaminer()
    return _examiner