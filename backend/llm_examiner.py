"""
backend/llm_examiner.py
LLM-based answer evaluation using the Google Gemini API.

Loads the prompt template from prompts/evaluation_prompt.txt, sends it
to Gemini, and parses the structured JSON response.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import google.generativeai as genai

from backend.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "evaluation_prompt.txt"


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
    """Evaluates student answers using Gemini."""

    def __init__(self):
        if not settings.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Add it to your .env file."
            )
        genai.configure(api_key=settings.gemini_api_key)
        self._model = genai.GenerativeModel(settings.gemini_model)
        self._prompt_template = PROMPT_PATH.read_text(encoding="utf-8")

    # ── Public API ────────────────────────────────────────────────────────────
    def evaluate(
        self,
        question: str,
        teacher_answer: str,
        student_answer: str,
        rubric: dict[str, float] | None = None,
        max_marks: float = 10.0,
    ) -> LLMEvaluation:
        """
        Send the evaluation prompt to Gemini and parse the response.

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
            response = self._model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,      # low temperature → consistent grading
                    max_output_tokens=1024,
                ),
            )
            raw = response.text
        except Exception as exc:
            logger.error("Gemini API error: %s", exc)
            raise

        latency = time.perf_counter() - t0
        return self._parse(raw, max_marks, latency)

    # ── Internal ──────────────────────────────────────────────────────────────
    @staticmethod
    def _parse(raw: str, max_marks: float, latency: float) -> LLMEvaluation:
        """Extract JSON from raw LLM output and populate LLMEvaluation."""
        # Strip markdown fences if present
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


# ── Module-level singleton ────────────────────────────────────────────────────
_examiner: Optional[LLMExaminer] = None


def get_llm_examiner() -> LLMExaminer:
    global _examiner
    if _examiner is None:
        _examiner = LLMExaminer()
    return _examiner
