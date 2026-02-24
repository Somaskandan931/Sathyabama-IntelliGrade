"""
IntelliGrade-H - LLM Evaluator Module
Uses Google Gemini API to evaluate answers like a professor.
"""

import os
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class LLMEvaluation:
    score: float                     # out of max_marks
    max_marks: float
    strengths: list = field(default_factory=list)
    missing_concepts: list = field(default_factory=list)
    feedback: str = ""
    confidence: float = 0.0
    raw_response: str = ""


class LLMEvaluator:
    """
    Sends structured prompts to Gemini and parses evaluation results.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-pro"):
        self._api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self._model_name = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            import google.generativeai as genai
            if not self._api_key:
                raise ValueError("GEMINI_API_KEY is not set. Add it to .env file.")
            genai.configure(api_key=self._api_key)
            self._client = genai.GenerativeModel(self._model_name)
            logger.info(f"Gemini client initialized with model: {self._model_name}")
        return self._client

    # ─────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────

    def evaluate(
        self,
        question: str,
        teacher_answer: str,
        student_answer: str,
        max_marks: float = 10.0,
        rubric_criteria: Optional[list] = None
    ) -> LLMEvaluation:
        """
        Evaluates a student answer against the teacher answer.
        Returns structured LLMEvaluation.
        """
        prompt = self._build_prompt(
            question, teacher_answer, student_answer,
            max_marks, rubric_criteria
        )

        try:
            client = self._get_client()
            response = client.generate_content(prompt)
            raw = response.text
            logger.info("LLM evaluation complete.")
            return self._parse_response(raw, max_marks)
        except Exception as e:
            logger.error(f"LLM evaluation failed: {e}")
            return LLMEvaluation(
                score=0.0,
                max_marks=max_marks,
                feedback=f"Evaluation failed: {str(e)}",
                confidence=0.0
            )

    # ─────────────────────────────────────────────────────────
    # Prompt Builder
    # ─────────────────────────────────────────────────────────

    def _build_prompt(
        self,
        question: str,
        teacher_answer: str,
        student_answer: str,
        max_marks: float,
        rubric_criteria: Optional[list]
    ) -> str:

        rubric_section = ""
        if rubric_criteria:
            items = "\n".join(f"  - {c}" for c in rubric_criteria)
            rubric_section = f"""
RUBRIC CRITERIA (check if student covered each):
{items}
"""

        prompt = f"""You are an expert university professor evaluating a student's handwritten answer.

QUESTION:
{question}

MODEL ANSWER (written by teacher):
{teacher_answer}

STUDENT ANSWER:
{student_answer}

MAXIMUM MARKS: {max_marks}
{rubric_section}

EVALUATION CRITERIA:
1. Conceptual understanding and accuracy
2. Depth of explanation
3. Use of relevant examples
4. Structure and clarity
5. Coverage of key points from model answer

IMPORTANT INSTRUCTIONS:
- Be fair and consistent
- Give partial credit for partially correct answers
- Do NOT penalize for minor spelling errors (this is a handwritten answer with OCR)
- Focus on content, not writing style
- If the student answer is completely irrelevant, give 0

Respond ONLY with a valid JSON object in exactly this format (no markdown, no extra text):
{{
  "score": <float between 0 and {max_marks}>,
  "confidence": <float between 0 and 1 indicating your certainty about the score>,
  "strengths": [<list of string: things the student did well>],
  "missing_concepts": [<list of string: key concepts student missed>],
  "feedback": "<one paragraph of constructive feedback for the student>"
}}"""

        return prompt

    # ─────────────────────────────────────────────────────────
    # Response Parser
    # ─────────────────────────────────────────────────────────

    def _parse_response(self, raw: str, max_marks: float) -> LLMEvaluation:
        """Extract JSON from LLM response (handles markdown fences)."""
        # strip markdown code fences if present
        cleaned = re.sub(r"```json|```", "", raw).strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            # try to extract JSON block with regex
            match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except Exception:
                    data = {}
            else:
                data = {}

        score = float(data.get("score", 0.0))
        score = max(0.0, min(max_marks, score))   # clamp

        return LLMEvaluation(
            score=score,
            max_marks=max_marks,
            strengths=data.get("strengths", []),
            missing_concepts=data.get("missing_concepts", []),
            feedback=data.get("feedback", "No feedback generated."),
            confidence=float(data.get("confidence", 0.7)),
            raw_response=raw
        )
