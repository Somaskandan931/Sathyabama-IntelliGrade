"""
IntelliGrade-H - LLM Evaluator Module
Evaluates student answers using Gemini 2.5 Flash (primary) or Groq/Llama (fallback).

Supports all question types:
  - open_ended   : full professor-style rubric evaluation
  - short_answer : concise factual check
  - fill_blank   : exact/near-exact match check
  - numerical    : numerical answer comparison with tolerance
  - diagram      : evaluate text description of a diagram
  - true_false   : not routed here (deterministic in evaluator.py)
  - mcq          : not routed here (deterministic in evaluator.py)
"""

import logging
import re
import json
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from prompts.evaluation_prompts import (
    STANDARD_PROMPT,
    CS_ENGINEERING_PROMPT,
    RUBRIC_PROMPT,
    STRICT_PROMPT,
    MCQ_VALIDATION_PROMPT,
)

logger = logging.getLogger(__name__)


@dataclass
class LLMEvaluation:
    score: float
    max_marks: float
    strengths: list = field(default_factory=list)
    missing_concepts: list = field(default_factory=list)
    feedback: str = ""
    confidence: float = 0.0
    provider: str = ""
    model: str = ""
    raw_response: str = ""
    rubric_breakdown: Optional[Dict] = None
    latency_ms: float = 0.0


class LLMEvaluator:
    """
    Evaluates student answers via the multi-provider LLM client.
    Builds different prompts depending on question type.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self._api_key    = api_key
        self._model_name = model
        self._client     = None

    def _get_client(self):
        if self._client is None:
            import os
            from backend.llm_provider import LLMClient

            # Set environment variables if provided
            if self._api_key:
                os.environ.setdefault("GEMINI_API_KEY", self._api_key)
            if self._model_name:
                os.environ.setdefault("GEMINI_MODEL", self._model_name)

            self._client = LLMClient.from_env()
            logger.info("LLMEvaluator using: %s", self._client.active_provider)
        return self._client

    # ─────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────

    def evaluate(
        self,
        question: str,
        teacher_answer: str,
        student_answer: str,
        max_marks: float = 10.0,
        rubric_criteria: Optional[list] = None,
        question_type: str = "open_ended",
    ) -> LLMEvaluation:
        """
        Route to the correct prompt builder based on question_type.
        MCQ and True/False are handled deterministically in evaluator.py.
        """
        if not student_answer or not student_answer.strip():
            return LLMEvaluation(
                score=0.0, max_marks=max_marks,
                feedback="No student answer was provided for evaluation.",
                confidence=1.0,
            )

        prompt_builders = {
            "open_ended":   self._build_open_ended_prompt,
            "short_answer": self._build_short_answer_prompt,
            "fill_blank":   self._build_fill_blank_prompt,
            "numerical":    self._build_numerical_prompt,
            "diagram":      self._build_diagram_prompt,
        }

        builder = prompt_builders.get(question_type, self._build_open_ended_prompt)
        prompt  = builder(question, teacher_answer, student_answer, max_marks, rubric_criteria)

        try:
            client   = self._get_client()

            # For JSON response, use generate_json for better parsing
            if question_type in ["open_ended", "short_answer", "diagram"]:
                data = client.generate_json(prompt)

                # Extract data with defaults
                score = float(data.get("score", 0.0))
                score = max(0.0, min(max_marks, score))

                # Get provider info from last used provider
                provider_info = client.active_provider
                provider_parts = provider_info.split('(')
                provider_name = provider_parts[0] if provider_parts else "unknown"
                model_name = provider_parts[1].rstrip(')') if len(provider_parts) > 1 else "unknown"

                return LLMEvaluation(
                    score=score,
                    max_marks=max_marks,
                    strengths=data.get("strengths", []),
                    missing_concepts=data.get("missing_concepts", []),
                    feedback=data.get("feedback", "No feedback generated."),
                    confidence=float(data.get("confidence", 0.7)),
                    provider=provider_name,
                    model=model_name,
                    rubric_breakdown=data.get("rubric_breakdown"),
                    raw_response=json.dumps(data),
                )
            else:
                # For simple response types, use regular generate
                response = client.generate(prompt)
                result = self._parse_response(response.text, max_marks)
                result.provider = response.provider
                result.model = response.model
                result.latency_ms = response.latency_ms
                logger.info("Evaluation complete via %s/%s (%.2fms)",
                          response.provider, response.model, response.latency_ms)
                return result

        except Exception as e:
            logger.error("LLM evaluation failed: %s", e, exc_info=True)
            return LLMEvaluation(
                score=0.0, max_marks=max_marks,
                feedback=f"Evaluation failed: {str(e)}. Please try again.",
                confidence=0.0,
            )

    # ─────────────────────────────────────────────────────
    # Prompt Builders
    # ─────────────────────────────────────────────────────

    def _rubric_section(self, rubric_criteria: Optional[list]) -> str:
        if not rubric_criteria:
            return ""
        items = "\n".join(f"  - {c['criterion']} ({c['marks']} marks)"
                          if isinstance(c, dict) else f"  - {c}"
                          for c in rubric_criteria)
        return f"\nRUBRIC CRITERIA (check if student covered each):\n{items}\n"

    def _json_footer(self, max_marks: float) -> str:
        return f"""
Respond ONLY with valid JSON (no markdown, no extra text):
{{
  "score": <float between 0 and {max_marks}>,
  "confidence": <float between 0 and 1>,
  "strengths": [<list of strings: what student did well>],
  "missing_concepts": [<list of strings: key concepts missed>],
  "feedback": "<one paragraph of constructive feedback>"
}}"""

    def _build_prompt(
        self,
        question: str,
        teacher_answer: str,
        student_answer: str,
        max_marks: float,
        rubric_criteria: Optional[list] = None,
        question_type: str = "open_ended",
        style: str = "standard",
    ) -> str:
        """
        Unified prompt builder — used by tests and internally.
        Routes to the correct specialised builder based on question_type.
        For open_ended questions the 'style' kwarg selects which prompt
        template from evaluation_prompts.py to use:
          'standard'    → STANDARD_PROMPT
          'cs'          → CS_ENGINEERING_PROMPT
          'rubric'      → RUBRIC_PROMPT  (requires rubric_criteria)
          'strict'      → STRICT_PROMPT
        """
        if question_type in ("open_ended",):
            if style == "cs":
                rubric_section = self._rubric_section(rubric_criteria) if rubric_criteria else ""
                return CS_ENGINEERING_PROMPT.format(
                    question=question,
                    teacher_answer=teacher_answer,
                    student_answer=student_answer,
                    max_marks=max_marks,
                    rubric_section=rubric_section,
                )
            elif style == "rubric" and rubric_criteria:
                rubric_items = "\n".join(
                    f"  - {c['criterion']}: {c['marks']} marks"
                    if isinstance(c, dict) else f"  - {c}"
                    for c in rubric_criteria
                )
                return RUBRIC_PROMPT.format(
                    question=question,
                    teacher_answer=teacher_answer,
                    student_answer=student_answer,
                    max_marks=max_marks,
                    rubric_items=rubric_items,
                )
            elif style == "strict":
                return STRICT_PROMPT.format(
                    question=question,
                    teacher_answer=teacher_answer,
                    student_answer=student_answer,
                    max_marks=max_marks,
                )
            else:
                # Default: STANDARD_PROMPT
                return STANDARD_PROMPT.format(
                    question=question,
                    teacher_answer=teacher_answer,
                    student_answer=student_answer,
                    max_marks=max_marks,
                )

        # Route to specialised builders for other question types
        builders = {
            "short_answer": self._build_short_answer_prompt,
            "fill_blank":   self._build_fill_blank_prompt,
            "numerical":    self._build_numerical_prompt,
            "diagram":      self._build_diagram_prompt,
        }
        builder = builders.get(question_type, self._build_open_ended_prompt)
        return builder(question, teacher_answer, student_answer, max_marks, rubric_criteria)

    def _build_open_ended_prompt(self, question, teacher_answer, student_answer, max_marks, rubric_criteria) -> str:
        # Use STANDARD_PROMPT template from evaluation_prompts.py
        # Fall back to inline prompt if rubric is present (STANDARD_PROMPT doesn't have rubric section)
        if rubric_criteria:
            rubric_items = "\n".join(
                f"  - {c['criterion']}: {c['marks']} marks"
                if isinstance(c, dict) else f"  - {c}"
                for c in rubric_criteria
            )
            return RUBRIC_PROMPT.format(
                question=question,
                teacher_answer=teacher_answer,
                student_answer=student_answer,
                max_marks=max_marks,
                rubric_items=rubric_items,
            )
        return STANDARD_PROMPT.format(
            question=question,
            teacher_answer=teacher_answer,
            student_answer=student_answer,
            max_marks=max_marks,
        )

    def _build_short_answer_prompt(self, question, teacher_answer, student_answer, max_marks, rubric_criteria) -> str:
        return f"""You are grading a short-answer question. Expected response is 1-3 sentences.

QUESTION TYPE: Short Answer

QUESTION:
{question}

EXPECTED ANSWER:
{teacher_answer}

STUDENT ANSWER:
{student_answer}

MAXIMUM MARKS: {max_marks}
{self._rubric_section(rubric_criteria)}
GRADING GUIDELINES:
- Full marks: Correct key facts, precise terminology, complete answer
- 50-99%: Mostly correct but missing minor detail or slightly imprecise
- 1-49%: Partially correct, shows some understanding
- 0: Wrong, irrelevant, or blank

Do NOT penalize OCR spelling errors. Award marks for correct concepts even if phrasing differs.

{self._json_footer(max_marks)}"""

    def _build_fill_blank_prompt(self, question, teacher_answer, student_answer, max_marks, rubric_criteria) -> str:
        return f"""You are grading a fill-in-the-blank question.

QUESTION TYPE: Fill in the Blank

QUESTION (with blank):
{question}

CORRECT ANSWER(S) for the blank:
{teacher_answer}

STUDENT'S ANSWER:
{student_answer}

MAXIMUM MARKS: {max_marks}

GRADING GUIDELINES:
- Full marks: Exact match or semantically equivalent answer
- Partial marks: Synonym, abbreviation, or partially correct term
- 0: Incorrect, blank, or irrelevant
- Accept spelling variations caused by OCR (e.g. "backpropogation" = "backpropagation")
- If there are multiple blanks, award proportional marks for each correct one

{self._json_footer(max_marks)}"""

    def _build_numerical_prompt(self, question, teacher_answer, student_answer, max_marks, rubric_criteria) -> str:
        return f"""You are grading a numerical / calculation question.

QUESTION TYPE: Numerical

QUESTION:
{question}

CORRECT ANSWER / WORKING:
{teacher_answer}

STUDENT'S ANSWER / WORKING:
{student_answer}

MAXIMUM MARKS: {max_marks}

GRADING GUIDELINES:
- Full marks: Correct final answer AND correct method/working
- 70-99%: Correct method but minor arithmetic error
- 40-69%: Correct setup/formula but wrong execution
- 10-39%: Some correct steps but fundamentally wrong approach
- 0: No working shown, completely wrong, or irrelevant
- OCR may misread digits/operators — give benefit of the doubt for clear method

{self._json_footer(max_marks)}"""

    def _build_diagram_prompt(self, question, teacher_answer, student_answer, max_marks, rubric_criteria) -> str:
        return f"""You are grading a diagram-based question. The student was asked to draw/label a diagram.
Their handwritten labels and annotations have been extracted via OCR.

QUESTION TYPE: Diagram

QUESTION:
{question}

EXPECTED DIAGRAM ELEMENTS / DESCRIPTION:
{teacher_answer}

STUDENT'S OCR-EXTRACTED TEXT (labels, annotations from their diagram):
{student_answer}

MAXIMUM MARKS: {max_marks}
{self._rubric_section(rubric_criteria)}
GRADING GUIDELINES:
- Evaluate based on labels, annotations, and textual descriptions
- Full marks: All key components labeled correctly with proper relationships
- Partial marks: Most elements present, minor omissions
- OCR may miss diagram elements — be generous with interpretation
- 0: No relevant content or completely wrong diagram

{self._json_footer(max_marks)}"""

    # ─────────────────────────────────────────────────────
    # Response Parser
    # ─────────────────────────────────────────────────────

    def _parse_response(self, raw: str, max_marks: float) -> LLMEvaluation:
        import json

        # Clean the response
        cleaned = re.sub(r"```json|```|`", "", raw).strip()

        try:
            # Try to find JSON object
            json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(cleaned)
        except Exception as e:
            logger.warning(f"Failed to parse JSON: {e}. Raw: {raw[:200]}")
            data = {}

        # Extract and validate score
        score = float(data.get("score", 0.0))
        score = max(0.0, min(max_marks, score))

        return LLMEvaluation(
            score=score,
            max_marks=max_marks,
            strengths=data.get("strengths", []),
            missing_concepts=data.get("missing_concepts", []),
            feedback=data.get("feedback", "No feedback generated."),
            confidence=float(data.get("confidence", 0.7)),
            raw_response=raw,
        )