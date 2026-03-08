"""
IntelliGrade-H — LLM Provider (v4 — Claude Primary)
====================================================
Changes vs v3:
  • Claude is now PRIMARY provider (best quality for evaluation tasks)
  • Groq is SECONDARY fallback (fast but less nuanced)
  • Claude uses claude-haiku-4-5 (fast + affordable) by default;
    set CLAUDE_MODEL=claude-sonnet-4-5 in .env for higher quality
  • generate() and generate_json() added (required by llm_evaluator.py)
  • Retry with exponential back-off on rate-limit (429) errors
  • LLMResponse dataclass returned by generate()
  • Offline JSON fallback unchanged
"""

import os
import re
import json
import time
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# ── Provider config ────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
# claude-haiku-4-5: fast + cheap; swap to claude-sonnet-4-5 for higher quality
CLAUDE_MODEL      = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001")

GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL    = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3")

MAX_TOKENS  = int(os.getenv("LLM_MAX_TOKENS", "1500"))
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))


@dataclass
class LLMResponse:
    """Standardised response returned by LLMClient.generate()."""
    text: str
    provider: str = ""
    model: str = ""
    latency_ms: float = 0.0


class LLMClient:
    """
    Multi-provider LLM client.
    Priority: Claude → Groq → Offline heuristic
    """

    def __init__(self):
        self._providers: list[str] = []
        self._anthropic_client = None
        self._groq_client      = None

        # ── Claude (PRIMARY) ──────────────────────────────────────────────────
        if ANTHROPIC_API_KEY:
            try:
                import anthropic
                self._anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
                self._providers.append(f"claude({CLAUDE_MODEL})")
                logger.info("✅ Claude client ready: %s", CLAUDE_MODEL)
            except ImportError:
                logger.warning("anthropic package not installed — Claude disabled.")
            except Exception as e:
                logger.warning("Claude init failed: %s", e)

        # ── Groq (FALLBACK) ───────────────────────────────────────────────────
        if GROQ_API_KEY:
            try:
                from groq import Groq
                self._groq_client = Groq(api_key=GROQ_API_KEY)
                self._providers.append(f"groq({GROQ_MODEL})")
                logger.info("✅ Groq client ready: %s", GROQ_MODEL)
            except ImportError:
                logger.warning("groq package not installed — Groq disabled.")
            except Exception as e:
                logger.warning("Groq init failed: %s", e)

        # ── Offline fallback ──────────────────────────────────────────────────
        self._providers.append("fallback(N/A)")

        logger.info(
            "🚀 LLMClient ready with %d providers: %s",
            len(self._providers), self._providers
        )

    @property
    def active_provider(self) -> str:
        return self._providers[0] if self._providers else "fallback"

    # ── Public API ─────────────────────────────────────────────────────────────

    def complete(self, system_prompt: str, user_prompt: str,
                 max_tokens: int = MAX_TOKENS) -> str:
        """
        Structured call (system + user). Returns raw text.
        Used by question_classifier and other callers that split prompts.
        """
        if self._anthropic_client:
            result = self._try_claude(system_prompt, user_prompt, max_tokens)
            if result is not None:
                return result
        if self._groq_client:
            result = self._try_groq(system_prompt, user_prompt, max_tokens)
            if result is not None:
                return result
        logger.warning("All LLM providers failed — using offline heuristic.")
        return self._offline_fallback(user_prompt)

    def generate(self, prompt: str, max_tokens: int = MAX_TOKENS) -> LLMResponse:
        """
        Single-prompt call. Returns LLMResponse(text, provider, model, latency_ms).
        Used by llm_evaluator.py and question_classifier.py.
        """
        t0 = time.time()

        # Try Claude first
        if self._anthropic_client:
            text = self._try_claude("You are an expert university exam grader.", prompt, max_tokens)
            if text is not None:
                return LLMResponse(
                    text=text,
                    provider="claude",
                    model=CLAUDE_MODEL,
                    latency_ms=round((time.time() - t0) * 1000, 1),
                )

        # Try Groq
        if self._groq_client:
            text = self._try_groq("You are an expert university exam grader.", prompt, max_tokens)
            if text is not None:
                return LLMResponse(
                    text=text,
                    provider="groq",
                    model=GROQ_MODEL,
                    latency_ms=round((time.time() - t0) * 1000, 1),
                )

        # Offline fallback
        logger.warning("All LLM providers failed — offline fallback.")
        return LLMResponse(
            text=self._offline_fallback(prompt),
            provider="fallback",
            model="N/A",
            latency_ms=round((time.time() - t0) * 1000, 1),
        )

    def generate_json(self, prompt: str, max_tokens: int = MAX_TOKENS) -> Dict[str, Any]:
        """
        generate() + JSON parsing with markdown fence stripping.
        Falls back to a safe zero-score dict on parse errors.
        """
        response = self.generate(prompt, max_tokens=max_tokens)
        raw = response.text
        cleaned = re.sub(r"```json|```|`", "", raw).strip()
        try:
            m = re.search(r'\{.*\}', cleaned, re.DOTALL)
            return json.loads(m.group() if m else cleaned)
        except Exception as e:
            logger.warning("generate_json: JSON parse failed (%s). Raw: %.300s", e, raw)
            return {
                "score": 0.0, "confidence": 0.3,
                "strengths": [], "missing_concepts": [],
                "feedback": "Evaluation response could not be parsed. Please try again.",
            }

    # ── Provider internals ─────────────────────────────────────────────────────

    def _try_claude(self, system: str, user: str, max_tokens: int,
                    max_attempts: int = 3) -> Optional[str]:
        """Try Claude with exponential back-off on rate-limit errors."""
        for attempt in range(1, max_attempts + 1):
            try:
                t0 = time.time()
                response = self._anthropic_client.messages.create(
                    model=CLAUDE_MODEL,
                    max_tokens=max_tokens,
                    temperature=TEMPERATURE,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                elapsed = (time.time() - t0) * 1000
                text = response.content[0].text.strip()
                i_tok = response.usage.input_tokens
                o_tok = response.usage.output_tokens
                cost  = (i_tok * 0.00000025) + (o_tok * 0.00000125)
                logger.info("✅ claude/%s in %.0fms (cost: $%.6f)", CLAUDE_MODEL, elapsed, cost)
                return text
            except Exception as e:
                err_str = str(e).lower()
                if "rate" in err_str or "429" in err_str:
                    wait = 2 ** attempt
                    logger.warning("Claude rate-limit (attempt %d/%d) — retry in %ds", attempt, max_attempts, wait)
                    time.sleep(wait)
                else:
                    logger.warning("Claude attempt %d/%d failed: %s", attempt, max_attempts, e)
                    if attempt >= max_attempts:
                        return None
                    time.sleep(1)
        return None

    def _try_groq(self, system: str, user: str, max_tokens: int,
                  max_attempts: int = 3) -> Optional[str]:
        """Try Groq with linear retry."""
        for attempt in range(1, max_attempts + 1):
            try:
                t0 = time.time()
                response = self._groq_client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ],
                    max_tokens=max_tokens,
                    temperature=TEMPERATURE,
                )
                elapsed = (time.time() - t0) * 1000
                text   = response.choices[0].message.content.strip()
                tokens = getattr(response.usage, "total_tokens", 0)
                cost   = tokens * 0.00000059
                logger.info("✅ groq/%s in %.0fms (cost: $%.6f)", GROQ_MODEL, elapsed, cost)
                return text
            except Exception as e:
                logger.warning("Groq attempt %d/%d failed: %s", attempt, max_attempts, e)
                if attempt < max_attempts:
                    time.sleep(1)
        return None

    def _offline_fallback(self, _prompt: str) -> str:
        return (
            '{"score": 5, "feedback": "Evaluation unavailable — all LLM providers offline. '
            'Please check API keys in Settings.", '
            '"strengths": [], "missing_concepts": [], "confidence": 0.3}'
        )

    @classmethod
    def from_env(cls) -> "LLMClient":
        return cls()


# ── Module-level singleton ─────────────────────────────────────────────────────
_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client