"""
IntelliGrade-H — LLM Provider (v6 — Groq + Claude)
====================================================
Primary  : Groq  (llama-3.3-70b-versatile) — default
Fallback : Anthropic Claude (claude-haiku-4-5-20251001) — if ANTHROPIC_API_KEY set
           Offline heuristic — if both APIs unavailable

Set in .env:
  LLM_PROVIDER=groq          # or "claude" to flip primary/fallback
  GROQ_API_KEY=gsk_...
  ANTHROPIC_API_KEY=sk-ant-...
  CLAUDE_MODEL=claude-haiku-4-5-20251001
  GROQ_MODEL=llama-3.3-70b-versatile
  LLM_MAX_TOKENS=6000
  LLM_TEMPERATURE=0.1

generate()      → LLMResponse(text, provider, model, latency_ms)
generate_json() → dict
complete()      → str (system + user prompt)
"""

import os
import re
import json
import time
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# ── Provider config (read directly from env so provider works standalone) ──────
LLM_PROVIDER      = os.getenv("LLM_PROVIDER", "groq").lower().strip()
GROQ_API_KEY      = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL        = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL      = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001")

MAX_TOKENS  = int(os.getenv("LLM_MAX_TOKENS", "6000"))   # raised from 1500
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))


@dataclass
class LLMResponse:
    """Standardised response returned by LLMClient.generate()."""
    text: str
    provider: str  = "groq"
    model: str     = GROQ_MODEL
    latency_ms: float = 0.0


class LLMClient:
    """
    Multi-provider LLM client.
    Provider order is determined by LLM_PROVIDER env var:
      - "groq"   → tries Groq first, then Claude, then offline
      - "claude" → tries Claude first, then Groq, then offline
    """

    def __init__(self):
        self._groq_client    = None
        self._claude_client  = None
        self._providers: list[str] = []

        self._init_groq()
        self._init_claude()

        # Always append offline as final fallback
        self._providers.append("offline(fallback)")
        logger.info("🚀 LLMClient ready. Providers: %s", self._providers)

    # ── Initialisation ─────────────────────────────────────────────────────────

    def _init_groq(self):
        if not GROQ_API_KEY:
            logger.warning("GROQ_API_KEY not set — Groq unavailable.")
            return
        try:
            from groq import Groq
            try:
                import httpx
                self._groq_client = Groq(
                    api_key=GROQ_API_KEY,
                    http_client=httpx.Client(),
                )
            except TypeError:
                self._groq_client = Groq(api_key=GROQ_API_KEY)
            self._providers.append(f"groq({GROQ_MODEL})")
            logger.info("✅ Groq client ready: %s", GROQ_MODEL)
        except ImportError:
            logger.warning("groq package not installed — run: pip install groq")
        except Exception as e:
            logger.warning("Groq init failed: %s", e)

    def _init_claude(self):
        if not ANTHROPIC_API_KEY:
            return   # silently skip — key is optional
        try:
            import anthropic
            self._claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            self._providers.append(f"claude({CLAUDE_MODEL})")
            logger.info("✅ Claude client ready: %s", CLAUDE_MODEL)
        except ImportError:
            logger.warning(
                "anthropic package not installed — run: pip install anthropic. "
                "Claude fallback disabled."
            )
        except Exception as e:
            logger.warning("Claude init failed: %s", e)

    # ── Provider ordering ──────────────────────────────────────────────────────

    def _ordered_providers(self) -> list[str]:
        """Return [primary, fallback] based on LLM_PROVIDER setting."""
        if LLM_PROVIDER == "claude":
            return ["claude", "groq"]
        return ["groq", "claude"]

    # ── Public API ─────────────────────────────────────────────────────────────

    def complete(self, system_prompt: str, user_prompt: str,
                 max_tokens: int = MAX_TOKENS) -> str:
        """Structured call (system + user). Returns raw text string."""
        for provider in self._ordered_providers():
            result = self._call(provider, system_prompt, user_prompt, max_tokens)
            if result is not None:
                return result
        logger.warning("All LLM providers failed — using offline fallback.")
        return self._offline_fallback(user_prompt)

    def generate(self, prompt: str, max_tokens: int = MAX_TOKENS) -> LLMResponse:
        """Single-prompt call. Returns LLMResponse."""
        t0     = time.time()
        system = "You are an expert university exam grader."

        for provider in self._ordered_providers():
            text = self._call(provider, system, prompt, max_tokens)
            if text is not None:
                return LLMResponse(
                    text=text,
                    provider=provider,
                    model=GROQ_MODEL if provider == "groq" else CLAUDE_MODEL,
                    latency_ms=round((time.time() - t0) * 1000, 1),
                )

        logger.warning("All LLM providers failed — returning offline fallback.")
        return LLMResponse(
            text=self._offline_fallback(prompt),
            provider="offline",
            model="N/A",
            latency_ms=round((time.time() - t0) * 1000, 1),
        )

    def generate_json(self, prompt: str, max_tokens: int = MAX_TOKENS) -> Dict[str, Any]:
        """generate() + JSON parsing. Falls back to safe zero-score dict."""
        response = self.generate(prompt, max_tokens=max_tokens)
        raw      = response.text
        cleaned  = re.sub(r"```json|```|`", "", raw).strip()
        try:
            m = re.search(r'\{.*\}', cleaned, re.DOTALL)
            return json.loads(m.group() if m else cleaned)
        except Exception as e:
            logger.warning(
                "generate_json: JSON parse failed (%s). Raw (first 300 chars): %.300s",
                e, raw,
            )
            return {
                "score": 0.0,
                "confidence": 0.3,
                "strengths": [],
                "missing_concepts": [],
                "feedback": (
                    "Evaluation response could not be parsed. "
                    "Check your API keys and try again."
                ),
            }

    # ── Provider dispatch ──────────────────────────────────────────────────────

    def _call(self, provider: str, system: str, user: str,
              max_tokens: int) -> Optional[str]:
        """Try one provider. Returns text or None."""
        if provider == "groq" and self._groq_client:
            return self._try_groq(system, user, max_tokens)
        if provider == "claude" and self._claude_client:
            return self._try_claude(system, user, max_tokens)
        return None

    # ── Groq ───────────────────────────────────────────────────────────────────

    def _try_groq(self, system: str, user: str, max_tokens: int,
                  max_attempts: int = 3) -> Optional[str]:
        for attempt in range(1, max_attempts + 1):
            try:
                t0       = time.time()
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
                text    = response.choices[0].message.content.strip()
                tokens  = getattr(response.usage, "total_tokens", 0)
                cost    = tokens * 0.00000059   # llama-3.3-70b approx
                logger.info(
                    "✅ groq/%s in %.0fms | %d tokens | ~$%.6f",
                    GROQ_MODEL, elapsed, tokens, cost,
                )
                return text
            except Exception as e:
                err  = str(e).lower()
                wait = (2 ** attempt) if ("rate" in err or "429" in err) else attempt
                logger.warning(
                    "Groq attempt %d/%d failed: %s — retry in %ds",
                    attempt, max_attempts, e, wait,
                )
                if attempt < max_attempts:
                    time.sleep(wait)
        logger.error("Groq failed after %d attempts — trying next provider.", max_attempts)
        return None

    # ── Claude ─────────────────────────────────────────────────────────────────

    def _try_claude(self, system: str, user: str, max_tokens: int,
                    max_attempts: int = 2) -> Optional[str]:
        for attempt in range(1, max_attempts + 1):
            try:
                t0       = time.time()
                response = self._claude_client.messages.create(
                    model=CLAUDE_MODEL,
                    max_tokens=min(max_tokens, 4096),  # Claude max
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                elapsed = (time.time() - t0) * 1000
                text    = response.content[0].text.strip()
                tokens  = getattr(response.usage, "input_tokens", 0) + \
                          getattr(response.usage, "output_tokens", 0)
                logger.info(
                    "✅ claude/%s in %.0fms | %d tokens",
                    CLAUDE_MODEL, elapsed, tokens,
                )
                return text
            except Exception as e:
                err  = str(e).lower()
                wait = (2 ** attempt) if ("rate" in err or "529" in err) else attempt
                logger.warning(
                    "Claude attempt %d/%d failed: %s — retry in %ds",
                    attempt, max_attempts, e, wait,
                )
                if attempt < max_attempts:
                    time.sleep(wait)
        logger.error("Claude failed after %d attempts.", max_attempts)
        return None

    # ── Offline fallback ───────────────────────────────────────────────────────

    @staticmethod
    def _offline_fallback(_prompt: str) -> str:
        return (
            '{"score": 5.0, '
            '"feedback": "Evaluation unavailable — all LLM providers offline or '
            'API keys missing. Check GROQ_API_KEY / ANTHROPIC_API_KEY in .env.", '
            '"strengths": [], "missing_concepts": [], "confidence": 0.3}'
        )

    @property
    def active_provider(self) -> str:
        return self._ordered_providers()[0] if self._ordered_providers() else "offline"

    @classmethod
    def from_env(cls) -> "LLMClient":
        return cls()


# ── Module-level singleton ─────────────────────────────────────────────────────

_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Return the shared LLMClient singleton (created on first call)."""
    global _client
    if _client is None:
        _client = LLMClient()
    return _client