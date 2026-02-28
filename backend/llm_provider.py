"""
IntelliGrade-H - Multi-Provider LLM Client (v2)
=================================================
Priority order:
  1. Claude (Anthropic) — best grading quality, structured JSON
  2. Gemini 1.5 Flash   — fast, reliable
  3. Groq / Llama       — free tier, very fast
  4. Ollama (local)     — fully offline fallback (llama3, mistral, etc.)
  5. Rule-based         — ultimate fallback, no API needed

Set in .env:
  ANTHROPIC_API_KEY=...
  GEMINI_API_KEY=...
  GROQ_API_KEY=...
  OLLAMA_BASE_URL=http://localhost:11434   (optional, for offline mode)
  OLLAMA_MODEL=llama3                      (default: llama3)
  LLM_PROVIDER=auto                        (auto | claude | gemini | groq | ollama)
"""

import os
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Claude Provider (Anthropic) — PRIMARY
# ─────────────────────────────────────────────────────────────────────────────

class ClaudeProvider:
    DEFAULT_MODEL = "claude-haiku-4-5-20251001"  # Fast, affordable, excellent at grading

    def __init__(self, api_key: str, model: Optional[str] = None):
        self.api_key = api_key
        self.model = model or os.getenv("CLAUDE_MODEL", self.DEFAULT_MODEL)
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)
            logger.info("✅ Claude client ready: %s", self.model)
        return self._client

    def generate(self, prompt: str) -> LLMResponse:
        start = time.time()
        client = self._get_client()

        message = client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=(
                "You are an expert university professor evaluating student handwritten answers. "
                "Always respond with valid JSON only. No markdown fences, no extra text."
            ),
            messages=[{"role": "user", "content": prompt}],
        )

        text = message.content[0].text
        latency = (time.time() - start) * 1000

        return LLMResponse(
            text=text,
            provider="claude",
            model=self.model,
            prompt_tokens=message.usage.input_tokens,
            completion_tokens=message.usage.output_tokens,
            latency_ms=round(latency, 2),
        )

    def is_available(self) -> bool:
        return bool(self.api_key) and self.api_key not in ("", "your_anthropic_api_key_here")


# ─────────────────────────────────────────────────────────────────────────────
# Gemini Provider
# ─────────────────────────────────────────────────────────────────────────────

class GeminiProvider:
    DEFAULT_MODEL = "gemini-1.5-flash"

    def __init__(self, api_key: str, model: Optional[str] = None):
        self.api_key = api_key
        self.model = model or os.getenv("GEMINI_MODEL", self.DEFAULT_MODEL)
        self._client = None

    def _get_client(self):
        if self._client is None:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._client = genai.GenerativeModel(
                model_name=self.model,
                generation_config={"temperature": 0.2, "top_p": 0.95, "max_output_tokens": 2048},
            )
            logger.info("✅ Gemini client ready: %s", self.model)
        return self._client

    def generate(self, prompt: str) -> LLMResponse:
        start = time.time()
        client = self._get_client()
        response = client.generate_content(prompt)
        text = response.text if hasattr(response, "text") else str(response)
        latency = (time.time() - start) * 1000
        return LLMResponse(text=text, provider="gemini", model=self.model, latency_ms=round(latency, 2))

    def is_available(self) -> bool:
        return bool(self.api_key) and self.api_key not in ("", "your_gemini_api_key_here")


# ─────────────────────────────────────────────────────────────────────────────
# Groq Provider
# ─────────────────────────────────────────────────────────────────────────────

class GroqProvider:
    DEFAULT_MODEL = "llama-3.3-70b-versatile"

    def __init__(self, api_key: str, model: Optional[str] = None):
        self.api_key = api_key
        self.model = model or os.getenv("GROQ_MODEL", self.DEFAULT_MODEL)
        self._client = None

    def _get_client(self):
        if self._client is None:
            from groq import Groq
            self._client = Groq(api_key=self.api_key)
            logger.info("✅ Groq client ready: %s", self.model)
        return self._client

    def generate(self, prompt: str) -> LLMResponse:
        start = time.time()
        client = self._get_client()
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert university professor. Always respond with valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=2048,
            response_format={"type": "json_object"},
        )
        text = resp.choices[0].message.content
        latency = (time.time() - start) * 1000
        return LLMResponse(
            text=text,
            provider="groq",
            model=self.model,
            prompt_tokens=resp.usage.prompt_tokens if resp.usage else 0,
            completion_tokens=resp.usage.completion_tokens if resp.usage else 0,
            latency_ms=round(latency, 2),
        )

    def is_available(self) -> bool:
        try:
            from groq import Groq
            return bool(self.api_key) and self.api_key not in ("", "your_groq_api_key_here")
        except ImportError:
            return False


# ─────────────────────────────────────────────────────────────────────────────
# Ollama Provider — FULLY OFFLINE FALLBACK
# ─────────────────────────────────────────────────────────────────────────────

class OllamaProvider:
    """
    Uses a locally-running Ollama server for 100% offline evaluation.
    Install Ollama: https://ollama.ai
    Run: ollama pull llama3
    Then: ollama serve
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3")

    def generate(self, prompt: str) -> LLMResponse:
        import urllib.request
        start = time.time()

        payload = json.dumps({
            "model": self.model,
            "prompt": (
                "You are an expert university professor evaluating student handwritten answers. "
                "Always respond with valid JSON only, no markdown.\n\n" + prompt
            ),
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": 1024},
        }).encode()

        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())

        text = data.get("response", "")
        latency = (time.time() - start) * 1000
        logger.info("✅ Ollama response (%.2fms): %s", latency, text[:80])

        return LLMResponse(
            text=text,
            provider="ollama",
            model=self.model,
            latency_ms=round(latency, 2),
        )

    def is_available(self) -> bool:
        """Check if Ollama is running locally."""
        import urllib.request
        try:
            urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=3)
            return True
        except Exception:
            return False


# ─────────────────────────────────────────────────────────────────────────────
# Rule-Based Fallback (no API, no internet)
# ─────────────────────────────────────────────────────────────────────────────

class RuleBasedFallbackProvider:
    """
    Heuristic scoring when all LLM providers are unavailable.
    Uses text length, keyword overlap, and basic NLP.
    """

    def is_available(self) -> bool:
        return True

    def generate(self, prompt: str) -> LLMResponse:
        student_answer = ""
        teacher_answer = ""
        max_marks = 10.0

        # Extract from prompt
        if "STUDENT ANSWER:" in prompt:
            student_answer = prompt.split("STUDENT ANSWER:")[-1].split("\n\nMAXIMUM")[0].strip()
        if "MODEL ANSWER" in prompt or "EXPECTED ANSWER" in prompt:
            key = "MODEL ANSWER" if "MODEL ANSWER" in prompt else "EXPECTED ANSWER"
            teacher_answer = prompt.split(key + ":")[-1].split("\n\nSTUDENT")[0].strip()
        try:
            max_marks = float(re.search(r"MAXIMUM MARKS:\s*(\d+\.?\d*)", prompt).group(1))
        except Exception:
            pass

        score = _heuristic_score(student_answer, teacher_answer, max_marks)

        result = {
            "score": round(score, 1),
            "confidence": 0.35,
            "strengths": ["Answer was provided"] if student_answer else [],
            "missing_concepts": ["Detailed AI analysis unavailable — offline mode"],
            "feedback": (
                f"⚠️ Offline fallback evaluation. Score estimated based on answer length and keyword overlap "
                f"({score:.1f}/{max_marks}). Please review manually for accuracy."
            ),
        }
        return LLMResponse(
            text=json.dumps(result),
            provider="fallback",
            model="rule-based",
            latency_ms=0.0,
        )


def _heuristic_score(student: str, teacher: str, max_marks: float) -> float:
    """Simple keyword-overlap heuristic for offline scoring."""
    if not student.strip():
        return 0.0

    student_words = set(re.findall(r'\b[a-z]{3,}\b', student.lower()))
    teacher_words = set(re.findall(r'\b[a-z]{3,}\b', teacher.lower())) if teacher else set()

    # Length score (0–40% of marks)
    target_len = max(len(teacher.split()), 50)
    length_ratio = min(len(student.split()) / target_len, 1.0)
    length_score = length_ratio * 0.4 * max_marks

    # Keyword overlap (0–60% of marks)
    if teacher_words:
        overlap = len(student_words & teacher_words) / len(teacher_words)
        keyword_score = overlap * 0.6 * max_marks
    else:
        keyword_score = length_ratio * 0.6 * max_marks

    return min(max_marks, length_score + keyword_score)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Provider Client
# ─────────────────────────────────────────────────────────────────────────────

class LLMClient:
    """
    Unified client that tries providers in priority order with fallback.
    """

    def __init__(self, providers: list):
        self._providers = providers
        self._last_used_provider = None

    @classmethod
    def from_env(cls) -> "LLMClient":
        anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
        gemini_key = os.getenv("GEMINI_API_KEY", "")
        groq_key = os.getenv("GROQ_API_KEY", "")
        preference = os.getenv("LLM_PROVIDER", "auto").lower()

        providers = []

        def add_claude():
            if ClaudeProvider(anthropic_key).is_available():
                providers.append(ClaudeProvider(anthropic_key))

        def add_gemini():
            if GeminiProvider(gemini_key).is_available():
                providers.append(GeminiProvider(gemini_key))

        def add_groq():
            if GroqProvider(groq_key).is_available():
                providers.append(GroqProvider(groq_key))

        def add_ollama():
            p = OllamaProvider()
            if p.is_available():
                providers.append(p)
                logger.info("✅ Ollama offline provider detected")

        if preference == "claude":
            add_claude(); add_gemini(); add_groq(); add_ollama()
        elif preference == "gemini":
            add_gemini(); add_claude(); add_groq(); add_ollama()
        elif preference == "groq":
            add_groq(); add_claude(); add_gemini(); add_ollama()
        elif preference == "ollama":
            add_ollama(); add_claude(); add_gemini(); add_groq()
        else:  # auto — Claude first (best grading), then others
            add_claude(); add_gemini(); add_groq(); add_ollama()

        # Always add rule-based as final safety net
        providers.append(RuleBasedFallbackProvider())

        provider_names = [
            f"{p.__class__.__name__}({getattr(p, 'model', 'N/A')})"
            for p in providers
        ]
        logger.info("🚀 LLMClient ready with %d providers: %s", len(providers), provider_names)

        return cls(providers)

    def generate(self, prompt: str) -> LLMResponse:
        for provider in self._providers:
            if not provider.is_available():
                continue
            try:
                logger.info("Trying: %s", provider.__class__.__name__)
                response = provider.generate(prompt)
                self._last_used_provider = provider
                logger.info(
                    "✅ %s/%s responded in %.2fms",
                    response.provider, response.model, response.latency_ms,
                )
                return response
            except Exception as e:
                logger.warning("⚠️ %s failed: %s — trying next", provider.__class__.__name__, str(e)[:120])

        # Should never reach here since RuleBasedFallbackProvider always succeeds
        return LLMResponse(
            text=json.dumps({"score": 0, "confidence": 0, "strengths": [], "missing_concepts": [], "feedback": "All providers failed."}),
            provider="none",
            model="none",
        )

    def generate_json(self, prompt: str, max_retries: int = 2) -> dict:
        """Generate and parse JSON with retries."""
        required_fields = ["score", "confidence", "strengths", "missing_concepts", "feedback"]
        defaults = {
            "score": 0.0, "confidence": 0.5,
            "strengths": [], "missing_concepts": [],
            "feedback": "Evaluation incomplete.",
        }

        for attempt in range(max_retries + 1):
            try:
                response = self.generate(prompt)
                cleaned = re.sub(r"```json|```|`", "", response.text).strip()

                json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                else:
                    data = json.loads(cleaned)

                for field in required_fields:
                    if field not in data:
                        data[field] = defaults[field]

                return data
            except Exception as e:
                if attempt == max_retries:
                    logger.error("JSON parse failed after %d retries: %s", max_retries, e)
                    return defaults
                logger.warning("JSON parse attempt %d failed: %s", attempt + 1, e)

        return defaults

    @property
    def active_provider(self) -> str:
        for p in self._providers:
            if p.is_available():
                return f"{p.__class__.__name__}({getattr(p, 'model', 'N/A')})"
        return "none"