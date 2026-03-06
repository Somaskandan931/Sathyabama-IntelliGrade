"""
IntelliGrade-H - Multi-Provider LLM Client (v4)
=================================================
Complete implementation with:
- Robust error handling and retries
- Provider health checking
- Cost tracking and explainability
- Graceful degradation with rule-based fallback
- Support for all providers: Gemini, Claude, Groq, Ollama
"""

import os
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider"""
    text: str
    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    cost_estimate: float = 0.0
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()


@dataclass
class ProviderHealth:
    """Health status of an LLM provider"""
    available: bool
    latency_ms: float = 0.0
    error: Optional[str] = None
    model: Optional[str] = None


# ============================================================================
# Provider Base Class
# ============================================================================

class BaseLLMProvider:
    """Base class for all LLM providers"""

    def __init__(self, name: str):
        self.name = name
        self._last_error = None
        self._success_count = 0
        self._error_count = 0

    def is_available(self) -> bool:
        """Check if provider is available"""
        raise NotImplementedError

    def generate(self, prompt: str) -> LLMResponse:
        """Generate response from provider"""
        raise NotImplementedError

    def health_check(self) -> ProviderHealth:
        """Quick health check"""
        start = time.time()
        try:
            available = self.is_available()
            latency = (time.time() - start) * 1000
            return ProviderHealth(
                available=available,
                latency_ms=round(latency, 2),
                model=getattr(self, 'model', None)
            )
        except Exception as e:
            return ProviderHealth(
                available=False,
                error=str(e)
            )

    def get_stats(self) -> dict:
        """Get provider statistics"""
        return {
            "name": self.name,
            "success_count": self._success_count,
            "error_count": self._error_count,
            "last_error": str(self._last_error) if self._last_error else None
        }


# ============================================================================
# Gemini Provider
# ============================================================================

class GeminiProvider(BaseLLMProvider):
    """Google Gemini API provider"""

    DEFAULT_MODEL = "gemini-1.5-flash"
    MODELS = {
        "gemini-1.5-flash": {"input_cost": 0.000375, "output_cost": 0.00125},
        "gemini-1.5-pro": {"input_cost": 0.0025, "output_cost": 0.0075},
    }

    def __init__(self, api_key: str, model: Optional[str] = None):
        super().__init__("gemini")
        self.api_key = api_key
        self.model = model or os.getenv("GEMINI_MODEL", self.DEFAULT_MODEL)
        self._client = None
        self._cost_config = self.MODELS.get(self.model, self.MODELS[self.DEFAULT_MODEL])

    def _get_client(self):
        """Lazy initialize Gemini client"""
        if self._client is None and self._validate_api_key():
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._client = genai.GenerativeModel(
                    model_name=self.model,
                    generation_config={
                        "temperature": 0.2,
                        "top_p": 0.95,
                        "max_output_tokens": 2048,
                    },
                )
                logger.info("✅ Gemini client ready: %s", self.model)
            except ImportError:
                logger.error("google-generativeai not installed")
            except Exception as e:
                logger.error("Failed to initialize Gemini: %s", e)
                self._last_error = e
        return self._client

    def _validate_api_key(self) -> bool:
        """Validate API key format"""
        if not self.api_key or self.api_key in ("", "your_gemini_api_key_here"):
            return False
        # Gemini API keys typically start with "AIza"
        if not self.api_key.startswith("AIza"):
            logger.warning("Gemini API key should start with 'AIza'")
        return True

    def is_available(self) -> bool:
        """Check if Gemini is available"""
        if not self._validate_api_key():
            return False

        client = self._get_client()
        if not client:
            return False

        try:
            # Simple test call
            response = client.generate_content("test")
            return True
        except Exception as e:
            self._last_error = e
            return False

    def generate(self, prompt: str) -> LLMResponse:
        """Generate response using Gemini"""
        start = time.time()
        client = self._get_client()

        if not client:
            raise Exception("Gemini client not available")

        try:
            response = client.generate_content(prompt)

            # Extract text from response
            if hasattr(response, 'text'):
                text = response.text
            elif hasattr(response, 'parts'):
                text = ''.join(part.text for part in response.parts)
            else:
                text = str(response)

            latency = (time.time() - start) * 1000

            # Estimate tokens (Gemini doesn't return counts)
            words = len(text.split())
            estimated_prompt_tokens = len(prompt.split()) * 1.3
            estimated_completion_tokens = words * 1.3

            # Calculate cost
            input_cost = (estimated_prompt_tokens / 1000) * self._cost_config["input_cost"]
            output_cost = (estimated_completion_tokens / 1000) * self._cost_config["output_cost"]

            self._success_count += 1

            return LLMResponse(
                text=text,
                provider="gemini",
                model=self.model,
                prompt_tokens=int(estimated_prompt_tokens),
                completion_tokens=int(estimated_completion_tokens),
                latency_ms=round(latency, 2),
                cost_estimate=round(input_cost + output_cost, 6),
            )

        except Exception as e:
            self._error_count += 1
            self._last_error = e
            logger.error("Gemini API error: %s", e)

            # Check for specific error types
            error_str = str(e).lower()
            if "api key" in error_str or "invalid" in error_str:
                raise Exception("Invalid Gemini API key")
            elif "quota" in error_str or "limit" in error_str:
                raise Exception("Gemini quota exceeded")
            else:
                raise


# ============================================================================
# Claude Provider (Anthropic)
# ============================================================================

class ClaudeProvider(BaseLLMProvider):
    """Anthropic Claude API provider"""

    DEFAULT_MODEL = "claude-3-haiku-20240307"
    MODELS = {
        "claude-3-haiku-20240307": {"input_cost": 0.00025, "output_cost": 0.00125},
        "claude-3-sonnet-20240229": {"input_cost": 0.003, "output_cost": 0.015},
        "claude-3-opus-20240229": {"input_cost": 0.015, "output_cost": 0.075},
    }

    def __init__(self, api_key: str, model: Optional[str] = None):
        super().__init__("claude")
        self.api_key = api_key
        self.model = model or os.getenv("CLAUDE_MODEL", self.DEFAULT_MODEL)
        self._client = None
        self._cost_config = self.MODELS.get(self.model, self.MODELS[self.DEFAULT_MODEL])

    def _get_client(self):
        """Lazy initialize Claude client"""
        if self._client is None and self._validate_api_key():
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
                logger.info("✅ Claude client ready: %s", self.model)
            except ImportError:
                logger.error("anthropic not installed")
            except Exception as e:
                logger.error("Failed to initialize Claude: %s", e)
                self._last_error = e
        return self._client

    def _validate_api_key(self) -> bool:
        """Validate API key format"""
        if not self.api_key or self.api_key in ("", "your_anthropic_api_key_here"):
            return False
        # Claude keys typically start with "sk-ant"
        if not self.api_key.startswith("sk-ant"):
            logger.warning("Claude API key should start with 'sk-ant'")
        return True

    def is_available(self) -> bool:
        """Check if Claude is available"""
        if not self._validate_api_key():
            return False

        client = self._get_client()
        if not client:
            return False

        try:
            # Simple test call
            client.messages.create(
                model=self.model,
                max_tokens=10,
                messages=[{"role": "user", "content": "test"}]
            )
            return True
        except Exception as e:
            self._last_error = e
            return False

    def generate(self, prompt: str) -> LLMResponse:
        """Generate response using Claude"""
        start = time.time()
        client = self._get_client()

        if not client:
            raise Exception("Claude client not available")

        try:
            message = client.messages.create(
                model=self.model,
                max_tokens=2048,
                temperature=0.2,
                system=(
                    "You are an expert university professor evaluating student handwritten answers. "
                    "Always respond with valid JSON only. Include detailed explanations for your scoring."
                ),
                messages=[{"role": "user", "content": prompt}],
            )

            text = message.content[0].text
            latency = (time.time() - start) * 1000

            # Calculate cost
            input_cost = (message.usage.input_tokens / 1000) * self._cost_config["input_cost"]
            output_cost = (message.usage.output_tokens / 1000) * self._cost_config["output_cost"]

            self._success_count += 1

            return LLMResponse(
                text=text,
                provider="claude",
                model=self.model,
                prompt_tokens=message.usage.input_tokens,
                completion_tokens=message.usage.output_tokens,
                latency_ms=round(latency, 2),
                cost_estimate=round(input_cost + output_cost, 6),
            )

        except Exception as e:
            self._error_count += 1
            self._last_error = e
            logger.error("Claude API error: %s", e)

            error_str = str(e).lower()
            if "credit balance" in error_str:
                raise Exception("Claude credit balance exhausted")
            elif "invalid" in error_str and "api key" in error_str:
                raise Exception("Invalid Claude API key")
            else:
                raise


# ============================================================================
# Groq Provider (Llama)
# ============================================================================

class GroqProvider(BaseLLMProvider):
    """Groq API provider (Llama models)"""

    DEFAULT_MODEL = "llama-3.3-70b-versatile"
    MODELS = {
        "llama-3.3-70b-versatile": {"cost": 0.0001},
        "mixtral-8x7b-32768": {"cost": 0.00008},
        "gemma-7b-it": {"cost": 0.00005},
    }

    def __init__(self, api_key: str, model: Optional[str] = None):
        super().__init__("groq")
        self.api_key = api_key
        self.model = model or os.getenv("GROQ_MODEL", self.DEFAULT_MODEL)
        self._client = None
        self._cost_config = self.MODELS.get(self.model, self.MODELS[self.DEFAULT_MODEL])

    def _get_client(self):
        """Lazy initialize Groq client"""
        if self._client is None and self._validate_api_key():
            try:
                from groq import Groq
                self._client = Groq(api_key=self.api_key)
                logger.info("✅ Groq client ready: %s", self.model)
            except ImportError:
                logger.error("groq not installed")
            except Exception as e:
                logger.error("Failed to initialize Groq: %s", e)
                self._last_error = e
        return self._client

    def _validate_api_key(self) -> bool:
        """Validate API key format"""
        if not self.api_key or self.api_key in ("", "your_groq_api_key_here"):
            return False
        # Groq keys typically start with "gsk_"
        if not self.api_key.startswith("gsk_"):
            logger.warning("Groq API key should start with 'gsk_'")
        return True

    def is_available(self) -> bool:
        """Check if Groq is available"""
        if not self._validate_api_key():
            return False

        try:
            from groq import Groq
            return True
        except ImportError:
            return False

    def generate(self, prompt: str) -> LLMResponse:
        """Generate response using Groq"""
        start = time.time()
        client = self._get_client()

        if not client:
            raise Exception("Groq client not available")

        try:
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

            # Calculate cost
            total_tokens = (resp.usage.prompt_tokens + resp.usage.completion_tokens) if resp.usage else 0
            cost = (total_tokens / 1000) * self._cost_config["cost"]

            self._success_count += 1

            return LLMResponse(
                text=text,
                provider="groq",
                model=self.model,
                prompt_tokens=resp.usage.prompt_tokens if resp.usage else 0,
                completion_tokens=resp.usage.completion_tokens if resp.usage else 0,
                latency_ms=round(latency, 2),
                cost_estimate=round(cost, 6),
            )

        except Exception as e:
            self._error_count += 1
            self._last_error = e
            logger.error("Groq API error: %s", e)
            raise


# ============================================================================
# Ollama Provider (Local)
# ============================================================================

class OllamaProvider(BaseLLMProvider):
    """Local Ollama provider (offline)"""

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None):
        super().__init__("ollama")
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3")
        logger.info("✅ Ollama provider configured: %s with model %s", self.base_url, self.model)

    def is_available(self) -> bool:
        """Check if Ollama is running locally"""
        try:
            req = urllib.request.Request(
                f"{self.base_url}/api/tags",
                method="GET",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode())
                # Check if our model is available
                models = [m["name"] for m in data.get("models", [])]
                if self.model not in models:
                    logger.warning(f"Model {self.model} not found in Ollama. Available: {models}")
                return True
        except Exception as e:
            logger.debug("Ollama not available: %s", e)
            return False

    def generate(self, prompt: str) -> LLMResponse:
        """Generate response using local Ollama"""
        start = time.time()

        # Enhanced prompt for better explainability
        enhanced_prompt = (
            "You are an expert university professor evaluating student handwritten answers. "
            "Provide detailed reasoning for your evaluation in JSON format.\n\n" + prompt
        )

        payload = json.dumps({
            "model": self.model,
            "prompt": enhanced_prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_predict": 2048,
                "top_k": 40,
                "top_p": 0.9,
            },
        }).encode()

        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode())

            text = data.get("response", "")
            latency = (time.time() - start) * 1000

            # Estimate tokens
            estimated_tokens = len(text.split()) * 1.3

            self._success_count += 1

            return LLMResponse(
                text=text,
                provider="ollama",
                model=self.model,
                prompt_tokens=int(estimated_tokens * 0.7),
                completion_tokens=int(estimated_tokens * 0.3),
                latency_ms=round(latency, 2),
                cost_estimate=0.0,  # Free for local models
            )

        except urllib.error.URLError as e:
            self._error_count += 1
            self._last_error = e
            raise Exception(f"Ollama connection failed: {e}")
        except Exception as e:
            self._error_count += 1
            self._last_error = e
            raise


# ============================================================================
# Rule-Based Fallback Provider
# ============================================================================

class RuleBasedFallbackProvider(BaseLLMProvider):
    """
    Heuristic fallback when all LLM providers are unavailable.
    Uses keyword matching and length analysis.
    """

    def __init__(self):
        super().__init__("fallback")

    def is_available(self) -> bool:
        return True  # Always available

    def _extract_from_prompt(self, prompt: str) -> dict:
        """Extract question, answers, and max marks from prompt"""
        result = {
            "student_answer": "",
            "teacher_answer": "",
            "max_marks": 10.0,
            "question": "",
        }

        # Extract question
        if "QUESTION:" in prompt:
            q_part = prompt.split("QUESTION:")[-1]
            if "MODEL ANSWER:" in q_part:
                result["question"] = q_part.split("MODEL ANSWER:")[0].strip()
            elif "EXPECTED ANSWER:" in q_part:
                result["question"] = q_part.split("EXPECTED ANSWER:")[0].strip()

        # Extract model/teacher answer
        for key in ["MODEL ANSWER:", "EXPECTED ANSWER:"]:
            if key in prompt:
                ans_part = prompt.split(key)[-1]
                if "STUDENT ANSWER:" in ans_part:
                    result["teacher_answer"] = ans_part.split("STUDENT ANSWER:")[0].strip()
                    break

        # Extract student answer
        if "STUDENT ANSWER:" in prompt:
            student_part = prompt.split("STUDENT ANSWER:")[-1]
            for stop in ["\n\nMAXIMUM", "\n\nRUBRIC", "\n\nGRADING"]:
                if stop in student_part:
                    student_part = student_part.split(stop)[0]
            result["student_answer"] = student_part.strip()

        # Extract max marks
        max_match = re.search(r"MAXIMUM MARKS:\s*(\d+\.?\d*)", prompt)
        if max_match:
            result["max_marks"] = float(max_match.group(1))

        return result

    def _compute_keywords(self, text: str) -> set:
        """Extract meaningful keywords from text"""
        # Remove common words and punctuation
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        words = text.split()

        # Filter out common stopwords
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                     'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
                     'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'should', 'could', 'may', 'might', 'must', 'can'}

        return {w for w in words if len(w) > 2 and w not in stopwords}

    def generate(self, prompt: str) -> LLMResponse:
        """Generate heuristic evaluation"""
        extracted = self._extract_from_prompt(prompt)

        student = extracted["student_answer"]
        teacher = extracted["teacher_answer"]
        max_marks = extracted["max_marks"]

        if not student.strip():
            result = {
                "score": 0.0,
                "confidence": 0.3,
                "strengths": [],
                "missing_concepts": ["No answer provided"],
                "feedback": "No student answer was provided for evaluation.",
                "explanation": {
                    "score_rationale": "Zero score because answer is empty",
                    "key_factors": ["Empty submission"],
                    "comparison_with_model": "No comparison possible",
                    "improvement_suggestions": ["Please provide an answer"],
                    "confidence_factors": {"answer_present": False}
                }
            }
        else:
            # Extract keywords
            student_keywords = self._compute_keywords(student)
            teacher_keywords = self._compute_keywords(teacher) if teacher else set()

            # Length score (0-40%)
            teacher_len = len(teacher.split()) if teacher else 100
            student_len = len(student.split())
            length_ratio = min(student_len / max(teacher_len, 50), 1.0)
            length_score = length_ratio * 0.4 * max_marks

            # Keyword overlap (0-60%)
            if teacher_keywords:
                overlap = len(student_keywords & teacher_keywords)
                unique_teacher = len(teacher_keywords)
                overlap_ratio = overlap / unique_teacher if unique_teacher > 0 else 0
                keyword_score = overlap_ratio * 0.6 * max_marks

                # Identify missing concepts
                missing = list(teacher_keywords - student_keywords)[:5]
            else:
                overlap_ratio = length_ratio
                keyword_score = length_ratio * 0.6 * max_marks
                missing = []

            total_score = min(max_marks, length_score + keyword_score)

            # Generate explanation
            result = {
                "score": round(total_score, 1),
                "confidence": 0.4,
                "strengths": ["Answer provided"] if student else [],
                "missing_concepts": [f"Missing: {c}" for c in missing[:3]] if missing else ["All key concepts covered"],
                "feedback": (
                    f"⚠️ Offline evaluation. Estimated score: {total_score:.1f}/{max_marks}. "
                    f"Your answer covers {overlap_ratio*100:.1f}% of key concepts. "
                    f"Please review manually for accuracy."
                ),
                "explanation": {
                    "score_rationale": f"Score based on length ({length_ratio*100:.0f}%) and keyword overlap ({overlap_ratio*100:.0f}%)",
                    "key_factors": [
                        f"Answer length: {student_len} words",
                        f"Keyword coverage: {overlap_ratio*100:.1f}%",
                    ],
                    "comparison_with_model": f"Matched {overlap} out of {len(teacher_keywords)} key concepts",
                    "improvement_suggestions": [
                        f"Include: {', '.join(missing[:3])}" if missing else "Good coverage",
                        "Add more detail" if length_ratio < 0.8 else "Good length"
                    ],
                    "confidence_factors": {
                        "length_ratio": round(length_ratio, 3),
                        "overlap_ratio": round(overlap_ratio, 3),
                        "teacher_keywords": len(teacher_keywords),
                        "student_keywords": len(student_keywords),
                    }
                }
            }

        self._success_count += 1

        return LLMResponse(
            text=json.dumps(result),
            provider="fallback",
            model="rule-based",
            latency_ms=0.0,
            cost_estimate=0.0,
        )


# ============================================================================
# Main LLM Client
# ============================================================================

class LLMClient:
    """
    Unified client that tries providers in priority order with automatic fallback.
    Includes health monitoring, cost tracking, and explainability.
    """

    def __init__(self, providers: List[BaseLLMProvider]):
        self._providers = providers
        self._last_used_provider = None
        self._provider_stats = {p.name: p.get_stats() for p in providers}
        self._health_cache = {}
        self._health_cache_time = 0

    @classmethod
    def from_env(cls) -> "LLMClient":
        """Create client from environment variables"""
        anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
        gemini_key = os.getenv("GEMINI_API_KEY", "")
        groq_key = os.getenv("GROQ_API_KEY", "")
        preference = os.getenv("LLM_PROVIDER", "auto").lower()

        providers = []

        def add_provider(provider_class, *args):
            try:
                p = provider_class(*args)
                providers.append(p)
            except Exception as e:
                logger.warning(f"Failed to initialize {provider_class.__name__}: {e}")

        # Build provider list based on preference
        if preference == "gemini":
            add_provider(GeminiProvider, gemini_key)
            add_provider(ClaudeProvider, anthropic_key)
            add_provider(GroqProvider, groq_key)
        elif preference == "claude":
            add_provider(ClaudeProvider, anthropic_key)
            add_provider(GeminiProvider, gemini_key)
            add_provider(GroqProvider, groq_key)
        elif preference == "groq":
            add_provider(GroqProvider, groq_key)
            add_provider(ClaudeProvider, anthropic_key)
            add_provider(GeminiProvider, gemini_key)
        elif preference == "ollama":
            add_provider(OllamaProvider)
            add_provider(GroqProvider, groq_key)  # Fallback to cloud
        else:  # auto - try all in sensible order
            add_provider(ClaudeProvider, anthropic_key)  # Best quality
            add_provider(GeminiProvider, gemini_key)     # Good balance
            add_provider(GroqProvider, groq_key)         # Fast
            add_provider(OllamaProvider)                  # Offline

        # Always add rule-based as final safety net
        providers.append(RuleBasedFallbackProvider())

        provider_names = [f"{p.name}({getattr(p, 'model', 'N/A')})" for p in providers]
        logger.info("🚀 LLMClient ready with %d providers: %s", len(providers), provider_names)

        return cls(providers)

    def check_all_health(self, force_refresh: bool = False) -> Dict[str, ProviderHealth]:
        """Check health of all providers"""
        current_time = time.time()

        # Return cached health if recent (within 60 seconds)
        if not force_refresh and self._health_cache and (current_time - self._health_cache_time) < 60:
            return self._health_cache

        health = {}
        for provider in self._providers:
            try:
                health[provider.name] = provider.health_check()
            except Exception as e:
                health[provider.name] = ProviderHealth(available=False, error=str(e))

        self._health_cache = health
        self._health_cache_time = current_time
        return health

    def get_best_available_provider(self) -> Optional[BaseLLMProvider]:
        """Get the best available provider based on health and priority"""
        health = self.check_all_health()

        for provider in self._providers:
            if health.get(provider.name, ProviderHealth(available=False)).available:
                return provider

        return None

    def generate(self, prompt: str, max_retries: int = 2) -> LLMResponse:
        """Generate with automatic retry and fallback"""
        last_error = None

        for attempt in range(max_retries + 1):
            for provider in self._providers:
                provider_name = provider.name

                try:
                    # Quick health check
                    if not provider.is_available():
                        logger.debug("Skipping %s (not available)", provider_name)
                        continue

                    logger.info("Trying provider: %s (attempt %d/%d)",
                              provider_name, attempt + 1, max_retries + 1)

                    response = provider.generate(prompt)
                    self._last_used_provider = provider

                    logger.info(
                        "✅ %s/%s responded in %.2fms (cost: $%.6f)",
                        response.provider, response.model,
                        response.latency_ms, response.cost_estimate
                    )

                    return response

                except Exception as e:
                    last_error = e
                    logger.warning(
                        "⚠️ %s attempt %d failed: %s",
                        provider_name, attempt + 1, str(e)[:120]
                    )

                    # Short delay before retry
                    if attempt < max_retries:
                        time.sleep(1 * (attempt + 1))

            # If we get here, all providers failed this attempt
            if attempt < max_retries:
                logger.info("All providers failed attempt %d, retrying...", attempt + 1)

        # Ultimate fallback
        logger.error("All providers failed. Last error: %s", last_error)

        # Find the fallback provider (should always be last)
        fallback = next((p for p in self._providers if p.name == "fallback"), None)
        if fallback:
            return fallback.generate(prompt)

        # Absolute last resort
        return LLMResponse(
            text=json.dumps({
                "score": 0,
                "confidence": 0,
                "strengths": [],
                "missing_concepts": ["System error"],
                "feedback": "All evaluation providers failed. Please try again later.",
                "explanation": {
                    "score_rationale": "Error in evaluation pipeline",
                    "key_factors": ["System unavailable"],
                    "comparison_with_model": "Could not complete",
                    "improvement_suggestions": ["Please try again"],
                    "confidence_factors": {"error": True}
                }
            }),
            provider="none",
            model="none",
        )

    def generate_json(self, prompt: str, max_retries: int = 2) -> dict:
        """Generate and parse JSON with robust error recovery"""
        required_fields = ["score", "confidence", "strengths", "missing_concepts", "feedback"]
        defaults = {
            "score": 0.0,
            "confidence": 0.5,
            "strengths": [],
            "missing_concepts": [],
            "feedback": "Evaluation incomplete due to technical issues.",
            "explanation": {
                "score_rationale": "Default values due to processing error",
                "key_factors": ["Error in evaluation pipeline"],
                "comparison_with_model": "Could not complete comparison",
                "improvement_suggestions": ["Please try again"],
                "confidence_factors": {"error": True}
            }
        }

        for attempt in range(max_retries + 1):
            try:
                response = self.generate(prompt, max_retries=1)

                # Clean response - handle various formats
                cleaned = response.text.strip()
                cleaned = re.sub(r"```json|```|`", "", cleaned)
                cleaned = re.sub(r"^[^{]*", "", cleaned)  # Remove anything before first {
                cleaned = re.sub(r"[^}]*$", "", cleaned)  # Remove anything after last }

                # Find JSON object
                json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                else:
                    data = json.loads(cleaned)

                # Ensure all required fields exist
                for field in required_fields:
                    if field not in data:
                        data[field] = defaults[field]

                # Add explanation if missing
                if "explanation" not in data:
                    data["explanation"] = {
                        "score_rationale": f"Score of {data.get('score', 0)} determined by AI evaluation",
                        "key_factors": ["AI-generated evaluation"],
                        "comparison_with_model": "Based on semantic understanding",
                        "improvement_suggestions": data.get("missing_concepts", []),
                        "confidence_factors": {"confidence": data.get("confidence", 0.5)}
                    }

                return data

            except json.JSONDecodeError as e:
                logger.warning("JSON parse attempt %d failed: %s", attempt + 1, e)
                if attempt == max_retries:
                    return defaults
                time.sleep(1)

            except Exception as e:
                logger.warning("Generation attempt %d failed: %s", attempt + 1, e)
                if attempt == max_retries:
                    return defaults
                time.sleep(1)

        return defaults

    def get_stats(self) -> dict:
        """Get comprehensive client statistics"""
        stats = {
            "providers": {},
            "last_used": self._last_used_provider.name if self._last_used_provider else None,
            "health": self.check_all_health(force_refresh=True)
        }

        for provider in self._providers:
            stats["providers"][provider.name] = provider.get_stats()

        return stats

    @property
    def active_provider(self) -> str:
        """Get name of currently active provider"""
        best = self.get_best_available_provider()
        if best:
            return f"{best.name}({getattr(best, 'model', 'N/A')})"
        return "none"