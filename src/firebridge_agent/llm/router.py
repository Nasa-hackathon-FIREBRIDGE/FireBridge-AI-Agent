"""Model router: keeps answering when individual free-tier models run out of quota.

For each request the router walks the configured chain (highest priority first) and skips models
that cannot serve it: cooling down after a 429 or outage, lacking tool support, too small a
context, or (proactively, from the last rate-limit headers) without enough tokens left this
minute. Failures move to the next model; transient errors get one retry with backoff. When every
model is cooling down, it waits for the soonest one if that is short, otherwise raises.
"""

import asyncio
import json
import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..errors import AllModelsExhausted, LLMConfigurationError
from ..observability import log
from .catalog import ModelSpec, spec_for
from .groq import (
    AuthFailed,
    BadRequest,
    GroqClient,
    LLMCallError,
    ModelUnavailable,
    RateLimited,
    RequestTooLarge,
    Transient,
)
from .types import LLMRequest, LLMResponse, RateState

logger = logging.getLogger(__name__)

UNAVAILABLE_COOLDOWN_S = 300.0
BAD_REQUEST_COOLDOWN_S = 30.0


def estimate_tokens(request: LLMRequest) -> int:
    """Rough prompt size (about 3.5 characters per token) plus the requested completion."""
    chars = sum(len(json.dumps(m, ensure_ascii=False)) for m in request.messages)
    if request.tools:
        chars += len(json.dumps(request.tools))
    return int(chars / 3.5) + request.max_tokens


@dataclass
class ModelState:
    spec: ModelSpec
    cooldown_until: float = 0.0
    cooldown_reason: str | None = None
    rate: RateState | None = None
    calls: int = 0
    failures: int = 0
    rate_limited: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    last_error: str | None = None
    by_purpose: dict[str, int] = field(default_factory=dict)

    def snapshot(self, now: float) -> dict[str, Any]:
        return {
            "model": self.spec.id,
            "available": now >= self.cooldown_until,
            "cooldown_s": round(max(0.0, self.cooldown_until - now), 1),
            "cooldown_reason": self.cooldown_reason,
            "remaining_tokens_minute": self.rate.remaining_tokens if self.rate else None,
            "remaining_requests": self.rate.remaining_requests if self.rate else None,
            "calls": self.calls,
            "failures": self.failures,
            "rate_limited": self.rate_limited,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "by_purpose": dict(self.by_purpose),
            "last_error": self.last_error,
        }


class ModelRouter:
    def __init__(
        self,
        client: GroqClient,
        models: list[str],
        *,
        max_wait_s: float = 20.0,
        clock: Callable[[], float] = time.monotonic,
        sleep=asyncio.sleep,
    ):
        if not models:
            raise LLMConfigurationError("No models configured (FBA_MODELS is empty).")
        self._client = client
        self._states = [ModelState(spec_for(m)) for m in models]
        self._max_wait_s = max_wait_s
        self._clock = clock
        self._sleep = sleep
        self.fallbacks = 0
        self.proactive_switches = 0

    def status(self) -> list[dict[str, Any]]:
        now = self._clock()
        return [state.snapshot(now) for state in self._states]

    def _capable(self, state: ModelState, request: LLMRequest, size: int) -> str | None:
        """Why this model cannot take the request at all (independent of quota), or None."""
        if request.restrict_to and state.spec.id not in request.restrict_to:
            return "not requested"
        if state.spec.id in request.exclude:
            return "excluded"
        if request.user_facing and not state.spec.answers:
            return "reserved for housekeeping"
        if request.tools and not state.spec.tools:
            return "no tool support"
        if size > state.spec.context_window:
            return f"needs ~{size} tokens, context is {state.spec.context_window}"
        return None

    @staticmethod
    def _projected_tokens(state: ModelState, now: float) -> float | None:
        """Tokens available now: the per-minute budget refills continuously after the last reading."""
        rate = state.rate
        if not rate or rate.remaining_tokens is None:
            return None
        elapsed = now - rate.observed_at
        if rate.limit_tokens:
            return min(rate.limit_tokens, rate.remaining_tokens + rate.limit_tokens / 60.0 * elapsed)
        if rate.reset_tokens_s is not None and elapsed >= rate.reset_tokens_s:
            return None
        return rate.remaining_tokens

    def _short_on_tokens(self, state: ModelState, size: int, now: float) -> bool:
        projected = self._projected_tokens(state, now)
        return projected is not None and projected < size

    def _order(self, request: LLMRequest) -> list[ModelState]:
        if not request.prefer:
            return list(self._states)
        rank = {model: i for i, model in enumerate(request.prefer)}
        return sorted(self._states, key=lambda s: (rank.get(s.spec.id, len(rank)), self._states.index(s)))

    async def complete(self, request: LLMRequest) -> LLMResponse:
        size = estimate_tokens(request)
        attempts: list[dict[str, Any]] = []
        deadline = self._clock() + (self._max_wait_s if request.max_wait_s is None else request.max_wait_s)
        while True:
            now = self._clock()
            capable = [(s, self._capable(s, request, size)) for s in self._order(request)]
            eligible = [s for s, why in capable if why is None]
            if not eligible:
                reasons = {s.spec.id: why for s, why in capable}
                raise AllModelsExhausted("No configured model can serve this request.", details={"reasons": reasons})
            # Proactive switching: models that probably lack tokens this minute go last, not out;
            # if one is tried anyway, a real 429 sets its exact cooldown.
            ready = sorted(
                (s for s in eligible if now >= s.cooldown_until), key=lambda s: self._short_on_tokens(s, size, now)
            )
            if ready and ready[0] is not eligible[0] and eligible[0] in ready:
                self.proactive_switches += 1
            for state in ready:
                response = await self._try(state, request, attempts)
                if response is not None:
                    if attempts:
                        self.fallbacks += 1
                        log(
                            logger,
                            logging.INFO,
                            "model fallback",
                            purpose=request.purpose,
                            served_by=response.model,
                            attempts=len(attempts),
                        )
                    response.attempts = attempts
                    return response
            # Nothing answered: wait for the soonest model if it is close, else give up.
            now = self._clock()
            soonest = min(s.cooldown_until for s in eligible)
            wait = max(0.5, soonest - now)
            if now + wait > deadline:
                raise AllModelsExhausted(
                    "All models are rate-limited or unavailable right now.",
                    details={"attempts": attempts, "models": self.status()},
                    retry_after=round(wait, 1),
                )
            log(
                logger,
                logging.WARNING,
                "all models cooling down, waiting",
                wait_s=round(wait, 1),
                purpose=request.purpose,
            )
            await self._sleep(wait)

    async def _try(self, state: ModelState, request: LLMRequest, attempts: list[dict[str, Any]]) -> LLMResponse | None:
        model = state.spec.id
        for attempt in range(2):
            state.calls += 1
            state.by_purpose[request.purpose] = state.by_purpose.get(request.purpose, 0) + 1
            try:
                response = await self._client.chat(state.spec, request)
            except RateLimited as exc:
                state.rate_limited += 1
                self._cool(state, exc.retry_after, f"rate limit ({exc.scope})", exc)
                attempts.append(
                    {
                        "model": model,
                        "outcome": "rate_limited",
                        "detail": f"{exc.scope}, retry in {exc.retry_after:.0f}s",
                    }
                )
                return None
            except AuthFailed as exc:
                raise LLMConfigurationError(
                    "The Groq API key was rejected.", details={"model": model, "error": str(exc)}
                ) from exc
            except ModelUnavailable as exc:
                self._cool(state, UNAVAILABLE_COOLDOWN_S, "unavailable", exc)
                attempts.append({"model": model, "outcome": "unavailable", "detail": str(exc)[:160]})
                return None
            except RequestTooLarge as exc:
                state.failures += 1
                state.last_error = str(exc)[:200]
                attempts.append({"model": model, "outcome": "too_large", "detail": str(exc)[:160]})
                return None
            except Transient as exc:
                state.failures += 1
                state.last_error = str(exc)[:200]
                attempts.append({"model": model, "outcome": "transient", "detail": str(exc)[:160]})
                if attempt == 0:
                    await self._sleep(0.5 + random.random())
                    continue
                return None
            except BadRequest as exc:
                # Typically a malformed tool call by this model: let another model try.
                self._cool(state, BAD_REQUEST_COOLDOWN_S if "tool" not in str(exc).lower() else 0.0, "bad request", exc)
                attempts.append({"model": model, "outcome": "bad_request", "detail": str(exc)[:160]})
                return None
            except LLMCallError as exc:  # pragma: no cover - defensive
                state.failures += 1
                attempts.append({"model": model, "outcome": "error", "detail": str(exc)[:160]})
                return None
            state.rate = response.rate
            state.prompt_tokens += response.prompt_tokens
            state.completion_tokens += response.completion_tokens
            return response
        return None

    def _cool(self, state: ModelState, seconds: float, reason: str, exc: Exception) -> None:
        state.failures += 1
        state.last_error = str(exc)[:200]
        if seconds > 0:
            state.cooldown_until = max(state.cooldown_until, self._clock() + seconds)
            state.cooldown_reason = reason
        log(
            logger, logging.WARNING, "model cooling down", model=state.spec.id, reason=reason, seconds=round(seconds, 1)
        )
