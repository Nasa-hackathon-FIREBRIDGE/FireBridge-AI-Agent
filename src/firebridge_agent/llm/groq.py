"""Thin async client for Groq's OpenAI-compatible chat API with typed failures."""

import json
import re
import time
from typing import Any

import httpx

from .catalog import ModelSpec
from .types import LLMRequest, LLMResponse, RateState, ToolCall


class LLMCallError(Exception):
    def __init__(self, model: str, message: str, *, status: int | None = None):
        super().__init__(message)
        self.model = model
        self.status = status


class RateLimited(LLMCallError):
    """429: per-minute token/request budget or daily budget of this model is used up."""

    def __init__(self, model: str, message: str, retry_after: float, scope: str):
        super().__init__(model, message, status=429)
        self.retry_after = retry_after
        self.scope = scope


class ModelUnavailable(LLMCallError):
    """Model missing, decommissioned or the provider is failing for it (404/5xx)."""


class RequestTooLarge(LLMCallError):
    """Prompt does not fit this model's context or per-request limits (413 / context_length_exceeded)."""


class Transient(LLMCallError):
    """Network error or timeout: worth one retry."""


class BadRequest(LLMCallError):
    """A 400 that another model may handle (e.g. a malformed tool call), but not worth retrying as-is."""


class AuthFailed(LLMCallError):
    """401/403: the API key is wrong; no model will work."""


_DURATION = re.compile(r"(?:(\d+(?:\.\d+)?)h)?(?:(\d+(?:\.\d+)?)m(?!s))?(?:(\d+(?:\.\d+)?)s)?(?:(\d+(?:\.\d+)?)ms)?$")


def parse_duration(value: str | None) -> float | None:
    """Groq reset headers look like '1m26.4s', '2.71s', '120ms', '7h3m2s'. Returns seconds."""
    if not value:
        return None
    value = value.strip()
    try:
        return float(value)
    except ValueError:
        pass
    match = _DURATION.match(value)
    if not match or not any(match.groups()):
        return None
    hours, minutes, seconds, millis = (float(g) if g else 0.0 for g in match.groups())
    return hours * 3600 + minutes * 60 + seconds + millis / 1000


def parse_rate_headers(headers: httpx.Headers) -> RateState:
    def number(name: str) -> int | None:
        raw = headers.get(name)
        try:
            return int(float(raw)) if raw is not None else None
        except ValueError:
            return None

    return RateState(
        limit_tokens=number("x-ratelimit-limit-tokens"),
        remaining_requests=number("x-ratelimit-remaining-requests"),
        remaining_tokens=number("x-ratelimit-remaining-tokens"),
        reset_requests_s=parse_duration(headers.get("x-ratelimit-reset-requests")),
        reset_tokens_s=parse_duration(headers.get("x-ratelimit-reset-tokens")),
        observed_at=time.monotonic(),
    )


_THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def clean_content(text: str | None) -> str:
    """Remove reasoning blocks some models emit inline even when asked to hide them."""
    return _THINK.sub("", text or "").strip()


class GroqClient:
    def __init__(
        self, api_key: str, base_url: str, timeout_s: float, transport: httpx.AsyncBaseTransport | None = None
    ):
        self._http = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout_s, connect=10.0),
            headers={"Authorization": f"Bearer {api_key}", "User-Agent": "FireBridge-AI-Agent/0.1"},
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def chat(self, spec: ModelSpec, request: LLMRequest) -> LLMResponse:
        body: dict[str, Any] = {
            "model": spec.id,
            "messages": request.messages,
            "max_tokens": min(request.max_tokens, spec.max_output),
            "temperature": request.temperature,
            **spec.params,
        }
        if request.tools and spec.tools:
            body["tools"] = request.tools
            body["tool_choice"] = "auto"
        if request.json_mode and spec.json_mode and not body.get("tools"):
            body["response_format"] = {"type": "json_object"}
        try:
            response = await self._http.post("/chat/completions", json=body)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise Transient(spec.id, f"{type(exc).__name__}: {exc}") from exc
        rate = parse_rate_headers(response.headers)
        if response.status_code == 200:
            return self._parse(spec.id, response.json(), rate)
        self._raise(spec.id, response, rate)
        raise AssertionError("unreachable")

    @staticmethod
    def _parse(model: str, data: dict[str, Any], rate: RateState) -> LLMResponse:
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        calls = [
            ToolCall(
                id=call.get("id") or f"call_{i}",
                name=call["function"]["name"],
                arguments=call["function"].get("arguments") or "{}",
            )
            for i, call in enumerate(message.get("tool_calls") or [])
            if call.get("function", {}).get("name")
        ]
        usage = data.get("usage") or {}
        return LLMResponse(
            model=model,
            content=clean_content(message.get("content")),
            tool_calls=calls,
            finish_reason=choice.get("finish_reason"),
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            rate=rate,
        )

    @staticmethod
    def _raise(model: str, response: httpx.Response, rate: RateState) -> None:
        status = response.status_code
        try:
            error = response.json().get("error") or {}
        except (json.JSONDecodeError, ValueError):
            error = {}
        message = str(error.get("message") or response.text[:300])
        code = str(error.get("code") or error.get("type") or "")
        if status == 429:
            retry_after = parse_duration(response.headers.get("retry-after"))
            daily = "per day" in message.lower() or "(rpd)" in message.lower() or "(tpd)" in message.lower()
            scope = "daily" if daily else "minute"
            if retry_after is None:
                retry_after = (rate.reset_requests_s if daily else rate.reset_tokens_s) or (3600.0 if daily else 20.0)
            raise RateLimited(model, message, retry_after, scope)
        if status in (401, 403):
            raise AuthFailed(model, message, status=status)
        if status == 413 or code == "context_length_exceeded" or "context length" in message.lower():
            raise RequestTooLarge(model, message, status=status)
        if status in (404,) or code in ("model_not_found", "model_decommissioned") or status >= 500:
            raise ModelUnavailable(model, message, status=status)
        raise BadRequest(model, f"{code}: {message}" if code else message, status=status)
