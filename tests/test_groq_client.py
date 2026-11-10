import json

import httpx
import pytest

from firebridge_agent.llm.catalog import spec_for
from firebridge_agent.llm.groq import (
    AuthFailed,
    BadRequest,
    GroqClient,
    ModelUnavailable,
    RateLimited,
    RequestTooLarge,
    Transient,
    clean_content,
    parse_duration,
)
from firebridge_agent.llm.types import LLMRequest


@pytest.mark.parametrize(
    ("raw", "seconds"),
    [
        ("2.71s", 2.71),
        ("1m26.4s", 86.4),
        ("120ms", 0.12),
        ("7h3m2s", 25382.0),
        ("12", 12.0),
        ("", None),
        ("soon", None),
    ],
)
def test_parses_groq_durations(raw, seconds):
    assert parse_duration(raw) == (pytest.approx(seconds) if seconds is not None else None)


def test_strips_inline_reasoning():
    assert clean_content("<think>secret plan</think>\nAnswer.") == "Answer."


def client(status: int, body: dict, headers: dict | None = None, seen: list | None = None) -> GroqClient:
    def handle(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(json.loads(request.content))
        return httpx.Response(status, json=body, headers=headers or {})

    return GroqClient("k", "https://groq.test/v1", 5, transport=httpx.MockTransport(handle))


async def test_sends_model_parameters_tools_and_json_mode():
    seen: list = []
    ok = {
        "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2},
    }
    headers = {
        "x-ratelimit-limit-tokens": "8000",
        "x-ratelimit-remaining-tokens": "7811",
        "x-ratelimit-reset-tokens": "1.4s",
    }
    response = await client(200, ok, headers, seen).chat(
        spec_for("qwen/qwen3.8-27b"), LLMRequest([{"role": "user", "content": "hi"}], json_mode=True)
    )
    assert seen[0]["reasoning_format"] == "hidden" and seen[0]["response_format"] == {"type": "json_object"}
    assert response.rate.limit_tokens == 8000 and response.rate.remaining_tokens == 7811
    await client(200, ok, seen=seen).chat(
        spec_for("allam-2-7b"), LLMRequest([{"role": "user", "content": "hi"}], tools=[{"type": "function"}])
    )
    assert "tools" not in seen[1] and "reasoning_format" not in seen[1]


async def test_parses_tool_calls():
    body = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [{"id": "c1", "function": {"name": "get_month", "arguments": '{"month":"2022-07"}'}}],
                },
                "finish_reason": "tool_calls",
            }
        ]
    }
    response = await client(200, body).chat(spec_for("openai/gpt-oss-20b"), LLMRequest([]))
    assert response.tool_calls[0].name == "get_month" and response.content == ""


@pytest.mark.parametrize(
    ("status", "body", "headers", "error"),
    [
        (
            429,
            {"error": {"message": "Rate limit reached on tokens per minute (TPM)"}},
            {"retry-after": "7"},
            RateLimited,
        ),
        (401, {"error": {"message": "Invalid API Key"}}, {}, AuthFailed),
        (404, {"error": {"message": "model not found", "code": "model_not_found"}}, {}, ModelUnavailable),
        (503, {"error": {"message": "overloaded"}}, {}, ModelUnavailable),
        (400, {"error": {"message": "too long", "code": "context_length_exceeded"}}, {}, RequestTooLarge),
        (413, {"error": {"message": "Request too large"}}, {}, RequestTooLarge),
        (400, {"error": {"message": "Failed to call a function", "code": "tool_use_failed"}}, {}, BadRequest),
    ],
)
async def test_maps_failures_to_typed_errors(status, body, headers, error):
    with pytest.raises(error):
        await client(status, body, headers).chat(spec_for("qwen/qwen3.8-27b"), LLMRequest([]))


async def test_rate_limit_scope_and_wait():
    with pytest.raises(RateLimited) as minute:
        await client(429, {"error": {"message": "tokens per minute"}}, {"retry-after": "7"}).chat(
            spec_for("qwen/qwen3.8-27b"), LLMRequest([])
        )
    assert minute.value.retry_after == 7 and minute.value.scope == "minute"
    with pytest.raises(RateLimited) as daily:
        await client(
            429,
            {"error": {"message": "Limit 1000, Used 1000 on requests per day (RPD)"}},
            {"x-ratelimit-reset-requests": "2h"},
        ).chat(spec_for("qwen/qwen3.8-27b"), LLMRequest([]))
    assert daily.value.scope == "daily" and daily.value.retry_after == 7200


async def test_network_errors_are_transient():
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    groq = GroqClient("k", "https://groq.test/v1", 5, transport=httpx.MockTransport(boom))
    with pytest.raises(Transient):
        await groq.chat(spec_for("qwen/qwen3.8-27b"), LLMRequest([]))
