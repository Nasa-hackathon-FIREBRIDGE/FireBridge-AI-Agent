import pytest

from firebridge_agent.errors import AllModelsExhausted, LLMConfigurationError
from firebridge_agent.llm.groq import AuthFailed, BadRequest, RateLimited, Transient
from firebridge_agent.llm.router import ModelRouter
from firebridge_agent.llm.types import LLMRequest, RateState

from .fakes import ScriptedLLM, reply

MODELS = ["qwen/qwen3.8-27b", "openai/gpt-oss-120b", "openai/gpt-oss-20b", "allam-2-7b"]


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now += seconds


def router(handler, max_wait: float = 20.0):
    clock = Clock()
    llm = ScriptedLLM(handler)
    return ModelRouter(llm, MODELS, max_wait_s=max_wait, clock=clock, sleep=clock.sleep), llm, clock


def request(**kwargs) -> LLMRequest:
    return LLMRequest([{"role": "user", "content": "How unusual was July 2022?"}], max_tokens=200, **kwargs)


async def test_switches_model_on_rate_limit_and_remembers_the_cooldown():
    def handler(spec, req):
        return RateLimited(spec.id, "tpm", 30.0, "minute") if spec.id == "qwen/qwen3.8-27b" else reply("ok")

    r, llm, clock = router(handler)
    first = await r.complete(request())
    assert first.model == "openai/gpt-oss-120b"
    assert [a["outcome"] for a in first.attempts] == ["rate_limited"]
    second = await r.complete(request())
    assert second.model == "openai/gpt-oss-120b" and len(llm.calls) == 3  # qwen skipped while cooling down
    clock.now += 31
    await r.complete(request())
    assert llm.calls[-2][0] == "qwen/qwen3.8-27b"  # tried again once its cooldown is over
    assert r.fallbacks == 2


async def test_proactively_tries_a_model_with_tokens_left_first():
    def handler(spec, req):
        rate = (
            RateState(limit_tokens=8000, remaining_tokens=50, reset_tokens_s=50.0, observed_at=1000.0)
            if spec.id == "qwen/qwen3.8-27b"
            else None
        )
        return reply("ok", rate=rate)

    r, _, _ = router(handler)
    await r.complete(request())  # qwen answers and reports 50 tokens left this minute
    second = await r.complete(request())
    assert second.model == "openai/gpt-oss-120b" and r.proactive_switches == 1


async def test_low_token_models_are_still_tried_as_a_last_resort():
    def handler(spec, req):
        if spec.id in ("openai/gpt-oss-120b", "openai/gpt-oss-20b"):
            return RateLimited(spec.id, "tpm", 60.0, "minute")
        return reply(
            "ok", rate=RateState(limit_tokens=8000, remaining_tokens=10, reset_tokens_s=60, observed_at=1000.0)
        )

    r, _, _ = router(handler)
    await r.complete(request())
    again = await r.complete(request())
    assert again.model == "qwen/qwen3.8-27b"


async def test_waits_for_a_short_cooldown_and_gives_up_on_a_long_one():
    calls = {"n": 0}

    def handler(spec, req):
        calls["n"] += 1
        return RateLimited(spec.id, "tpm", 5.0, "minute") if calls["n"] <= 3 else reply("ok")

    r, _, clock = router(handler)
    started = clock.now
    response = await r.complete(request())
    assert response.model == "qwen/qwen3.8-27b" and clock.now - started >= 5

    r2, _, _ = router(lambda spec, req: RateLimited(spec.id, "rpd", 3600.0, "daily"))
    with pytest.raises(AllModelsExhausted) as exc:
        await r2.complete(request())
    assert exc.value.retry_after == pytest.approx(3600, abs=1)


async def test_respects_capabilities_and_request_restrictions():
    r, _, _ = router(lambda spec, req: reply("ok"))
    assert (await r.complete(request(restrict_to=("openai/gpt-oss-20b",)))).model == "openai/gpt-oss-20b"
    assert (await r.complete(request(exclude=("qwen/qwen3.8-27b",)))).model == "openai/gpt-oss-120b"
    assert (await r.complete(request(user_facing=False, prefer=("allam-2-7b",)))).model == "allam-2-7b"
    with pytest.raises(AllModelsExhausted) as exc:
        await r.complete(request(restrict_to=("allam-2-7b",)))  # allam is reserved for housekeeping
    assert exc.value.details["reasons"]["allam-2-7b"] == "reserved for housekeeping"
    big = LLMRequest(
        [{"role": "user", "content": "x" * 20_000}], max_tokens=200, user_facing=False, prefer=("allam-2-7b",)
    )
    assert (await r.complete(big)).model != "allam-2-7b"


async def test_retries_transient_errors_once_then_moves_on():
    seen: list[str] = []

    def handler(spec, req):
        seen.append(spec.id)
        return Transient(spec.id, "timeout") if spec.id == "qwen/qwen3.8-27b" else reply("ok")

    r, _, _ = router(handler)
    assert (await r.complete(request())).model == "openai/gpt-oss-120b"
    assert seen == ["qwen/qwen3.8-27b", "qwen/qwen3.8-27b", "openai/gpt-oss-120b"]


async def test_bad_request_falls_through_and_auth_failure_stops():
    r, _, _ = router(
        lambda spec, req: BadRequest(spec.id, "tool_use_failed") if spec.id == "qwen/qwen3.8-27b" else reply("ok")
    )
    assert (await r.complete(request())).model == "openai/gpt-oss-120b"
    r2, _, _ = router(lambda spec, req: AuthFailed(spec.id, "Invalid API Key", status=401))
    with pytest.raises(LLMConfigurationError):
        await r2.complete(request())


async def test_reports_status_per_model():
    r, _, _ = router(
        lambda spec, req: RateLimited(spec.id, "tpm", 10.0, "minute") if spec.id == "qwen/qwen3.8-27b" else reply("ok")
    )
    await r.complete(request(purpose="summary"))
    status = {m["model"]: m for m in r.status()}
    assert status["qwen/qwen3.8-27b"]["available"] is False and status["qwen/qwen3.8-27b"]["rate_limited"] == 1
    assert status["openai/gpt-oss-120b"]["by_purpose"] == {"summary": 1}
