"""Test doubles: a scripted LLM client (no network) and a FIREBRIDGE API transport."""

import json
from collections.abc import Callable

import httpx

from firebridge_agent.llm.catalog import ModelSpec
from firebridge_agent.llm.types import LLMRequest, LLMResponse, RateState, ToolCall

from .fixtures import ANALYSIS_ID, native_bundle

Handler = Callable[[ModelSpec, LLMRequest], LLMResponse | Exception]


class ScriptedLLM:
    """Implements GroqClient.chat. Each call is answered by `handler`, which may return a response
    or an exception to raise; every call is recorded."""

    def __init__(self, handler: Handler):
        self.handler = handler
        self.calls: list[tuple[str, LLMRequest]] = []

    async def chat(self, spec: ModelSpec, request: LLMRequest) -> LLMResponse:
        self.calls.append((spec.id, request))
        result = self.handler(spec, request)
        if isinstance(result, Exception):
            raise result
        result.model = spec.id
        return result

    async def aclose(self) -> None:
        return None


def reply(content: str = "", tool_calls: list[ToolCall] | None = None, rate: RateState | None = None) -> LLMResponse:
    return LLMResponse(
        model="",
        content=content,
        tool_calls=tool_calls or [],
        finish_reason="stop",
        prompt_tokens=100,
        completion_tokens=20,
        rate=rate,
    )


def agent_handler(spec: ModelSpec, request: LLMRequest) -> LLMResponse:
    """A well-behaved model for end-to-end API tests."""
    if request.purpose.startswith("summary"):
        return reply(
            json.dumps(
                {
                    "headline": "July 2022 was the most unusual month, and both satellites agree.",
                    "findings": [
                        {
                            "title": "Unusual July",
                            "detail": "30 Aqua-equivalent active cell-days against a usual 1.5.",
                            "facts": ["F99", "bogus"],
                        }
                    ],
                    "caveats": ["Detections are not fires."],
                    "follow_up_questions": ["Why is August 2021 disputed?"],
                }
            )
        )
    if request.purpose == "memory":
        return reply(
            json.dumps(
                {
                    "memories": [
                        {
                            "scope": "user",
                            "kind": "preference",
                            "content": "The user prefers short answers.",
                            "importance": 0.8,
                        },
                        {
                            "scope": "user",
                            "kind": "interest",
                            "content": "The user is investigating the July 2022 fire season.",
                            "importance": 0.7,
                        },
                    ]
                }
            )
        )
    if request.purpose == "compact":
        return reply("The user asked about July 2022 [F1].")
    tool_results = [json.loads(m["content"]) for m in request.messages if m["role"] == "tool"]
    if request.tools and not tool_results:
        return reply(tool_calls=[ToolCall(id="call_1", name="get_month", arguments='{"month": "July 2022"}')])
    if tool_results and "native_detections_all_satellites" in tool_results[-1]:
        return reply(f"July 2022 had {tool_results[-1]['native_detections_all_satellites']:,} native detections [F1].")
    return reply("July 2022 stands out in the record [F1].")


def firebridge_transport(bundle: dict | None = None) -> httpx.MockTransport:
    bundle = bundle or native_bundle()
    routes = {
        f"/api/v1/analyses/{ANALYSIS_ID}": bundle["analysis"],
        f"/api/v1/analyses/{ANALYSIS_ID}/summary": bundle["summary"],
        f"/api/v1/analyses/{ANALYSIS_ID}/calendar": bundle["calendar"],
        "/api/v1/areas/area-1": bundle["area"],
    }

    def handle(request: httpx.Request) -> httpx.Response:
        data = routes.get(request.url.path)
        if data is None:
            return httpx.Response(404, json={"error": {"code": "not_found"}})
        return httpx.Response(200, json={"schema_version": "1", "data": data})

    return httpx.MockTransport(handle)
