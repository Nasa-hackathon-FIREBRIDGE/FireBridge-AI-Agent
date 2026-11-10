"""The explainer agent: grounded summaries and tool-using chat over one evidence pack."""

import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from ..config import Settings
from ..data.evidence import EvidencePack
from ..errors import AllModelsExhausted
from ..llm.router import ModelRouter
from ..llm.types import LLMRequest, LLMResponse
from ..memory.bm25 import bm25_scores
from ..memory.long_term import LongTermMemory
from ..memory.short_term import ShortTermMemory
from ..memory.store import Store
from ..observability import log
from .guard import Verification, allowed_values, is_degenerate, strip_tool_references, verify
from .jsonutil import parse_json_object
from .prompts import REPAIR, summary_task, system_prompt
from .tools import Toolbox

logger = logging.getLogger(__name__)
_CITED = re.compile(r"\bF(\d+)\b")


FALLBACK = {
    "en": (
        "All language models are at their usage limits right now.",
        "Try again in about {s} s.",
        "The evidence most related to your question:",
    ),
    "fr": (
        "Tous les modèles de langage ont atteint leurs limites d'utilisation.",
        "Réessayez dans environ {s} s.",
        "Les éléments de preuve les plus liés à votre question :",
    ),
    "es": (
        "Todos los modelos de lenguaje han alcanzado sus límites de uso.",
        "Inténtelo de nuevo en unos {s} s.",
        "La evidencia más relacionada con su pregunta:",
    ),
    "ar": (
        "بلغت جميع النماذج اللغوية حدود الاستخدام حاليًا.",
        "حاول مرة أخرى بعد نحو {s} ثانية.",
        "الأدلة الأكثر صلة بسؤالك:",
    ),
}


def fallback_text(language: str, retry_after: float | None) -> str:
    head, wait, lead = FALLBACK.get(language.lower().split("-")[0], FALLBACK["en"])
    return " ".join([head, *([wait.format(s=round(retry_after))] if retry_after else []), lead])


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def add(self, response: LLMResponse) -> None:
        self.calls += 1
        self.prompt_tokens += response.prompt_tokens
        self.completion_tokens += response.completion_tokens


@dataclass
class ChatResult:
    answer: str
    generated_from: str
    model: str | None
    verification: dict[str, Any]
    citations: list[dict[str, str]]
    tool_calls: list[dict[str, Any]]
    attempts: list[dict[str, Any]]
    usage: dict[str, int]
    memories_used: list[str]
    message_id: int = 0
    retry_after: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def citations_in(text: str, pack: EvidencePack) -> list[dict[str, str]]:
    seen: list[str] = []
    for number in _CITED.findall(text):
        fid = f"F{number}"
        if fid not in seen and pack.fact(fid):
            seen.append(fid)
    return [{"id": fid, "text": pack.fact(fid).text} for fid in seen]  # type: ignore[union-attr]


def _attempts(response: LLMResponse) -> list[dict[str, Any]]:
    return [*response.attempts, {"model": response.model, "outcome": "ok"}]


class Agent:
    def __init__(
        self,
        router: ModelRouter,
        store: Store,
        short_term: ShortTermMemory,
        long_term: LongTermMemory,
        settings: Settings,
    ):
        self.router = router
        self.store = store
        self.short_term = short_term
        self.long_term = long_term
        self.settings = settings

    # ── summary ──────────────────────────────────────────────────────────────
    async def summarize(self, pack: EvidencePack, language: str, refresh: bool = False) -> dict[str, Any]:
        if not refresh and (cached := await self.store.get_summary(pack.analysis_id, language)):
            return {**cached, "cached": True}
        digest = pack.digest(self.settings.digest_token_budget)
        allowed = allowed_values(digest)
        messages = [
            {"role": "system", "content": system_prompt(language, digest, [], "")},
            {"role": "user", "content": summary_task(language)},
        ]
        usage, attempts = Usage(), []
        try:
            response = await self.router.complete(
                LLMRequest(messages, max_tokens=self.settings.summary_max_tokens, json_mode=True, purpose="summary")
            )
            usage.add(response)
            attempts += _attempts(response)
            summary = self._parse_summary(response.content, pack)
            verification = verify(self._summary_text(summary), allowed)
            if summary and verification.unsupported:
                repaired = await self._repair(
                    messages,
                    response,
                    verification,
                    json_mode=True,
                    max_tokens=self.settings.summary_max_tokens,
                    purpose="summary-repair",
                )
            else:
                repaired = None
            if repaired:
                usage.add(repaired)
                attempts += _attempts(repaired)
                candidate = self._parse_summary(repaired.content, pack)
                checked = verify(self._summary_text(candidate), allowed) if candidate else verification
                if candidate and len(checked.unsupported) < len(verification.unsupported):
                    summary, verification, response = candidate, checked, repaired
            if not summary:
                raise ValueError("summary was not valid JSON")
            payload = {
                **summary,
                "generated_from": "llm",
                "model": response.model,
                "verification": verification.as_dict(),
            }
        except (AllModelsExhausted, ValueError) as exc:
            log(logger, logging.WARNING, "summary fell back to template", error=str(exc)[:120])
            payload = {
                **self._template_summary(pack),
                "generated_from": "template",
                "model": None,
                "verification": Verification().as_dict(),
            }
            if isinstance(exc, AllModelsExhausted):
                payload["retry_after"] = exc.retry_after
        payload |= {
            "analysis_id": pack.analysis_id,
            "language": language,
            "attempts": attempts,
            "usage": usage.__dict__,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        payload["markdown"] = self._markdown(payload, pack)
        if payload["generated_from"] == "llm":
            await self.store.save_summary(pack.analysis_id, language, payload)
        return {**payload, "cached": False}

    async def _repair(
        self,
        messages: list[dict[str, Any]],
        answer: LLMResponse,
        verification: Verification,
        *,
        json_mode: bool,
        max_tokens: int,
        purpose: str,
    ) -> LLMResponse | None:
        """Ask the model that wrote the answer to drop unsupported numbers. Never a different (weaker)
        model, never a long wait, and a failed repair leaves the original answer in place."""
        try:
            repaired = await self.router.complete(
                LLMRequest(
                    [
                        *messages,
                        {"role": "assistant", "content": answer.content},
                        {"role": "user", "content": REPAIR.format(numbers=", ".join(verification.unsupported))},
                    ],
                    max_tokens=max_tokens,
                    json_mode=json_mode,
                    purpose=purpose,
                    restrict_to=(answer.model,),
                    max_wait_s=3.0,
                )
            )
        except AllModelsExhausted:
            return None
        if not repaired.content or is_degenerate(repaired.content) or repaired.finish_reason == "length":
            return None
        return repaired

    @staticmethod
    def _parse_summary(text: str, pack: EvidencePack) -> dict[str, Any] | None:
        data = parse_json_object(text)
        if not data or not isinstance(data.get("headline"), str) or not isinstance(data.get("findings"), list):
            return None
        findings = []
        for item in data["findings"][:8]:
            if not isinstance(item, dict) or not item.get("detail"):
                continue
            cited = [f for f in (str(x).strip("[] ") for x in item.get("facts") or []) if pack.fact(f)]
            cited += [c["id"] for c in citations_in(str(item["detail"]), pack) if c["id"] not in cited]
            findings.append(
                {"title": str(item.get("title") or "").strip(), "detail": str(item["detail"]).strip(), "facts": cited}
            )
        strings = lambda key: [str(x).strip() for x in data.get(key) or [] if str(x).strip()][:5]  # noqa: E731
        return (
            {
                "headline": data["headline"].strip(),
                "findings": findings,
                "caveats": strings("caveats"),
                "follow_up_questions": strings("follow_up_questions"),
            }
            if findings
            else None
        )

    @staticmethod
    def _summary_text(summary: dict[str, Any] | None) -> str:
        if not summary:
            return ""
        return "\n".join(
            [summary["headline"], *(f"{f['title']} {f['detail']}" for f in summary["findings"]), *summary["caveats"]]
        )

    @staticmethod
    def _template_summary(pack: EvidencePack) -> dict[str, Any]:
        """Deterministic summary from the highest-priority facts, used when no model is available."""
        top = [f for f in pack.facts if f.priority == 1 and f.topic not in ("method",)]
        headline = next((f.text for f in top if f.topic == "jury"), top[0].text if top else pack.title)
        return {
            "headline": headline,
            "findings": [{"title": f.topic.capitalize(), "detail": f.text, "facts": [f.id]} for f in top[:6]],
            "caveats": [f.text for f in pack.facts if f.topic == "method"],
            "follow_up_questions": [
                "Which months were unusual, and do the satellites agree?",
                "How much of the VIIRS–MODIS difference is instrumental?",
                "What changed in the observing system during this period?",
            ],
        }

    @staticmethod
    def _markdown(payload: dict[str, Any], pack: EvidencePack) -> str:
        lines = [f"## {payload['headline']}", ""]
        for f in payload["findings"]:
            refs = f" [{', '.join(f['facts'])}]" if f["facts"] and payload["generated_from"] == "llm" else ""
            lines.append(f"- **{f['title']}** {f['detail']}{refs}" if f["title"] else f"- {f['detail']}{refs}")
        if payload["caveats"]:
            lines += ["", "**Caveats**", *(f"- {c}" for c in payload["caveats"])]
        if payload["follow_up_questions"]:
            lines += ["", "**Ask next**", *(f"- {q}" for q in payload["follow_up_questions"])]
        lines += ["", f"_{pack.title} · sources: {', '.join(pack.sources)}_"]
        return "\n".join(lines)

    # ── chat ─────────────────────────────────────────────────────────────────
    async def chat(self, session: dict[str, Any], pack: EvidencePack, user_text: str) -> ChatResult:
        conversation_summary, recent = await self.short_term.context(session)
        memories = await self.long_term.recall(session.get("user_id"), session["analysis_id"], user_text)
        memory_lines = [m["content"] for m in memories]
        digest = pack.digest(self.settings.chat_digest_token_budget)
        toolbox = Toolbox(pack)
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": system_prompt(session["language"], digest, memory_lines, conversation_summary),
            },
            *({"role": m["role"], "content": m["content"]} for m in recent),
            {"role": "user", "content": user_text},
        ]
        verified_history = [
            m["content"]
            for m in recent
            if m["role"] == "assistant" and m["meta"].get("verification", {}).get("status") == "verified"
        ]
        tool_outputs: list[str] = []
        trace: list[dict[str, Any]] = []
        attempts: list[dict[str, Any]] = []
        usage = Usage()
        try:
            answer, model = await self._tool_loop(messages, toolbox, tool_outputs, trace, attempts, usage)
            allowed = allowed_values(
                digest, user_text, conversation_summary, *memory_lines, *verified_history, *tool_outputs
            )
            verification = verify(answer, allowed)
            repaired = None
            if verification.unsupported:
                repaired = await self._repair(
                    messages,
                    LLMResponse(model=model, content=answer),
                    verification,
                    json_mode=False,
                    max_tokens=self.settings.answer_max_tokens,
                    purpose="chat-repair",
                )
            if repaired:
                usage.add(repaired)
                attempts += _attempts(repaired)
                checked = verify(repaired.content, allowed)
                if len(checked.unsupported) < len(verification.unsupported):
                    answer, verification, model = repaired.content, checked, repaired.model
            answer = strip_tool_references(answer, [t["function"]["name"] for t in toolbox.schemas()])
            result = ChatResult(
                answer,
                "llm",
                model,
                verification.as_dict(),
                citations_in(answer, pack),
                trace,
                attempts,
                usage.__dict__,
                memory_lines,
            )
        except AllModelsExhausted as exc:
            result = self._fallback_answer(
                session["language"], pack, user_text, exc, trace, attempts, usage, memory_lines
            )
        await self.store.add_message(session["id"], "user", user_text)
        result.message_id = await self.store.add_message(
            session["id"],
            "assistant",
            result.answer,
            {
                "model": result.model,
                "generated_from": result.generated_from,
                "verification": result.verification,
                "tool_calls": result.tool_calls,
                "citations": [c["id"] for c in result.citations],
            },
        )
        return result

    async def _tool_loop(
        self,
        messages: list[dict[str, Any]],
        toolbox: Toolbox,
        outputs: list[str],
        trace: list[dict[str, Any]],
        attempts: list[dict[str, Any]],
        usage: Usage,
    ) -> tuple[str, str]:
        schemas = toolbox.schemas()
        excluded: tuple[str, ...] = ()
        for step in range(self.settings.max_tool_steps + 1):
            final_step = step == self.settings.max_tool_steps
            response = await self.router.complete(
                LLMRequest(
                    messages,
                    tools=None if final_step else schemas,
                    max_tokens=self.settings.answer_max_tokens,
                    purpose="chat",
                    exclude=excluded,
                )
            )
            usage.add(response)
            attempts += _attempts(response)
            if response.tool_calls and not final_step:
                messages.append(
                    {
                        "role": "assistant",
                        "content": response.content or "",
                        "tool_calls": [
                            {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                            for c in response.tool_calls
                        ],
                    }
                )
                for call in response.tool_calls[:4]:
                    output = toolbox.run(call.name, call.arguments)
                    outputs.append(output)
                    trace.append(
                        {
                            "name": call.name,
                            "arguments": call.arguments,
                            "model": response.model,
                            "error": '"error"' in output[:12],
                        }
                    )
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": output})
                for call in response.tool_calls[4:]:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": '{"error":"too many tool calls in one step"}',
                        }
                    )
                continue
            if response.content and not is_degenerate(response.content) and response.finish_reason != "length":
                return response.content, response.model
            if response.content:
                # Looping or truncated output: let another model answer this step.
                excluded = (*excluded, response.model)
                attempts.append(
                    {"model": response.model, "outcome": "rejected", "detail": "degenerate or truncated output"}
                )
                continue
            messages.append(
                {"role": "user", "content": "Answer the question now from the evidence and tool results above."}
            )
        raise AllModelsExhausted("The models did not produce an answer.")

    @staticmethod
    def _fallback_answer(
        language: str,
        pack: EvidencePack,
        question: str,
        exc: AllModelsExhausted,
        trace: list[dict[str, Any]],
        attempts: list[dict[str, Any]],
        usage: Usage,
        memories: list[str],
    ) -> ChatResult:
        """Degraded but honest answer: the facts most related to the question, no generated text."""
        scores = bm25_scores(question, [f.text for f in pack.facts])
        ranked = [f for score, f in sorted(zip(scores, pack.facts, strict=True), key=lambda x: -x[0]) if score > 0][
            :3
        ] or [f for f in pack.facts if f.priority == 1][:3]
        answer = "\n".join([fallback_text(language, exc.retry_after), *(f"- [{f.id}] {f.text}" for f in ranked)])
        result = ChatResult(
            answer,
            "fallback",
            None,
            Verification().as_dict(),
            [{"id": f.id, "text": f.text} for f in ranked],
            trace,
            attempts + exc.details.get("attempts", []),
            usage.__dict__,
            memories,
        )
        result.retry_after = exc.retry_after
        return result
