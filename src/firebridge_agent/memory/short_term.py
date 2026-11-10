"""Short-term memory: the recent turns verbatim, older turns folded into a rolling summary."""

import logging
from typing import Any

from ..agent.prompts import COMPACT
from ..errors import AgentError
from ..llm.router import ModelRouter
from ..llm.types import LLMRequest
from ..observability import log
from .store import Store

logger = logging.getLogger(__name__)


def _tokens(text: str) -> float:
    return len(text) / 3.5


class ShortTermMemory:
    def __init__(self, store: Store, router: ModelRouter, turns: int, token_budget: int, prefer: tuple[str, ...] = ()):
        self.prefer = prefer
        self.store = store
        self.router = router
        self.max_messages = max(2, turns * 2)
        self.token_budget = token_budget

    async def context(self, session: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
        """Rolling summary plus the most recent messages that fit the window and the token budget."""
        messages = await self.store.messages(session["id"], after_id=session["summarized_upto"])
        recent: list[dict[str, Any]] = []
        used = 0.0
        for message in reversed(messages):
            cost = _tokens(message["content"])
            if len(recent) >= self.max_messages or (recent and used + cost > self.token_budget):
                break
            recent.append(message)
            used += cost
        recent.reverse()
        while recent and recent[0]["role"] != "user":
            recent.pop(0)
        return session["rolling_summary"], recent

    async def compact(self, session_id: str) -> bool:
        """Fold everything but the latest turns into the rolling summary once the window overflows."""
        session = await self.store.get_session(session_id)
        if not session:
            return False
        messages = await self.store.messages(session_id, after_id=session["summarized_upto"])
        total = sum(_tokens(m["content"]) for m in messages)
        if len(messages) <= self.max_messages and total <= self.token_budget:
            return False
        keep = self.max_messages // 2
        old = messages[:-keep] if keep else messages
        while old and old[-1]["role"] != "assistant":
            old.pop()
        if not old:
            return False
        exchanges = "\n".join(f"{m['role'].upper()}: {m['content'][:700]}" for m in old)
        try:
            response = await self.router.complete(
                LLMRequest(
                    messages=[
                        {
                            "role": "user",
                            "content": COMPACT.format(
                                summary=session["rolling_summary"] or "(none)", exchanges=exchanges
                            ),
                        }
                    ],
                    max_tokens=320,
                    purpose="compact",
                    prefer=self.prefer,
                    user_facing=False,
                )
            )
            summary = response.content.strip()
        except AgentError as exc:
            log(logger, logging.WARNING, "compaction fell back to extractive summary", error=exc.code)
            summary = ""
        if not summary:
            asked = "; ".join(m["content"][:120] for m in old if m["role"] == "user")
            summary = (
                session["rolling_summary"] + " " if session["rolling_summary"] else ""
            ) + f"Earlier the user asked: {asked}."
        await self.store.update_session_summary(session_id, summary[-2400:], old[-1]["id"])
        log(logger, logging.INFO, "conversation compacted", session=session_id, folded_messages=len(old))
        return True
