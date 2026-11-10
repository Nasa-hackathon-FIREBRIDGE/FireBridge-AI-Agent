"""Long-term memory: durable facts about each user, kept across sessions and analyses.

Only what the user said about themselves is kept: preferences, profile, and what they are
investigating. Conclusions about the data are never stored from model output: a mistaken answer
would otherwise become a "memory" shown to every later user of the analysis. The evidence facts
are the authoritative conclusions.

Writing: after a turn, a small model proposes memories as JSON; each is validated, de-duplicated
against what is already stored (a near-duplicate reinforces the existing memory instead) and the
user's memories are pruned to the most valuable. Reading: ranked by BM25 relevance to the
question, importance and use; preferences and profile always come along.
"""

import logging
from typing import Any

from ..agent.jsonutil import parse_json_object
from ..agent.prompts import EXTRACT
from ..agent.safety import is_safe_memory
from ..errors import AgentError
from ..llm.router import ModelRouter
from ..llm.types import LLMRequest
from ..observability import log
from .bm25 import bm25_scores, similarity
from .store import Store

logger = logging.getLogger(__name__)
KINDS = {"user": {"preference", "profile", "interest"}}
MAX_PER_SCOPE = 200
DUPLICATE_SIMILARITY = 0.6


def scopes(user_id: str | None, analysis_id: str) -> list[tuple[str, str]]:
    """Only the user's own memories (the analysis id is accepted for callers' symmetry)."""
    del analysis_id
    return [("user", user_id)] if user_id else []


class LongTermMemory:
    def __init__(self, store: Store, router: ModelRouter, top_k: int, enabled: bool, prefer: tuple[str, ...] = ()):
        self.prefer = prefer
        self.store = store
        self.router = router
        self.top_k = top_k
        self.enabled = enabled

    async def recall(self, user_id: str | None, analysis_id: str, query: str) -> list[dict[str, Any]]:
        rows = await self.store.memories(scopes(user_id, analysis_id))
        if not rows:
            return []
        relevance = bm25_scores(query, [r["content"] for r in rows])
        best = max(relevance) or 1.0
        scored = [
            (rel / best + 0.3 * r["importance"] + 0.05 * min(r["uses"], 5), r)
            for rel, r in zip(relevance, rows, strict=True)
        ]
        ranked = [r for _, r in sorted(scored, key=lambda x: -x[0])]
        # Who the user is and how they want answers always come along; the rest must be relevant.
        personal = [r for r in ranked if r["kind"] in ("preference", "profile")][:3]
        best_score = {r["id"]: score for score, r in scored}
        others = [
            r
            for r in ranked
            if r not in personal and r["kind"] not in ("preference", "profile") and best_score[r["id"]] >= 0.35
        ]
        others = others[: max(0, self.top_k - len(personal))]
        chosen = personal + others
        await self.store.touch_memories([r["id"] for r in chosen])
        return chosen

    async def remember(self, session: dict[str, Any], user_text: str, answer: str) -> list[int]:
        if not self.enabled:
            return []
        try:
            response = await self.router.complete(
                LLMRequest(
                    messages=[
                        {"role": "user", "content": EXTRACT.format(user=user_text[:1200], assistant=answer[:1600])}
                    ],
                    max_tokens=350,
                    json_mode=True,
                    temperature=0.0,
                    purpose="memory",
                    prefer=self.prefer,
                    user_facing=False,
                )
            )
        except AgentError as exc:
            log(logger, logging.WARNING, "memory extraction skipped", error=exc.code)
            return []
        proposed = (parse_json_object(response.content) or {}).get("memories") or []
        existing = await self.store.memories(scopes(session.get("user_id"), session["analysis_id"]))
        stored: list[int] = []
        touched: set[tuple[str, str]] = set()
        for item in proposed[:5]:
            if not isinstance(item, dict):
                continue
            scope, kind = str(item.get("scope", "")), str(item.get("kind", ""))
            content = " ".join(str(item.get("content", "")).split())[:300]
            if scope not in KINDS or kind not in KINDS[scope] or len(content) < 8:
                continue
            if not is_safe_memory(content):  # would be replayed into every later prompt of this user
                log(logger, logging.WARNING, "memory candidate dropped", reason="reads as an instruction")
                continue
            scope_id = session.get("user_id") if scope == "user" else session["analysis_id"]
            if not scope_id:
                continue
            try:
                importance = min(1.0, max(0.0, float(item.get("importance", 0.5))))
            except (TypeError, ValueError):
                importance = 0.5
            duplicate = next(
                (
                    e
                    for e in existing
                    if e["scope"] == scope
                    and e["scope_id"] == scope_id
                    and similarity(e["content"], content) >= DUPLICATE_SIMILARITY
                ),
                None,
            )
            if duplicate:
                await self.store.reinforce_memory(duplicate["id"], importance)
                continue
            memory_id = await self.store.add_memory(scope, scope_id, kind, content, importance, session["id"])
            existing.append({"id": memory_id, "scope": scope, "scope_id": scope_id, "content": content})
            stored.append(memory_id)
            touched.add((scope, scope_id))
        for scope, scope_id in touched:
            await self.store.prune_memories(scope, scope_id, MAX_PER_SCOPE)
        if stored:
            log(logger, logging.INFO, "memories stored", count=len(stored), session=session["id"])
        return stored
