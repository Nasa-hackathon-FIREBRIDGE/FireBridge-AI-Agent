"""Long-lived service objects, built once at startup and shared by every request."""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from .agent.agent import Agent
from .config import Settings
from .data.evidence import EvidencePack, build_pack
from .data.firebridge import FireBridgeClient
from .errors import InvalidInput, NotFound
from .llm.groq import GroqClient
from .llm.router import ModelRouter
from .memory.long_term import LongTermMemory
from .memory.short_term import ShortTermMemory
from .memory.store import Store
from .observability import log

logger = logging.getLogger(__name__)


@dataclass
class Services:
    settings: Settings
    store: Store
    router: ModelRouter
    firebridge: FireBridgeClient
    agent: Agent
    groq: GroqClient | None = None
    background: set[asyncio.Task[Any]] = field(default_factory=set)

    @classmethod
    async def start(
        cls,
        settings: Settings,
        *,
        router: ModelRouter | None = None,
        firebridge: FireBridgeClient | None = None,
        store: Store | None = None,
    ) -> "Services":
        store = store or Store(settings.database_path)
        await store.open()
        groq = None
        if router is None:
            groq = GroqClient(settings.groq_api_key.get_secret_value(), settings.groq_base_url, settings.llm_timeout_s)
            router = ModelRouter(groq, settings.models, max_wait_s=settings.max_wait_for_model_s)
        firebridge = firebridge or FireBridgeClient(settings.firebridge_api_url, settings.firebridge_timeout_s)
        housekeeping = tuple(settings.housekeeping_models)
        short_term = ShortTermMemory(
            store, router, settings.short_term_turns, settings.short_term_token_budget, housekeeping
        )
        long_term = LongTermMemory(store, router, settings.long_term_top_k, settings.memory_extraction, housekeeping)
        agent = Agent(router, store, short_term, long_term, settings)
        return cls(settings, store, router, firebridge, agent, groq)

    async def stop(self) -> None:
        if self.background:
            await asyncio.wait(self.background, timeout=10)
        await self.firebridge.aclose()
        if self.groq:
            await self.groq.aclose()
        await self.store.close()

    def spawn(self, coroutine: Any, name: str) -> None:
        """Run housekeeping after the response without losing errors or the task reference."""

        async def guarded() -> None:
            try:
                await coroutine
            except Exception:
                logger.exception("background task failed", extra={"fields": {"task": name}})

        task = asyncio.create_task(guarded(), name=name)
        self.background.add(task)
        task.add_done_callback(self.background.discard)

    # ── analyses ─────────────────────────────────────────────────────────────
    async def ingest(
        self, analysis_id: str, *, report: dict[str, Any] | None = None, use_api: bool = True
    ) -> EvidencePack:
        if report is not None:
            body = report.get("report", report)
            if not isinstance(body, dict) or "months" not in body or "meta" not in body:
                raise InvalidInput(
                    "The report does not look like a FIREBRIDGE sensor-bridge export (expected 'report.meta' and 'report.months')."
                )
            declared = report.get("analysis_id")
            if declared and declared != analysis_id:
                raise InvalidInput(f"The report belongs to analysis {declared}, not {analysis_id}.")
        if not use_api and report is None:
            raise InvalidInput("Nothing to ingest: enable use_api or provide a report.")
        bundle = await self.firebridge.analysis_bundle(analysis_id) if use_api else None
        pack = build_pack(analysis_id, bundle=bundle, report=report)
        await self.store.save_analysis(analysis_id, pack.title, pack.sources, pack.to_json())
        log(
            logger,
            logging.INFO,
            "analysis ingested",
            analysis=analysis_id,
            facts=len(pack.facts),
            sources=",".join(pack.sources),
        )
        return pack

    async def pack(self, analysis_id: str, *, auto_ingest: bool = True) -> EvidencePack:
        row = await self.store.get_analysis(analysis_id)
        if row:
            return EvidencePack.from_json(row["pack"])
        if not auto_ingest:
            raise NotFound(f"Analysis {analysis_id} has not been ingested.")
        return await self.ingest(analysis_id)


def sources_of(row: dict[str, Any]) -> list[str]:
    return json.loads(row["sources"])
