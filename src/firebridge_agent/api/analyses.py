from typing import Annotated

from fastapi import APIRouter, Depends

from ..agent.references import fact_refs
from ..services import Services, sources_of
from .deps import require_api_key, services
from .schemas import AnalysisOut, FactOut, IngestRequest, SummaryOut, SummaryRequest

Svc = Annotated[Services, Depends(services)]
router = APIRouter(prefix="/v1", tags=["analyses"], dependencies=[Depends(require_api_key)])


def _out(pack) -> AnalysisOut:
    return AnalysisOut(
        analysis_id=pack.analysis_id,
        title=pack.title,
        sources=pack.sources,
        period_start=pack.period_start,
        period_end_exclusive=pack.period_end_exclusive,
        built_at=pack.built_at,
        facts=[
            FactOut(id=f.id, topic=f.topic, text=f.text, priority=f.priority, refs=fact_refs(f)) for f in pack.facts
        ],
    )


@router.post(
    "/analyses", response_model=AnalysisOut, status_code=201, summary="Load (or reload) an analysis into the agent"
)
async def ingest(body: IngestRequest, svc: Svc) -> AnalysisOut:
    return _out(await svc.ingest(body.analysis_id, report=body.report, use_api=body.use_api))


@router.get("/analyses", summary="Analyses the agent knows")
async def list_analyses(svc: Svc) -> list[dict]:
    return [
        {**{k: row[k] for k in ("id", "title", "created_at", "updated_at")}, "sources": sources_of(row)}
        for row in await svc.store.list_analyses()
    ]


@router.get("/analyses/{analysis_id}", response_model=AnalysisOut, summary="Evidence facts of an ingested analysis")
async def get_analysis(analysis_id: str, svc: Svc) -> AnalysisOut:
    return _out(await svc.pack(analysis_id, auto_ingest=False))


@router.post(
    "/analyses/{analysis_id}/summary",
    response_model=SummaryOut,
    summary="Grounded executive summary (cached per language)",
)
async def summarize(svc: Svc, analysis_id: str, body: SummaryRequest | None = None) -> SummaryOut:
    body = body or SummaryRequest()
    pack = await svc.pack(analysis_id)
    return SummaryOut(**await svc.agent.summarize(pack, body.language, refresh=body.refresh))


@router.get("/models", tags=["models"], summary="Model fallback chain: availability, cooldowns, usage")
async def models(svc: Svc) -> dict:
    return {
        "fallbacks": svc.router.fallbacks,
        "proactive_switches": svc.router.proactive_switches,
        "models": svc.router.status(),
    }
