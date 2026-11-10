from typing import Annotated

from fastapi import APIRouter, Depends, Response

from .. import __version__
from ..services import Services
from .deps import services

Svc = Annotated[Services, Depends(services)]
router = APIRouter(tags=["health"])


@router.get("/health", summary="Liveness")
async def health(svc: Svc) -> dict:
    return {"status": "ok", "version": __version__, "revision": svc.settings.release_revision}


@router.get("/ready", summary="Readiness: database reachable and a Groq key configured")
async def ready(svc: Svc, response: Response) -> dict:
    try:
        database = await svc.store.ping()
    except Exception:
        database = False
    key = bool(svc.settings.groq_api_key.get_secret_value()) or svc.groq is None
    models = svc.router.status()
    available = sum(m["available"] for m in models)
    is_ready = database and key and available > 0
    response.status_code = 200 if is_ready else 503
    return {
        "status": "ready" if is_ready else "not_ready",
        "revision": svc.settings.release_revision,
        "database": database,
        "groq_key_configured": key,
        "models_available": available,
    }
