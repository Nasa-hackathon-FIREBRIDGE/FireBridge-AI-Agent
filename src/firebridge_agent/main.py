"""FastAPI application: wiring, middleware and error handling."""

import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import __version__
from .api import analyses, health, sessions
from .api.deps import SlidingWindowLimiter, client_key
from .config import Settings, get_settings
from .errors import AgentError
from .observability import configure_logging, log, request_id
from .services import Services

logger = logging.getLogger("firebridge_agent")


def _error(
    status: int, code: str, message: str, details: dict | None = None, retry_after: float | None = None
) -> JSONResponse:
    body = {
        "error": {
            "code": code,
            "message": message,
            "request_id": request_id.get(),
            **({"details": details} if details else {}),
        }
    }
    headers = {"Retry-After": str(max(1, round(retry_after)))} if retry_after else None
    return JSONResponse(status_code=status, content=body, headers=headers)


def create_app(settings: Settings | None = None, *, services_factory=None) -> FastAPI:
    settings = settings or get_settings()
    if settings.environment == "production":
        if not settings.service_api_keys:
            raise ValueError("Production requires FBA_SERVICE_API_KEYS.")
        if not settings.groq_api_key.get_secret_value():
            raise ValueError("Production requires FBA_GROQ_API_KEY.")
        if not settings.models:
            raise ValueError("Production requires at least one model.")
    configure_logging(settings.log_level, settings.log_json)
    limiter = SlidingWindowLimiter(settings.rate_limit_per_minute)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if not settings.groq_api_key.get_secret_value() and services_factory is None:
            log(
                logger,
                logging.WARNING,
                "GROQ_API_KEY is not set: summaries and chat will fall back to deterministic output",
            )
        app.state.services = await (services_factory(settings) if services_factory else Services.start(settings))
        log(logger, logging.INFO, "service started", version=__version__, models=",".join(settings.models))
        try:
            yield
        finally:
            await app.state.services.stop()

    app = FastAPI(
        root_path=settings.root_path,
        title="FireBridge AI Agent",
        version=__version__,
        description="Grounded explanations of FIREBRIDGE satellite fire analyses: evidence-backed summaries, tool-using chat, short- and long-term memory, Groq model fallback.",
        lifespan=lifespan,
    )
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"], allow_headers=["*"]
        )

    @app.middleware("http")
    async def context(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        token = request_id.set(rid)
        started = time.perf_counter()
        try:
            if request.url.path.startswith("/v1"):
                try:
                    limiter.check(client_key(request))
                except AgentError as exc:
                    return _error(exc.status, exc.code, exc.message, retry_after=exc.retry_after)
            response = await call_next(request)
            response.headers["X-Request-ID"] = rid
            log(
                logger,
                logging.INFO,
                "request",
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                ms=round((time.perf_counter() - started) * 1000),
            )
            return response
        finally:
            request_id.reset(token)

    @app.exception_handler(AgentError)
    async def agent_error(_: Request, exc: AgentError) -> JSONResponse:
        if exc.status >= 500:
            log(logger, logging.ERROR, "request failed", code=exc.code, error=exc.message)
        return _error(exc.status, exc.code, exc.message, exc.details, exc.retry_after)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        return _error(
            422,
            "invalid_input",
            "The request is invalid.",
            {"errors": [{"loc": list(e["loc"]), "msg": e["msg"]} for e in exc.errors()]},
        )

    @app.exception_handler(Exception)
    async def unexpected(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error")
        return _error(500, "internal_error", "Unexpected server error.")

    app.include_router(health.router)
    app.include_router(analyses.router)
    app.include_router(sessions.router)
    return app
