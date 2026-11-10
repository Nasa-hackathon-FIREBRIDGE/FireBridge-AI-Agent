from typing import Annotated

from fastapi import APIRouter, Depends, Query

from ..errors import NotFound
from ..services import Services
from .deps import require_api_key, services
from .schemas import ChatOut, ChatRequest, MemoryOut, MessageOut, SessionCreate, SessionDetail, SessionOut

Svc = Annotated[Services, Depends(services)]
router = APIRouter(prefix="/v1", dependencies=[Depends(require_api_key)])
SESSION_FIELDS = ("id", "analysis_id", "user_id", "language", "rolling_summary", "created_at", "updated_at")


async def _session(svc: Services, session_id: str) -> dict:
    session = await svc.store.get_session(session_id)
    if not session:
        raise NotFound(f"Session {session_id} does not exist.")
    return session


@router.post(
    "/sessions",
    response_model=SessionOut,
    status_code=201,
    tags=["chat"],
    summary="Start a conversation about an analysis",
)
async def create_session(body: SessionCreate, svc: Svc) -> SessionOut:
    await svc.pack(body.analysis_id)  # loads the analysis from FIREBRIDGE on first use
    session = await svc.store.create_session(body.analysis_id, body.user_id, body.language)
    return SessionOut(**{k: session[k] for k in SESSION_FIELDS})


@router.get("/sessions", response_model=list[SessionOut], tags=["chat"])
async def list_sessions(
    svc: Svc,
    user_id: str | None = None,
    analysis_id: str | None = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> list[SessionOut]:
    return [
        SessionOut(**{k: s[k] for k in SESSION_FIELDS})
        for s in await svc.store.list_sessions(user_id, analysis_id, limit)
    ]


@router.get("/sessions/{session_id}", response_model=SessionDetail, tags=["chat"])
async def get_session(session_id: str, svc: Svc) -> SessionDetail:
    session = await _session(svc, session_id)
    messages = [
        MessageOut(**{k: m[k] for k in ("id", "role", "content", "meta", "created_at")})
        for m in await svc.store.messages(session_id)
    ]
    return SessionDetail(**{k: session[k] for k in SESSION_FIELDS}, messages=messages)


@router.delete("/sessions/{session_id}", status_code=204, tags=["chat"])
async def delete_session(session_id: str, svc: Svc) -> None:
    if not await svc.store.delete_session(session_id):
        raise NotFound(f"Session {session_id} does not exist.")


@router.post(
    "/sessions/{session_id}/messages",
    response_model=ChatOut,
    tags=["chat"],
    summary="Ask a question; the agent answers from the evidence",
)
async def chat(session_id: str, body: ChatRequest, svc: Svc) -> ChatOut:
    session = await _session(svc, session_id)
    pack = await svc.pack(session["analysis_id"])
    result = await svc.agent.chat(session, pack, body.content.strip())
    # Housekeeping after the response: fold old turns into the summary, keep durable memories.
    svc.spawn(svc.agent.short_term.compact(session_id), f"compact:{session_id}")
    if result.generated_from == "llm":
        svc.spawn(svc.agent.long_term.remember(session, body.content, result.answer), f"remember:{session_id}")
    return ChatOut(
        message_id=result.message_id,
        session_id=session_id,
        answer=result.answer,
        generated_from=result.generated_from,
        model=result.model,
        citations=result.citations,
        verification=result.verification,
        tool_calls=result.tool_calls,
        attempts=result.attempts,
        usage=result.usage,
        memories_used=result.memories_used,
        retry_after=result.retry_after,
    )


@router.get(
    "/memories",
    response_model=list[MemoryOut],
    tags=["memory"],
    summary="Long-term memories of a user and/or an analysis",
)
async def list_memories(svc: Svc, user_id: str | None = None, analysis_id: str | None = None) -> list[MemoryOut]:
    scopes = ([("user", user_id)] if user_id else []) + ([("analysis", analysis_id)] if analysis_id else [])
    return [MemoryOut(**{k: m[k] for k in MemoryOut.model_fields}) for m in await svc.store.memories(scopes)]


@router.delete("/memories/{memory_id}", status_code=204, tags=["memory"])
async def delete_memory(memory_id: int, svc: Svc) -> None:
    if not await svc.store.delete_memory(memory_id):
        raise NotFound(f"Memory {memory_id} does not exist.")


@router.delete("/users/{user_id}/memories", tags=["memory"], summary="Forget everything remembered about a user")
async def forget_user(user_id: str, svc: Svc) -> dict:
    return {"deleted": await svc.store.delete_memories("user", user_id)}
