import secrets
import time
from collections import defaultdict, deque

from fastapi import Request

from ..errors import RateLimitedClient, Unauthorized
from ..services import Services


def services(request: Request) -> Services:
    return request.app.state.services


def client_key(request: Request) -> str:
    key = request.headers.get("x-api-key")
    return f"key:{key[:12]}" if key else f"ip:{request.client.host if request.client else 'unknown'}"


async def require_api_key(request: Request) -> None:
    """Enforced only when FBA_SERVICE_API_KEYS is set (local development runs open)."""
    keys = request.app.state.services.settings.service_api_keys
    if not keys:
        return
    given = request.headers.get("x-api-key", "")
    if not any(secrets.compare_digest(given, k.get_secret_value()) for k in keys):
        raise Unauthorized("Missing or invalid X-API-Key header.")


class SlidingWindowLimiter:
    """Per-client request limit over a rolling 60-second window (in memory, per process)."""

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str) -> None:
        if self.per_minute <= 0:
            return
        now = time.monotonic()
        hits = self._hits[key]
        while hits and now - hits[0] > 60:
            hits.popleft()
        if len(hits) >= self.per_minute:
            raise RateLimitedClient(
                f"More than {self.per_minute} requests per minute.", retry_after=round(60 - (now - hits[0]), 1)
            )
        hits.append(now)
