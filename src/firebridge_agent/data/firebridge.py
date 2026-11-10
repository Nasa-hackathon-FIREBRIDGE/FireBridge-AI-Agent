"""Async client for the public FIREBRIDGE API (read-only)."""

from typing import Any

import httpx

from ..errors import NotFound, UpstreamError


class FireBridgeClient:
    def __init__(self, base_url: str, timeout_s: float, transport: httpx.AsyncBaseTransport | None = None):
        self._http = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout_s, connect=10.0),
            headers={"User-Agent": "FireBridge-AI-Agent/0.1"},
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _get(self, path: str) -> dict[str, Any]:
        try:
            response = await self._http.get(path)
        except httpx.HTTPError as exc:
            raise UpstreamError(f"FIREBRIDGE API unreachable: {type(exc).__name__}", details={"path": path}) from exc
        if response.status_code == 404:
            raise NotFound(f"FIREBRIDGE has no resource at {path}.", details={"path": path})
        if response.status_code >= 400:
            raise UpstreamError(
                f"FIREBRIDGE API returned {response.status_code}.", details={"path": path, "body": response.text[:300]}
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise UpstreamError("FIREBRIDGE API returned invalid JSON.", details={"path": path}) from exc
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, dict):
            raise UpstreamError("FIREBRIDGE API response has no data object.", details={"path": path})
        return data

    async def analysis_bundle(self, analysis_id: str) -> dict[str, Any]:
        """Everything the evidence builder needs for one saved analysis."""
        analysis = await self._get(f"/analyses/{analysis_id}")
        if analysis.get("status") not in ("ready", "partial"):
            raise UpstreamError(
                f"Analysis {analysis_id} is not ready (status: {analysis.get('status')}).",
                details={"status": analysis.get("status")},
            )
        summary = await self._get(f"/analyses/{analysis_id}/summary")
        calendar = await self._get(f"/analyses/{analysis_id}/calendar")
        area = None
        if analysis.get("area_id"):
            try:
                area = await self._get(f"/areas/{analysis['area_id']}")
            except (NotFound, UpstreamError):
                area = None
        return {"analysis": analysis, "summary": summary, "calendar": calendar, "area": area}
