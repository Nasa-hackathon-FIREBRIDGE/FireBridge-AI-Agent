import json
import re
from typing import Any

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Parse a JSON object from model output, tolerating code fences and surrounding prose."""
    cleaned = _FENCE.sub("", text or "").strip()
    for candidate in (cleaned, cleaned[cleaned.find("{") : cleaned.rfind("}") + 1] if "{" in cleaned else ""):
        if not candidate:
            continue
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None
