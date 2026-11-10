from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass
class LLMRequest:
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None = None
    max_tokens: int = 800
    temperature: float = 0.2
    json_mode: bool = False
    # Purpose label for metrics and logs: "summary", "chat", "compact", "memory"...
    purpose: str = "chat"
    # Models to try first, in order (housekeeping calls that must not starve the main chain);
    # the rest of the chain follows as usual.
    prefer: tuple[str, ...] = ()
    # Only these models (e.g. a repair must come from the model that wrote the answer).
    restrict_to: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    # False for housekeeping (compaction, memory): may use models reserved for it.
    user_facing: bool = True
    # Overrides the router's default wait for a cooling-down model.
    max_wait_s: float | None = None


@dataclass
class RateState:
    limit_tokens: int | None = None
    remaining_requests: int | None = None
    remaining_tokens: int | None = None
    reset_requests_s: float | None = None
    reset_tokens_s: float | None = None
    observed_at: float = 0.0


@dataclass
class LLMResponse:
    model: str
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    rate: RateState | None = None
    # Every attempt made before this answer: [{"model", "outcome", "detail"}].
    attempts: list[dict[str, Any]] = field(default_factory=list)
