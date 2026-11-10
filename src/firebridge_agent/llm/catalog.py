from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelSpec:
    """What the router needs to know about a model to decide whether a request can go to it."""

    id: str
    context_window: int
    max_output: int
    tools: bool
    json_mode: bool
    # Extra request parameters, e.g. to hide reasoning tokens from the answer.
    params: dict = field(default_factory=dict)
    # Whether the model may write user-facing answers; small models are kept for housekeeping.
    answers: bool = True


# Measured against the Groq API on 2026-11-10 (context windows from /models; parameters probed).
KNOWN_MODELS: dict[str, ModelSpec] = {
    "qwen/qwen3.8-27b": ModelSpec(
        "qwen/qwen3.8-27b",
        131_072,
        16_384,
        tools=True,
        json_mode=True,
        params={"reasoning_format": "hidden", "reasoning_effort": "none"},
    ),
    "openai/gpt-oss-120b": ModelSpec(
        "openai/gpt-oss-120b",
        131_072,
        65_536,
        tools=True,
        json_mode=True,
        params={"reasoning_format": "hidden", "reasoning_effort": "low"},
    ),
    "openai/gpt-oss-20b": ModelSpec(
        "openai/gpt-oss-20b",
        131_072,
        65_536,
        tools=True,
        json_mode=True,
        params={"reasoning_format": "hidden", "reasoning_effort": "low"},
    ),
    "allam-2-7b": ModelSpec("allam-2-7b", 4_096, 4_096, tools=False, json_mode=True, answers=False),
}


def spec_for(model_id: str) -> ModelSpec:
    """Known spec, or a conservative default for a model added through configuration."""
    return KNOWN_MODELS.get(model_id) or ModelSpec(model_id, 8_192, 4_096, tools=True, json_mode=False)
