from typing import Any

from pydantic import BaseModel, Field

ANALYSIS_ID = r"^[A-Za-z0-9-]{8,64}$"
LANGUAGE = r"^[A-Za-z]{2,3}(-[A-Za-z0-9]{2,8})?$"


class IngestRequest(BaseModel):
    analysis_id: str = Field(pattern=ANALYSIS_ID, description="FIREBRIDGE analysis id.")
    report: dict[str, Any] | None = Field(
        None, description="Optional sensor-bridge export from the FIREBRIDGE frontend ('Full report (JSON)')."
    )
    use_api: bool = Field(True, description="Also load native records from the FIREBRIDGE API.")


class FactOut(BaseModel):
    id: str
    topic: str
    text: str
    priority: int
    refs: dict[str, list[Any]] = Field(
        default_factory=dict, description="Months, days, episodes and app sections the fact is about."
    )


class AnalysisOut(BaseModel):
    analysis_id: str
    title: str
    sources: list[str]
    period_start: str
    period_end_exclusive: str
    built_at: str
    facts: list[FactOut]


class SummaryRequest(BaseModel):
    language: str = Field("en", pattern=LANGUAGE)
    refresh: bool = False


class Finding(BaseModel):
    title: str
    detail: str
    facts: list[str]
    refs: dict[str, list[Any]] = Field(default_factory=dict)


class SummaryOut(BaseModel):
    analysis_id: str
    language: str
    headline: str
    findings: list[Finding]
    caveats: list[str]
    follow_up_questions: list[str]
    markdown: str
    generated_from: str = Field(description="'llm', or 'template' when every model was unavailable.")
    model: str | None
    verification: dict[str, Any]
    attempts: list[dict[str, Any]]
    usage: dict[str, int]
    cached: bool
    created_at: str
    retry_after: float | None = None


class SessionCreate(BaseModel):
    analysis_id: str = Field(pattern=ANALYSIS_ID)
    user_id: str | None = Field(None, max_length=128, description="Stable user identifier for long-term memory.")
    language: str = Field("en", pattern=LANGUAGE)


class SessionOut(BaseModel):
    id: str
    analysis_id: str
    user_id: str | None
    language: str
    rolling_summary: str
    created_at: str
    updated_at: str


class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    meta: dict[str, Any]
    created_at: str


class SessionDetail(SessionOut):
    messages: list[MessageOut]


class ChatContext(BaseModel):
    """Where the user is in the app, so "this month" or "here" can be understood."""

    view: str = Field("", max_length=40, pattern=r"^[a-z0-9-]*$")
    # One plain line: the label is placed in the system prompt.
    label: str = Field("", max_length=160, pattern=r"^[^\x00-\x1f\x7f<>]*$")
    month: str | None = Field(None, pattern=r"^\d{4}-\d{2}$")

    def describe(self) -> str:
        parts = [self.label or self.view.replace("-", " ")]
        if self.month:
            parts.append(f"focused on month {self.month}")
        return ", ".join(p for p in parts if p)


class ChatRequest(BaseModel):
    content: str = Field(min_length=1, max_length=4000)
    context: ChatContext | None = None


class ChatOut(BaseModel):
    message_id: int
    session_id: str
    answer: str
    generated_from: str = Field(
        description="'llm'; 'fallback' (relevant facts only) when every model was unavailable; or 'refusal' "
        "when the message was outside the analysis or tried to change the agent's rules."
    )
    model: str | None
    citations: list[dict[str, str]]
    verification: dict[str, Any]
    tool_calls: list[dict[str, Any]]
    attempts: list[dict[str, Any]]
    usage: dict[str, int]
    memories_used: list[str]
    references: dict[str, list[Any]] = Field(
        default_factory=dict, description="Months, days, episodes and app sections the answer is about."
    )
    retry_after: float | None = None
    refusal: str | None = Field(
        None,
        description="Why the message was refused: 'injection' (it tried to change the agent's rules; no model "
        "was called) or 'out_of_scope' (the model judged it unrelated to the analysis).",
    )


class MemoryOut(BaseModel):
    id: int
    scope: str
    scope_id: str
    kind: str
    content: str
    importance: float
    uses: int
    created_at: str
    last_used_at: str | None
