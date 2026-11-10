class AgentError(Exception):
    """Base error carrying an HTTP status and a stable machine-readable code."""

    status = 500
    code = "internal_error"

    def __init__(self, message: str, *, details: dict | None = None, retry_after: float | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.retry_after = retry_after


class NotFound(AgentError):
    status = 404
    code = "not_found"


class InvalidInput(AgentError):
    status = 422
    code = "invalid_input"


class Unauthorized(AgentError):
    status = 401
    code = "unauthorized"


class RateLimitedClient(AgentError):
    status = 429
    code = "rate_limited"


class UpstreamError(AgentError):
    """The FIREBRIDGE data API failed or returned something unusable."""

    status = 502
    code = "upstream_error"


class AllModelsExhausted(AgentError):
    """Every model in the fallback chain is rate-limited or unavailable."""

    status = 503
    code = "models_exhausted"


class LLMConfigurationError(AgentError):
    status = 500
    code = "llm_configuration"
