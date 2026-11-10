from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration from environment variables (prefix FBA_) and an optional .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="FBA_", extra="ignore")

    # The project variable wins over a generic GROQ_API_KEY that may already be exported in the shell.
    project_groq_api_key: SecretStr = Field(default=SecretStr(""), validation_alias="FBA_GROQ_API_KEY")
    generic_groq_api_key: SecretStr = Field(default=SecretStr(""), validation_alias="GROQ_API_KEY")
    groq_base_url: str = "https://api.groq.com/openai/v1"
    models: list[str] = ["qwen/qwen3.8-27b", "openai/gpt-oss-120b", "openai/gpt-oss-20b", "allam-2-7b"]
    llm_timeout_s: float = 60.0
    # Longest the router waits for a cooling-down model before giving up with 503.
    max_wait_for_model_s: float = 20.0

    firebridge_api_url: str = "https://astramind.solutions/api/v1"
    firebridge_timeout_s: float = 60.0

    database_path: Path = Path("data/agent.db")

    service_api_keys: list[SecretStr] = []
    rate_limit_per_minute: int = 30
    cors_origins: list[str] = []

    max_tool_steps: int = 4
    answer_max_tokens: int = 900
    summary_max_tokens: int = 1400
    digest_token_budget: int = 1800
    chat_digest_token_budget: int = 1200
    short_term_turns: int = 8
    short_term_token_budget: int = 1600
    long_term_top_k: int = 5
    memory_extraction: bool = True
    # Conversation compaction and memory extraction try these first to spare the main chain.
    housekeeping_models: list[str] = ["openai/gpt-oss-20b", "allam-2-7b"]

    environment: Literal["development", "production"] = "development"
    release_revision: str = "development"
    root_path: str = ""

    log_level: str = "INFO"
    log_json: bool = True

    @property
    def groq_api_key(self) -> SecretStr:
        return self.project_groq_api_key if self.project_groq_api_key.get_secret_value() else self.generic_groq_api_key


@lru_cache
def get_settings() -> Settings:
    return Settings()
