import pytest
from fastapi.testclient import TestClient

from firebridge_agent.config import Settings
from firebridge_agent.data.firebridge import FireBridgeClient
from firebridge_agent.llm.router import ModelRouter
from firebridge_agent.main import create_app
from firebridge_agent.services import Services

from .fakes import ScriptedLLM, agent_handler, firebridge_transport

MODELS = ["qwen/qwen3.8-27b", "openai/gpt-oss-120b", "openai/gpt-oss-20b", "allam-2-7b"]


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        FBA_GROQ_API_KEY="test-key",
        database_path=tmp_path / "agent.db",
        models=MODELS,
        rate_limit_per_minute=1000,
        log_json=False,
        max_wait_for_model_s=0.0,
    )


@pytest.fixture
def llm() -> ScriptedLLM:
    return ScriptedLLM(agent_handler)


@pytest.fixture
def client(settings, llm):
    async def factory(s: Settings) -> Services:
        async def no_sleep(_: float) -> None:
            return None

        router = ModelRouter(llm, s.models, max_wait_s=s.max_wait_for_model_s, sleep=no_sleep)
        return await Services.start(
            s,
            router=router,
            firebridge=FireBridgeClient("http://firebridge.test/api/v1", 5, transport=firebridge_transport()),
        )

    with TestClient(create_app(settings, services_factory=factory)) as test_client:
        yield test_client
