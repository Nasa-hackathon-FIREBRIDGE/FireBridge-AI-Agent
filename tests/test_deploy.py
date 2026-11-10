"""Deployment failure paths use temporary files, never a real Docker daemon."""

import hashlib
import importlib.util
import io
import json
import sqlite3
from pathlib import Path

import pytest

from firebridge_agent.main import create_app


def load(name):
    path = Path(__file__).resolve().parents[1] / "deploy" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def gateway(tmp_path):
    module = load("gateway")
    module.ROOT = tmp_path
    return module


@pytest.mark.parametrize(
    "command",
    [
        "bash",
        "activate main",
        "activate " + "a" * 40 + " extra",
        "image " + "a" * 40 + " wrong",
        "status; id",
        "rollback ../../etc",
    ],
)
def test_gateway_rejects_shell_commands(gateway, command):
    with pytest.raises(ValueError):
        gateway.parse_command(command)


def test_gateway_only_accepts_exact_protocol(gateway):
    assert gateway.parse_command("status") == ["status"]
    assert gateway.parse_command("activate " + "a" * 40) == ["activate", "a" * 40]
    assert gateway.parse_command("image " + "a" * 40 + " " + "b" * 64)[0] == "image"


def test_bad_checksum_never_loads_docker(gateway, monkeypatch):
    monkeypatch.setattr(gateway, "run", lambda *a, **kw: pytest.fail("Docker must not run"))
    with pytest.raises(ValueError, match="checksum"):
        gateway.receive("a" * 40, "b" * 64, io.BytesIO(b"not-an-image"))
    assert not (gateway.ROOT / "incoming.tar.gz").exists()


def test_oversized_transfer_is_removed(gateway, monkeypatch):
    monkeypatch.setattr(gateway, "MAX_BYTES", 3)
    with pytest.raises(ValueError, match="exceeds"):
        gateway.receive("a" * 40, hashlib.sha256(b"1234").hexdigest(), io.BytesIO(b"1234"))
    assert not (gateway.ROOT / "incoming.tar.gz").exists()


def test_backup_includes_wal_and_has_integrity(gateway):
    (gateway.ROOT / "data").mkdir()
    connection = sqlite3.connect(gateway.ROOT / "data/agent.db")
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("CREATE TABLE evidence (id INTEGER)")
        connection.execute("INSERT INTO evidence VALUES (42)")
        connection.commit()
        name = gateway.backup_database()
        with sqlite3.connect(gateway.ROOT / "backups" / name) as backup:
            assert backup.execute("SELECT id FROM evidence").fetchone() == (42,)
    finally:
        connection.close()


def test_failed_activation_restores_previous_release(gateway, monkeypatch):
    original = "AGENT_IMAGE=firebridge-ai-agent:" + "a" * 40 + "\n"
    (gateway.ROOT / ".env").write_text(original)
    receipt = {"state": "healthy", "revision": "a" * 40}
    (gateway.ROOT / "deployment.json").write_text(json.dumps(receipt))
    monkeypatch.setattr(gateway, "verify", lambda sha: "sha256:verified")
    checked, calls = [], []

    def compose(*args):
        calls.append(args)
        if args[0] == "up" and len([c for c in calls if c[0] == "up"]) == 1:
            raise RuntimeError("Candidate readiness failed")

    monkeypatch.setattr(gateway, "compose", compose)
    monkeypatch.setattr(gateway, "check_runtime", checked.append)
    with pytest.raises(RuntimeError, match="Candidate"):
        gateway.activate("b" * 40)
    assert (gateway.ROOT / ".env").read_text() == original
    assert json.loads((gateway.ROOT / "deployment.json").read_text()) == receipt
    assert checked == ["a" * 40]
    assert len([c for c in calls if c[0] == "up"]) == 2


def test_first_activation_failure_keeps_data_and_stops_candidate(gateway, monkeypatch):
    monkeypatch.setattr(gateway, "verify", lambda sha: "sha256:verified")
    calls = []

    def compose(*args):
        calls.append(args)
        if args[0] == "up":
            raise RuntimeError("Candidate failed")

    monkeypatch.setattr(gateway, "compose", compose)
    with pytest.raises(RuntimeError):
        gateway.activate("b" * 40)
    assert ("down",) in calls
    assert not (gateway.ROOT / ".env").exists()
    assert not (gateway.ROOT / "deployment.json").exists()


def test_ingress_is_idempotent_and_preserves_other_routes():
    ingress = load("install_ingress")
    content = "before\n" + ingress.MARKER + "\nafter\n"
    updated = ingress.add_route(content)
    assert ingress.MARKER in updated and updated.startswith("before\n") and updated.endswith("after\n")
    assert ingress.add_route(updated) == updated
    with pytest.raises(ValueError):
        ingress.add_route("unexpected nginx configuration")


@pytest.mark.parametrize(
    "field,value,reason",
    [("service_api_keys", [], "SERVICE_API_KEYS"), ("FBA_GROQ_API_KEY", "", "GROQ_API_KEY"), ("models", [], "model")],
)
def test_production_refuses_missing_credentials(settings, monkeypatch, field, value, reason):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    from firebridge_agent.config import Settings

    values = {
        **settings.model_dump(),
        "FBA_GROQ_API_KEY": "test",
        "environment": "production",
        "service_api_keys": ["test"],
    }
    values[field] = value
    with pytest.raises(ValueError, match=reason):
        create_app(Settings(_env_file=None, **values))


def test_failed_database_is_not_ready(client, monkeypatch):
    async def broken():
        raise RuntimeError("Database unavailable")

    monkeypatch.setattr(client.app.state.services.store, "ping", broken)
    response = client.get("/ready")
    assert response.status_code == 503 and response.json()["database"] is False
    assert client.get("/health").status_code == 200
