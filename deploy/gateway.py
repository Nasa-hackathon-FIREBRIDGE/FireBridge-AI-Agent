#!/usr/bin/python3
"""Root-owned, image-only deployment gateway for the forced-command SSH key."""

import fcntl
import hashlib
import json
import os
import re
import shlex
import sqlite3
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path("/opt/firebridge/agent")
SOURCE = "https://github.com/Nasa-hackathon-FIREBRIDGE/FireBridge-AI-Agent"
MAX_BYTES = 1024**3
REVISION = re.compile(r"[a-f0-9]{40}")


def parse_command(command):
    args = shlex.split(command)
    if args == ["status"]:
        return args
    if len(args) not in (2, 3) or not REVISION.fullmatch(args[1]):
        raise ValueError("Invalid revision or command.")
    if args[0] == "image" and len(args) == 3 and re.fullmatch(r"[a-f0-9]{64}", args[2]):
        return args
    if args[0] in ("activate", "rollback") and len(args) == 2:
        return args
    raise ValueError("Unsupported deployment command.")


def image(revision):
    return f"firebridge-ai-agent:{revision}"


def run(*command, timeout=240):
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        with (ROOT / "operations.log").open("a") as log:
            log.write(f"{datetime.now(UTC).isoformat()} {command[0]} failed\n")
            log.write(result.stdout + result.stderr)
        raise RuntimeError("Deployment command failed; inspect the private operations.log on the server.")
    return result.stdout.strip()


def compose(*args):
    return run("docker", "compose", "--env-file", str(ROOT / ".env"), "-f", str(ROOT / "compose.yaml"), *args)


def verify(revision):
    info = json.loads(run("docker", "image", "inspect", image(revision)))[0]
    labels = info["Config"].get("Labels") or {}
    if labels.get("org.opencontainers.image.source") != SOURCE:
        raise ValueError("Image source does not match the agent repository.")
    if labels.get("org.opencontainers.image.revision") != revision:
        raise ValueError("Image revision mismatch.")
    if info["Config"].get("User") not in ("agent", "10001", "10001:10001"):
        raise ValueError("The agent image must run as its unprivileged user.")
    return info["Id"]


def receive(revision, expected, stream):
    archive = ROOT / "incoming.tar.gz"
    digest = hashlib.sha256()
    size = 0
    try:
        with archive.open("wb") as output:
            archive.chmod(0o600)
            while chunk := stream.read(1024**2):
                size += len(chunk)
                if size > MAX_BYTES:
                    raise ValueError("Image archive exceeds 1 GiB.")
                digest.update(chunk)
                output.write(chunk)
        if digest.hexdigest() != expected:
            raise ValueError("Image archive checksum mismatch.")
        run("docker", "load", "--input", str(archive), timeout=600)
        verify(revision)
    finally:
        archive.unlink(missing_ok=True)


def write_env(content):
    temporary = ROOT / ".env.next"
    temporary.write_text(content)
    temporary.chmod(0o600)
    temporary.replace(ROOT / ".env")


def backup_database():
    database = ROOT / "data/agent.db"
    if not database.exists():
        return None
    directory = ROOT / "backups"
    directory.mkdir(mode=0o700, exist_ok=True)
    target = directory / (datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + ".db")
    # SQLite's online backup includes committed WAL content while the old app runs.
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as source, sqlite3.connect(target) as dest:
        source.backup(dest)
        if dest.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("Database backup integrity check failed.")
    target.chmod(0o600)
    return target.name


def check_runtime(revision):
    # Check the running image, readiness payload and both sides of the auth boundary.
    program = """import json, os, urllib.request, urllib.error
base = 'http://127.0.0.1:8000'
with urllib.request.urlopen(base + '/ready', timeout=5) as r:
    ready = json.load(r)
assert ready['status'] == 'ready' and ready['revision'] == os.environ['FBA_RELEASE_REVISION']
try:
    urllib.request.urlopen(base + '/v1/analyses', timeout=5)
    raise AssertionError('Unauthenticated API is accessible')
except urllib.error.HTTPError as error:
    assert error.code == 401
key = json.loads(os.environ['FBA_SERVICE_API_KEYS'])[0]
request = urllib.request.Request(base + '/v1/analyses', headers={'X-API-Key': key})
with urllib.request.urlopen(request, timeout=5) as r:
    assert r.status == 200
print(ready['revision'])
"""
    actual = compose("exec", "-T", "agent", "python", "-c", program)
    if actual != revision:
        raise RuntimeError("Running revision differs from the selected release.")


def activate(revision):
    image_id = verify(revision)
    env = ROOT / ".env"
    original = env.read_text() if env.exists() else None
    receipt_path = ROOT / "deployment.json"
    previous = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    backup = backup_database()
    write_env(f"AGENT_IMAGE={image(revision)}\n")
    try:
        compose("config", "--quiet")
        compose("up", "-d", "--wait", "--wait-timeout", "120")
        check_runtime(revision)
    except Exception:
        if original:
            write_env(original)
            compose("up", "-d", "--wait", "--wait-timeout", "120")
            if previous.get("revision"):
                check_runtime(previous["revision"])
        else:
            compose("down")  # Persistent data stays on disk, including on first-deploy failure.
            env.unlink(missing_ok=True)
        raise
    receipt = {
        "state": "healthy",
        "revision": revision,
        "image_id": image_id,
        "previous_revision": previous.get("revision"),
        "database_backup": backup,
        "deployed_at": datetime.now(UTC).isoformat(),
        "url": "https://astramind.solutions/agent",
    }
    temporary = receipt_path.with_suffix(".next")
    temporary.write_text(json.dumps(receipt, indent=2) + "\n")
    temporary.chmod(0o600)
    temporary.replace(receipt_path)
    # Bound database backup retention. Never prune shared Docker images or volumes.
    for old in sorted((ROOT / "backups").glob("*.db"))[:-14]:
        old.unlink()
    print(json.dumps(receipt))


def main():
    if os.geteuid() != 0:
        raise RuntimeError("Use the restricted sudo gateway.")
    os.umask(0o077)
    args = parse_command(os.environ.get("SSH_ORIGINAL_COMMAND", ""))
    ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (ROOT / "deployment.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args == ["status"]:
            receipt = ROOT / "deployment.json"
            print(receipt.read_text() if receipt.exists() else '{"state":"not_deployed"}')
        elif args[0] == "image":
            receive(args[1], args[2], sys.stdin.buffer)
            print(json.dumps({"image": "verified", "revision": args[1]}))
        else:
            activate(args[1])


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Agent deployment failed: {error}", file=sys.stderr)
        sys.exit(1)
