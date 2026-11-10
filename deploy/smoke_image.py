"""Exercise the production image, auth and persisted SQLite data without provider calls."""

import argparse
import json
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path


def smoke(image, revision):
    suffix = uuid.uuid4().hex[:12]
    name, volume = f"agent-ci-{suffix}", f"agent-ci-data-{suffix}"
    base = None

    def docker(*args):
        return subprocess.run(["docker", *args], capture_output=True, text=True, check=True).stdout.strip()

    def request(path, data=None, *, authorized=True):
        headers = {"X-API-Key": "ci-only-service-key"} if authorized else {}
        if data is not None:
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(
            base + path, data=json.dumps(data).encode() if data is not None else None, headers=headers
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            return json.load(response)

    def wait():
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            try:
                ready = request("/ready")
                assert ready["status"] == "ready" and ready["revision"] == revision
                return
            except (OSError, AssertionError):
                time.sleep(1)
        raise RuntimeError("Production container did not become ready.")

    try:
        docker("volume", "create", volume)
        docker(
            "run",
            "--rm",
            "--user",
            "0:0",
            "--entrypoint",
            "chown",
            "-v",
            volume + ":/app/data",
            image,
            "10001:10001",
            "/app/data",
        )
        docker(
            "run",
            "-d",
            "--name",
            name,
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges:true",
            "--tmpfs",
            "/tmp:size=64m",
            "--memory",
            "512m",
            "--cpus",
            "1",
            "--pids-limit",
            "128",
            "-p",
            "127.0.0.1::8000",
            "-v",
            volume + ":/app/data",
            "-e",
            "FBA_ENVIRONMENT=production",
            "-e",
            "FBA_GROQ_API_KEY=ci-only-not-a-provider-key",
            "-e",
            'FBA_SERVICE_API_KEYS=["ci-only-service-key"]',
            "-e",
            "FBA_MEMORY_EXTRACTION=false",
            image,
        )
        port = docker("port", name, "8000/tcp").split(":")[-1]
        base = "http://127.0.0.1:" + port
        wait()
        try:
            request("/v1/analyses", authorized=False)
            raise AssertionError("The production API accepted an unauthenticated request.")
        except urllib.error.HTTPError as error:
            assert error.code == 401
        fixture = json.loads(
            (Path(__file__).resolve().parents[1] / "examples/bridge_report_northern_morocco.json").read_text()
        )
        report = fixture.get("report", fixture)
        analysis_id = fixture.get("analysis_id", "ci-persistence-analysis")
        request(
            "/v1/analyses",
            {"analysis_id": analysis_id, "report": fixture if "report" in fixture else report, "use_api": False},
        )
        docker("restart", name)
        # Docker Desktop can assign a new ephemeral host port on restart.
        port = docker("port", name, "8000/tcp").split(":")[-1]
        base = "http://127.0.0.1:" + port
        wait()
        assert request("/v1/analyses/" + analysis_id)["analysis_id"] == analysis_id
        print("Production image passed readiness, authentication and restart persistence checks.")
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        subprocess.run(["docker", "volume", "rm", volume], capture_output=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    smoke(args.image, args.revision)
