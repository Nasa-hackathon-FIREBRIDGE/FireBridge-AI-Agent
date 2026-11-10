"""Add the agent route to the existing HTTPS proxy, validating before reload."""

import os
import subprocess
from pathlib import Path

MARKER = "        location /api/ { proxy_pass http://backend; }"
ROUTE = """        # FireBridge AI Agent: Docker DNS is resolved at request time across releases.
        location = /agent { return 308 /agent/; }
        location ^~ /agent/ {
            resolver 127.0.0.11 valid=10s ipv6=off;
            set $agent_endpoint http://firebridge-ai-agent:8000;
            rewrite ^/agent/(.*)$ /$1 break;
            proxy_pass $agent_endpoint;
            proxy_read_timeout 180s;
            proxy_connect_timeout 5s;
            proxy_buffering off;
        }
"""


def add_route(content):
    if "location ^~ /agent/" in content:
        return content
    if content.count(MARKER) != 1:
        raise ValueError("Unexpected proxy configuration; review it before adding the agent route.")
    return content.replace(MARKER, ROUTE + MARKER)


def install():
    if os.geteuid() != 0:
        raise RuntimeError("Run on the VPS as an administrator.")
    root = Path("/opt/firebridge/webapp")
    config = root / "nginx.proxy.conf"
    original = config.read_text()
    updated = add_route(original)
    if updated == original:
        return
    backup = root / "nginx.proxy.before-agent.conf"
    if not backup.exists():
        backup.write_text(original)
        backup.chmod(0o600)
    # Write in place: Nginx has a bind mount of this file's inode.
    config.write_text(updated)
    command = [
        "docker",
        "compose",
        "-f",
        str(root / "compose.yaml"),
        "--env-file",
        str(root / ".env"),
        "exec",
        "-T",
        "proxy",
        "nginx",
    ]
    try:
        subprocess.run([*command, "-t"], cwd=root, check=True)
        subprocess.run([*command, "-s", "reload"], cwd=root, check=True)
    except Exception:
        config.write_text(original)
        subprocess.run([*command, "-s", "reload"], cwd=root, check=False)
        raise
    print("HTTPS /agent/ route installed and proxy reloaded.")


if __name__ == "__main__":
    install()
