"""Minimal client: load an analysis, read its summary, then chat with the agent.

uv run python examples/client.py                      # northern Morocco sample, English
uv run python examples/client.py --language fr --report examples/bridge_report_northern_morocco.json
"""

import argparse
import json
from pathlib import Path

import httpx

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://localhost:8000")
parser.add_argument("--analysis", default="13ad2e0c-cd43-4ee9-bcc6-fd8b273beb70")
parser.add_argument("--report", type=Path, default=Path(__file__).with_name("bridge_report_northern_morocco.json"))
parser.add_argument("--language", default="en")
parser.add_argument("--user", default="example-user")
parser.add_argument("--api-key", default=None)
args = parser.parse_args()

http = httpx.Client(base_url=args.url, timeout=120, headers={"x-api-key": args.api_key} if args.api_key else {})


def call(method: str, path: str, **kwargs) -> dict:
    response = http.request(method, path, **kwargs)
    if response.status_code >= 400:
        raise SystemExit(f"{response.status_code}: {response.text}")
    return response.json()


report = json.loads(args.report.read_text()) if args.report and args.report.exists() else None
analysis = call("POST", "/v1/analyses", json={"analysis_id": args.analysis, "report": report})
print(f"Loaded {analysis['title']} ({len(analysis['facts'])} evidence facts from {', '.join(analysis['sources'])})\n")

summary = call("POST", f"/v1/analyses/{args.analysis}/summary", json={"language": args.language})
print(
    summary["markdown"],
    f"\n\n[{summary['generated_from']} · {summary['model']} · numbers {summary['verification']['status']}]\n",
)

session = call(
    "POST", "/v1/sessions", json={"analysis_id": args.analysis, "user_id": args.user, "language": args.language}
)
print("Ask anything about this analysis (empty line to quit).")
while question := input("\nyou> ").strip():
    answer = call("POST", f"/v1/sessions/{session['id']}/messages", json={"content": question})
    tools = ", ".join(t["name"] for t in answer["tool_calls"]) or "none"
    print(
        f"\nagent> {answer['answer']}\n[{answer['model']} · tools: {tools} · numbers {answer['verification']['status']}]"
    )
