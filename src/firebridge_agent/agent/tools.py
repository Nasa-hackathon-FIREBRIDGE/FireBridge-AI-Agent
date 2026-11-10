"""Deterministic tools the agent calls to read the evidence tables. No tool computes anything the
language model then has to trust blindly: every output is a direct lookup or a simple sum."""

import json
import re
from collections.abc import Callable
from typing import Any

from ..data.evidence import MONTH_NAMES, EvidencePack, month_name
from ..data.satellites import SATELLITES, label
from ..memory.bm25 import bm25_scores

_MONTHS = (
    {name.lower(): i + 1 for i, name in enumerate(MONTH_NAMES)}
    | {name[:3].lower(): i + 1 for i, name in enumerate(MONTH_NAMES)}
    | {"sept": 9}
)


def parse_month(text: str) -> str | None:
    """'2022-07', '2022/7', 'July 2022', 'Jul 2022', '07/2022' → '2022-07'."""
    t = text.strip().lower()
    if m := re.fullmatch(r"(\d{4})[-/.](\d{1,2})(?:[-/.]\d{1,2})?", t):
        return f"{m[1]}-{int(m[2]):02d}" if 1 <= int(m[2]) <= 12 else None
    if m := re.fullmatch(r"(\d{1,2})[-/.](\d{4})", t):
        return f"{m[2]}-{int(m[1]):02d}" if 1 <= int(m[1]) <= 12 else None
    if m := re.fullmatch(r"([a-z]+)\.?,?\s+(\d{4})", t):
        number = _MONTHS.get(m[1])
        return f"{m[2]}-{number:02d}" if number else None
    return None


def parse_day(text: str) -> str | None:
    t = text.strip().lower()
    if m := re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", t):
        return f"{m[1]}-{int(m[2]):02d}-{int(m[3]):02d}"
    if m := re.fullmatch(r"(\d{1,2})\s+([a-z]+)\.?,?\s+(\d{4})", t):
        number = _MONTHS.get(m[2])
        return f"{m[3]}-{number:02d}-{int(m[1]):02d}" if number else None
    if m := re.fullmatch(r"([a-z]+)\.?\s+(\d{1,2}),?\s+(\d{4})", t):
        number = _MONTHS.get(m[1])
        return f"{m[3]}-{number:02d}-{int(m[2]):02d}" if number else None
    return None


def _period(pack: EvidencePack, text: str) -> tuple[str, str, dict[str, Any]] | None:
    """A year or a month → (kind, key, record)."""
    t = text.strip()
    if re.fullmatch(r"\d{4}", t):
        return ("year", t, pack.years[t]) if t in pack.years else None
    month = parse_month(t)
    if month and month in pack.months:
        return "month", month, pack.months[month]
    return None


def _sensors(by_sensor: dict[str, int]) -> dict[str, int]:
    return {label(k): v for k, v in sorted(by_sensor.items(), key=lambda kv: -kv[1])}


def _change(a: float | None, b: float | None) -> str | None:
    if a is None or b is None or not a:
        return None
    return f"{(b - a) / a * 100:+.0f}%"


def _jury_view(jury: dict[str, Any] | None) -> dict[str, Any] | None:
    if not jury:
        return None
    return {
        "verdict": jury["kind"] or "within the usual range",
        "outlier_satellite": label(jury["outlier"]) if jury.get("outlier") else None,
        "harmonized_vs_usual": jury.get("harmonized"),
        "median_satellite_z": jury.get("consensus_z"),
        "satellites_agreeing": f"{jury.get('agreeing')} of {jury.get('testifying')}"
        if jury.get("testifying")
        else None,
        "each_satellite": {
            f"{label(k)} ({SATELLITES[k].instrument if k in SATELLITES else 'unknown'})": v
            for k, v in jury.get("witnesses", {}).items()
        },
    }


class Toolbox:
    def __init__(self, pack: EvidencePack):
        self.pack = pack
        self._handlers: dict[str, Callable[[dict[str, Any]], Any]] = {
            "get_month": self.get_month,
            "get_day": self.get_day,
            "compare_periods": self.compare_periods,
            "top_months": self.top_months,
            "unusual_months": self.unusual_months,
            "largest_episodes": self.largest_episodes,
            "handover_rehearsals": self.handover_rehearsals,
            "search_facts": self.search_facts,
        }

    def schemas(self) -> list[dict[str, Any]]:
        def fn(
            name: str, description: str, properties: dict[str, Any] | None = None, required: list[str] | None = None
        ) -> dict[str, Any]:
            return {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": {"type": "object", "properties": properties or {}, "required": required or []},
                },
            }

        month = {"type": "string", "description": "Month as YYYY-MM, e.g. 2022-07"}
        tools = [
            fn(
                "get_month",
                "Detections by satellite for one month, plus the harmonized value and the satellite-jury verdict when available.",
                {"month": month},
                ["month"],
            ),
            fn(
                "get_day",
                "Detections by satellite on one day (YYYY-MM-DD), including satellites whose record is missing that day.",
                {"date": {"type": "string"}},
                ["date"],
            ),
            fn(
                "compare_periods",
                "Compare two years (YYYY) or two months (YYYY-MM): totals by satellite, changes, satellites present in only one, harmonized change.",
                {"first": {"type": "string"}, "second": {"type": "string"}},
                ["first", "second"],
            ),
            fn(
                "top_months",
                "Busiest months by native detections or by harmonized activity.",
                {"metric": {"type": "string", "enum": ["native", "harmonized"]}, "limit": {"type": "integer"}},
            ),
            fn("search_facts", "Search the evidence facts by keywords.", {"query": {"type": "string"}}, ["query"]),
        ]
        if any(m.get("jury") for m in self.pack.months.values()):
            tools.append(
                fn(
                    "unusual_months",
                    "Months the satellite jury flags: unusual, disputed, or raw-total artifacts.",
                    {"kind": {"type": "string", "enum": ["unusual", "disputed", "artifact", "all"]}},
                )
            )
        if self.pack.episodes:
            tools.append(
                fn(
                    "largest_episodes",
                    "Largest detection clusters (multi-day fire episodes) with dates, size and satellites.",
                    {"limit": {"type": "integer"}},
                )
            )
        if self.pack.rehearsals:
            tools.append(
                fn(
                    "handover_rehearsals",
                    "Results of switching Aqua off in past years: un-harmonized vs FIREBRIDGE continuation and false alarms.",
                )
            )
        return tools

    def run(self, name: str, arguments: str) -> str:
        handler = self._handlers.get(name)
        if not handler:
            return json.dumps({"error": f"unknown tool {name}"})
        try:
            args = json.loads(arguments or "{}")
            if not isinstance(args, dict):
                raise ValueError("arguments must be an object")
        except (json.JSONDecodeError, ValueError) as exc:
            return json.dumps({"error": f"invalid arguments: {exc}"})
        try:
            result = handler(args)
        except (KeyError, TypeError, ValueError) as exc:
            result = {"error": f"{type(exc).__name__}: {exc}"}
        return json.dumps(result, ensure_ascii=False, separators=(",", ":"), default=str)

    # ── handlers ─────────────────────────────────────────────────────────────
    def get_month(self, args: dict[str, Any]) -> dict[str, Any]:
        month = parse_month(str(args.get("month", "")))
        record = self.pack.months.get(month or "")
        if not record:
            return {
                "error": f"no data for month {args.get('month')!r}",
                "available": f"{min(self.pack.months)} to {max(self.pack.months)}" if self.pack.months else None,
            }
        return {
            "month": month_name(month),
            "native_detections_all_satellites": record.get("total"),
            "by_satellite": _sensors(record.get("by_sensor") or {}),
            "days_without_any_record": record.get("days_without_record"),
            "harmonized": record.get("harmonized"),
            "jury": _jury_view(record.get("jury")),
        }

    def get_day(self, args: dict[str, Any]) -> dict[str, Any]:
        day = parse_day(str(args.get("date", "")))
        cell = self.pack.days.get(day or "")
        if not cell:
            return {
                "error": f"no daily record for {args.get('date')!r}"
                + ("" if self.pack.days else " (daily data not loaded)")
            }
        return {
            "date": day,
            "total": cell["total"],
            "by_satellite": _sensors(cell["by_sensor"]),
            "satellites_without_record": [label(k) for k in cell["missing"]],
        }

    def compare_periods(self, args: dict[str, Any]) -> dict[str, Any]:
        first, second = _period(self.pack, str(args.get("first", ""))), _period(self.pack, str(args.get("second", "")))
        if not first or not second:
            return {
                "error": "give two years (YYYY) or two months (YYYY-MM) inside the analysis",
                "years": list(self.pack.years),
            }
        (_, ka, a), (_, kb, b) = first, second
        sa, sb = a.get("by_sensor") or {}, b.get("by_sensor") or {}
        out: dict[str, Any] = {
            "first": ka,
            "second": kb,
            "native_total": {ka: a.get("total"), kb: b.get("total"), "change": _change(a.get("total"), b.get("total"))},
            "by_satellite": {
                label(k): {ka: sa.get(k, 0), kb: sb.get(k, 0), "change": _change(sa.get(k), sb.get(k))}
                for k in sorted(set(sa) | set(sb))
            },
            "only_in_first": [label(k) for k in sa if sa[k] and not sb.get(k)],
            "only_in_second": [label(k) for k in sb if sb[k] and not sa.get(k)],
            "partial": {ka: a.get("partial", False), kb: b.get("partial", False)},
        }
        ha, hb = a.get("harmonized"), b.get("harmonized")
        if ha and hb:
            out["harmonized"] = {ka: ha, kb: hb, "change": _change(ha["value"], hb["value"])}
        if out["only_in_first"] or out["only_in_second"]:
            out["note"] = (
                "The satellite fleet differs between the periods, so the raw totals are not directly comparable."
            )
        return out

    def top_months(self, args: dict[str, Any]) -> dict[str, Any]:
        metric = args.get("metric") if args.get("metric") in ("native", "harmonized") else "native"
        limit = max(1, min(12, int(args.get("limit") or 5)))
        value = (
            (lambda m: (m.get("harmonized") or {}).get("value"))
            if metric == "harmonized"
            else (lambda m: m.get("total"))
        )
        ranked = sorted(
            ((k, value(m)) for k, m in self.pack.months.items() if value(m) is not None), key=lambda kv: -kv[1]
        )
        return {"metric": metric, "months": [{"month": month_name(k), "value": v} for k, v in ranked[:limit]]}

    def unusual_months(self, args: dict[str, Any]) -> dict[str, Any]:
        kind = args.get("kind") or "all"
        wanted = {"unusual": ("unusual-high", "unusual-low"), "disputed": ("disputed",), "artifact": ("artifact",)}.get(
            kind, ("unusual-high", "unusual-low", "disputed", "artifact")
        )
        rows = []
        for month, record in sorted(self.pack.months.items()):
            jury = record.get("jury")
            if jury and jury.get("kind") in wanted:
                rows.append({"month": month_name(month), **(_jury_view(jury) or {})})
        return {"kind": kind, "months": rows}

    def largest_episodes(self, args: dict[str, Any]) -> dict[str, Any]:
        limit = max(1, min(8, int(args.get("limit") or 5)))
        return {
            "episodes": [
                {
                    **{k: v for k, v in e.items() if k not in ("lat", "lon")},
                    "by_sensor": _sensors(e["by_sensor"]),
                    "location": f"{abs(e['lat']):.2f}°{'N' if e['lat'] >= 0 else 'S'}, {abs(e['lon']):.2f}°{'E' if e['lon'] >= 0 else 'W'}",
                }
                for e in self.pack.episodes[:limit]
            ],
            "note": "Clusters of detections within 2 km and 24 h; not confirmed incidents. Locations are coordinates only: the data contain no place names, so do not name a town, province or region.",
        }

    def handover_rehearsals(self, args: dict[str, Any]) -> dict[str, Any]:
        return {"rehearsals": [{**r, "platform": label(r["platform"])} for r in self.pack.rehearsals]}

    def search_facts(self, args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query", ""))
        scores = bm25_scores(query, [f.text for f in self.pack.facts])
        ranked = sorted(zip(scores, self.pack.facts, strict=True), key=lambda x: -x[0])
        return {"facts": [{"id": f.id, "text": f.text} for score, f in ranked[:4] if score > 0]}
