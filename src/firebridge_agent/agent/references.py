"""What an answer is about, in terms a user interface can act on: months to open in the calendar,
days and fire episodes to show on the map, and sections of the analysis to open."""

import json
import re
from typing import Any

from ..data.evidence import MONTH_NAMES, EvidencePack, Fact

_FRENCH = [
    "janvier",
    "février",
    "mars",
    "avril",
    "mai",
    "juin",
    "juillet",
    "août",
    "septembre",
    "octobre",
    "novembre",
    "décembre",
]
_SPANISH = [
    "enero",
    "febrero",
    "marzo",
    "abril",
    "mayo",
    "junio",
    "julio",
    "agosto",
    "septiembre",
    "octubre",
    "noviembre",
    "diciembre",
]
_NAMES = {name.lower(): i + 1 for names in (MONTH_NAMES, _FRENCH, _SPANISH) for i, name in enumerate(names)}
_MONTH_YEAR = re.compile(
    r"\b(" + "|".join(sorted(map(re.escape, _NAMES), key=len, reverse=True)) + r")\s+(?:de\s+)?(\d{4})\b", re.IGNORECASE
)
_ISO_DAY = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_EPISODE = re.compile(r"Largest: (\d{4}-\d{2}-\d{2}) to (\d{4}-\d{2}-\d{2})")

SECTIONS = ("overview", "calendar", "gap", "regime", "replay", "rehearsal", "bridge", "episodes")
MAX = {"months": 4, "days": 2, "episodes": 3, "sections": 3}


def months_in_text(text: str) -> list[str]:
    found: list[str] = []
    for name, year in _MONTH_YEAR.findall(text):
        month = f"{year}-{_NAMES[name.lower()]:02d}"
        if month not in found:
            found.append(month)
    return found


def no_refs() -> dict[str, list[Any]]:
    return {"months": [], "days": [], "episodes": [], "sections": []}


def fact_refs(fact: Fact) -> dict[str, list[Any]]:
    """References implied by a fact's topic and wording."""
    refs = no_refs()
    text = fact.text
    if fact.topic == "harmonization":
        refs["sections"].append(
            "replay"
            if "replay" in text or "not calibrated" in text
            else "bridge"
            if text.startswith("Harmonized activity")
            else "gap"
        )
    elif fact.topic == "drift":
        refs["sections"].append("regime")
    elif fact.topic == "rehearsal":
        refs["sections"].append("rehearsal")
    elif fact.topic == "jury":
        refs["sections"].append("calendar")
    elif fact.topic == "episodes":
        refs["sections"].append("episodes")
        if match := _EPISODE.search(text):
            refs["episodes"].append({"start": match[1], "end": match[2]})
    if fact.topic in ("jury", "activity", "harmonization"):
        refs["months"] = months_in_text(text)
    if fact.topic == "activity" and text.startswith("Busiest day"):
        refs["days"] = ["-".join(m) for m in _ISO_DAY.findall(text)][:1]
    return refs


def _merge(into: dict[str, list[Any]], refs: dict[str, list[Any]]) -> None:
    for key, values in refs.items():
        for value in values:
            if value not in into[key] and len(into[key]) < MAX[key]:
                into[key].append(value)


def collect(
    pack: EvidencePack, *, answer: str, fact_ids: list[str], tool_calls: list[dict[str, Any]]
) -> dict[str, list[Any]]:
    from .tools import parse_day, parse_month

    refs = no_refs()
    for call in tool_calls:
        try:
            args = json.loads(call.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        name = call.get("name")
        if name == "get_month" and (month := parse_month(str(args.get("month", "")))):
            _merge(refs, {"months": [month]})
        elif name == "get_day" and (day := parse_day(str(args.get("date", "")))):
            _merge(refs, {"days": [day]})
        elif name == "compare_periods":
            _merge(refs, {"months": [m for m in (parse_month(str(args.get(k, ""))) for k in ("first", "second")) if m]})
        elif name == "largest_episodes":
            _merge(
                refs,
                {
                    "episodes": [{"start": e["start"][:10], "end": e["end"][:10]} for e in pack.episodes[:3]],
                    "sections": ["episodes"],
                },
            )
        elif name == "unusual_months":
            _merge(refs, {"sections": ["calendar"]})
        elif name == "handover_rehearsals":
            _merge(refs, {"sections": ["rehearsal"]})
    for fid in fact_ids:
        if fact := pack.fact(fid):
            _merge(refs, fact_refs(fact))
    _merge(refs, {"months": [m for m in months_in_text(answer) if m in pack.months]})
    refs["months"] = [m for m in refs["months"] if m in pack.months]
    return refs
