"""Evidence pack: every number the agent may state, computed in code.

The language model never computes statistics. It reads short numbered facts built here from the
FIREBRIDGE API (native detections, per-satellite inventory) and, when supplied, the harmonization
report exported by the FIREBRIDGE frontend (sensor bridge, drift, rehearsal, satellite jury).
Detailed tables stay available to the agent's tools.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any

from .satellites import ORDER, SATELLITES, label, satellite_of

MONTH_NAMES = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
MODIS = ("terra", "aqua")
VIIRS = ("snpp", "noaa20", "noaa21")


def month_name(month: str) -> str:
    return f"{MONTH_NAMES[int(month[5:7]) - 1]} {month[:4]}"


def fmt_int(value: float) -> str:
    return f"{round(value):,}"


def fmt_num(value: float, digits: int = 1) -> str:
    return f"{value:,.{digits}f}"


def fmt_count(value: float) -> str:
    """Whole counts without a decimal, estimates with one."""
    return fmt_int(value) if float(value).is_integer() else fmt_num(value)


def pct(value: float) -> str:
    return f"{round(value * 100)}%"


def clock(hours: float) -> str:
    minutes = round(hours * 60)
    return f"{(minutes // 60) % 24:02d}:{minutes % 60:02d}"


def _sat_order(key: str) -> int:
    return ORDER.index(key) if key in ORDER else len(ORDER)


@dataclass
class Fact:
    id: str
    topic: str
    text: str
    priority: int = 2


@dataclass
class EvidencePack:
    analysis_id: str
    title: str
    period_start: str
    period_end_exclusive: str
    sources: list[str]
    facts: list[Fact]
    satellites: dict[str, dict[str, Any]] = field(default_factory=dict)
    months: dict[str, dict[str, Any]] = field(default_factory=dict)
    days: dict[str, dict[str, Any]] = field(default_factory=dict)
    years: dict[str, dict[str, Any]] = field(default_factory=dict)
    episodes: list[dict[str, Any]] = field(default_factory=list)
    rehearsals: list[dict[str, Any]] = field(default_factory=list)
    built_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_json(cls, text: str) -> EvidencePack:
        data = json.loads(text)
        data["facts"] = [Fact(**f) for f in data["facts"]]
        return cls(**data)

    def fact(self, fact_id: str) -> Fact | None:
        return next((f for f in self.facts if f.id == fact_id), None)

    def digest(self, token_budget: int) -> str:
        """Facts for the prompt, most important first, within an approximate token budget."""
        header = f"ANALYSIS {self.analysis_id} · {self.title} · sources: {', '.join(self.sources)}"
        lines, used = [header], len(header) / 3.5
        for f in sorted(self.facts, key=lambda f: (f.priority, int(f.id[1:]))):
            line = f"[{f.id}] {f.text}"
            cost = len(line) / 3.5
            if used + cost > token_budget:
                continue
            lines.append(line)
            used += cost
        return "\n".join(lines)


class _Facts:
    def __init__(self) -> None:
        self.items: list[Fact] = []

    def add(self, topic: str, text: str, priority: int = 2) -> str:
        fact = Fact(f"F{len(self.items) + 1}", topic, text, priority)
        self.items.append(fact)
        return fact.id


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1] if items else ""


# ── Native records (FIREBRIDGE API) ────────────────────────────────────────────


def _native(bundle: dict[str, Any], facts: _Facts, pack: EvidencePack) -> None:
    analysis, summary, calendar = bundle["analysis"], bundle["summary"], bundle["calendar"]
    area = bundle.get("area") or {}
    start, end_ex = analysis["start_date"], analysis["end_date_exclusive"]
    last_day = (date.fromisoformat(end_ex) - timedelta(days=1)).isoformat()
    name = area.get("name") or "the selected area"
    surface = area.get("surface_km2")
    pack.title = f"{name} · {start} to {last_day}"
    facts.add(
        "overview",
        f"Analysis of area '{name}'{f' ({fmt_int(surface)} km²)' if surface else ''} from {start} to {last_day} (UTC days), pinned data release {str(analysis.get('release_id', ''))[:8]}, status {analysis.get('status')}.",
        1,
    )

    # Satellite totals: collections of one satellite (e.g. MODIS C6 and C6.1) are merged.
    totals: dict[str, int] = {}
    lower_bound = False
    for source in summary.get("native_detections_by_source", []):
        key = satellite_of(source["source_id"], source.get("platform"))
        if source.get("count_status") == "lower_bound":
            lower_bound = True
        if source.get("count") is None:
            lower_bound = True
            continue
        totals[key] = totals.get(key, 0) + int(source["count"])

    # Daily inventory, judged per satellite inside its own record span (a superseded collection
    # or a satellite not yet launched is not missing data).
    periods = calendar.get("periods", [])
    spans: dict[str, tuple[str, str]] = {}
    for period in periods:
        day = period["start"][:10]
        for source in period.get("counts_by_source", []):
            if source.get("count_status") == "unknown" or source.get("count") is None:
                continue
            key = satellite_of(source["source_id"])
            first, last = spans.get(key, (day, day))
            spans[key] = (min(first, day), max(last, day))
    for period in periods:
        day = period["start"][:10]
        by: dict[str, int] = {}
        expected: set[str] = set()
        for source in period.get("counts_by_source", []):
            key = satellite_of(source["source_id"])
            span = spans.get(key)
            if not span or span[0] <= day <= span[1]:
                expected.add(key)
            if source.get("count_status") != "unknown" and source.get("count") is not None:
                by[key] = by.get(key, 0) + int(source["count"])
        pack.days[day] = {
            "total": sum(by.values()) if by else None,
            "by_sensor": by,
            "missing": sorted(expected - set(by), key=_sat_order),
        }

    for day, cell in pack.days.items():
        month = pack.months.setdefault(
            day[:7], {"total": 0, "by_sensor": {}, "days_known": 0, "days_without_record": 0, "days_partial": 0}
        )
        if cell["total"] is None:
            month["days_without_record"] += 1
            continue
        month["days_known"] += 1
        month["total"] += cell["total"]
        month["days_partial"] += bool(cell["missing"])
        for key, value in cell["by_sensor"].items():
            month["by_sensor"][key] = month["by_sensor"].get(key, 0) + value
    _years_from_months(pack)

    sats = sorted(set(totals) | set(spans), key=_sat_order)
    for key in sats:
        span = spans.get(key)
        missing = sum(1 for d, cell in pack.days.items() if key in cell["missing"]) if span else None
        pack.satellites[key] = {
            "label": label(key),
            "instrument": SATELLITES.get(key, SATELLITES["other"]).instrument,
            "detections": totals.get(key),
            "first_day": span[0] if span else None,
            "last_day": span[1] if span else None,
            "missing_days_in_span": missing,
        }

    total = sum(totals.values())
    breakdown = ", ".join(f"{label(k)} {fmt_int(totals[k])}" for k in sorted(totals, key=_sat_order))
    facts.add(
        "overview",
        f"{'At least ' if lower_bound else ''}{fmt_int(total)} native satellite detections ({breakdown}). A detection is one hot pixel at one pass, not one fire.",
        1,
    )
    modis = sum(totals.get(k, 0) for k in MODIS)
    viirs = sum(totals.get(k, 0) for k in VIIRS)
    if modis and viirs:
        facts.add(
            "satellites",
            f"VIIRS satellites recorded {fmt_num(viirs / modis)}× as many detections as MODIS ({fmt_int(viirs)} vs {fmt_int(modis)}). VIIRS 375 m pixels resolve smaller fires than MODIS 1 km pixels, so this ratio reflects the instruments, not more burning.",
            1,
        )

    known = [d for d, c in pack.days.items() if c["total"] is not None]
    active = [d for d in known if pack.days[d]["total"] > 0]
    unknown = len(pack.days) - len(known)
    partial = sum(1 for d in known if pack.days[d]["missing"])
    facts.add(
        "activity",
        f"Detections on {fmt_int(len(active))} of {fmt_int(len(known))} days with a satellite record; {fmt_int(unknown)} days have no satellite record at all (missing, not zero){f' and {partial} days lack one expected satellite' if partial else ''}.",
        1,
    )
    if pack.months and total:
        peak = max(pack.months, key=lambda m: pack.months[m]["total"])
        peak_total = pack.months[peak]["total"]
        facts.add(
            "activity",
            f"Busiest month: {month_name(peak)} with {fmt_int(peak_total)} detections, {pct(peak_total / total)} of the whole period.",
            1,
        )
    if active:
        peak_day = max(active, key=lambda d: pack.days[d]["total"])
        cell = pack.days[peak_day]
        parts = ", ".join(f"{label(k)} {v}" for k, v in sorted(cell["by_sensor"].items(), key=lambda kv: -kv[1]) if v)
        facts.add("activity", f"Busiest day: {peak_day} with {fmt_int(cell['total'])} detections ({parts}).", 1)
    calendar_months: dict[int, int] = {}
    for month, info in pack.months.items():
        calendar_months[int(month[5:7])] = calendar_months.get(int(month[5:7]), 0) + info["total"]
    if total and len(pack.years) >= 2:
        top = sorted(calendar_months, key=lambda m: -calendar_months[m])[:3]
        share = sum(calendar_months[m] for m in top) / total
        facts.add(
            "activity",
            f"Seasonality: {_join([MONTH_NAMES[m - 1] for m in sorted(top)])} together hold {pct(share)} of all detections across the years.",
            2,
        )

    for key in sats:
        s = pack.satellites[key]
        meta = SATELLITES.get(key, SATELLITES["other"])
        missing_text = (
            f"; {s['missing_days_in_span']} days missing inside that span" if s["missing_days_in_span"] else ""
        )
        count_text = fmt_int(s["detections"]) if s["detections"] is not None else "an unknown number of"
        facts.add(
            "satellites",
            f"{meta.label} ({meta.instrument}, {meta.pixel} pixels): {count_text} detections; record {s['first_day'] or 'n/a'} to {s['last_day'] or 'n/a'}{missing_text}.",
            2,
        )

    changes = []
    for key in sats:
        s = pack.satellites[key]
        if s["first_day"] and s["first_day"] > (date.fromisoformat(start) + timedelta(days=15)).isoformat():
            changes.append(f"{label(key)} record begins {s['first_day']}")
        if s["last_day"] and s["last_day"] < (date.fromisoformat(last_day) - timedelta(days=15)).isoformat():
            changes.append(f"{label(key)} record ends {s['last_day']}")
    if changes:
        facts.add(
            "satellites",
            f"The satellite fleet changes inside the period: {_join(changes)}. Raw totals before and after these dates are not directly comparable.",
            1,
        )

    if len(pack.years) >= 2:
        years = " · ".join(
            f"{y} {fmt_int(info['total'])}{' (partial)' if info['partial'] else ''}" for y, info in pack.years.items()
        )
        facts.add("activity", f"Native detections by year, all satellites: {years}.", 2)
    for text in summary.get("limitations", [])[:4]:
        facts.add("limitations", f"Data limitation reported by FIREBRIDGE: {text}", 3)


def _years_from_months(pack: EvidencePack) -> None:
    for month, info in sorted(pack.months.items()):
        year = pack.years.setdefault(month[:4], {"total": 0, "by_sensor": {}, "months": 0, "partial": False})
        year["months"] += 1
        year["total"] += info.get("total") or 0
        year["partial"] = year["partial"] or bool(info.get("days_without_record"))
        for key, value in (info.get("by_sensor") or {}).items():
            year["by_sensor"][key] = year["by_sensor"].get(key, 0) + value
    for info in pack.years.values():
        info["partial"] = info["partial"] or info["months"] < 12


# ── Harmonization report (FIREBRIDGE sensor bridge export) ─────────────────────


def _r2(value: Any) -> Any:
    return round(value, 2) if isinstance(value, float) else value


def _bridge(report: dict[str, Any], facts: _Facts, pack: EvidencePack, native_loaded: bool) -> None:
    meta = report.get("meta", {})
    if not native_loaded:
        pack.period_start, pack.period_end_exclusive = meta.get("start", ""), meta.get("endExclusive", "")
        pack.title = f"bridge report · {pack.period_start} to {pack.period_end_exclusive}"
    floor = (report.get("sensitivity") or {}).get("floor")

    for m in report.get("months", []):
        month = pack.months.setdefault(
            m["month"], {"total": m.get("raw"), "by_sensor": dict(m.get("rawBySensor") or {})}
        )
        if m.get("bridge"):
            b = m["bridge"]
            month["harmonized"] = {
                "value": _r2(b["value"]),
                "low": _r2(b["low"]),
                "high": _r2(b["high"]),
                "source": b["source"],
                "aqua_observed": m.get("aqua"),
                "aqua_drifted": m.get("referenceDrift", False),
            }
    if not native_loaded:
        _years_from_months(pack)
    for y in report.get("years", []):
        if y.get("bridge") and y["year"] in pack.years:
            pack.years[y["year"]]["harmonized"] = {
                "value": _r2(y["bridge"]["value"]),
                "low": _r2(y["bridge"]["low"]),
                "high": _r2(y["bridge"]["high"]),
                "observed_share": _r2(y.get("observedShare")),
                "partial": y.get("partial"),
            }

    gap = report.get("headline")
    if gap and gap.get("ratioDetections"):
        shares = gap.get("shares") or {}
        explained = (shares.get("pixel") or 0) + (shares.get("smallFires") or 0)
        facts.add(
            "harmonization",
            f"{label(gap['platform'])} records {fmt_num(gap['ratioDetections'])}× as many daytime detections as Aqua in the same months, {fmt_num(gap['ratioCells'])}× when counting active 5 km cells, and {fmt_num(gap['ratioVisible'])}× counting only fires above Aqua's measured detection floor{f' of {floor} MW' if floor else ''}. About {pct(explained)} of the gap comes from pixel size and small fires, i.e. from the instruments.",
            1,
        )

    regime = report.get("regime") or {}
    aqua_drift = next((d for d in regime.get("drift", []) if d.get("sensor") == "aqua"), None)
    aqua_orbit = next((o for o in regime.get("orbits", []) if o.get("sensor") == "aqua"), None)
    if aqua_drift and aqua_orbit and aqua_orbit.get("onset"):
        facts.add(
            "drift",
            f"Aqua's afternoon overpass drifted from {clock(aqua_drift['from'])} ({aqua_drift['fromYear']}) to {clock(aqua_drift['to'])} ({aqua_drift['toYear']}), {aqua_drift['shiftMinutes']:+d} minutes, measured from detection timestamps. From {aqua_orbit['onset']} Aqua is outside its original regime, so those years are estimated from VIIRS in the harmonized record.",
            1,
        )
    fingerprint = regime.get("fingerprint") or {}
    before, after = fingerprint.get("inRegime") or {}, fingerprint.get("drifted") or {}
    if before.get("share") is not None and after.get("share") is not None:
        facts.add(
            "drift",
            f"Share of Aqua fire cell-days that no VIIRS satellite saw that day: {pct(before['share'])} with the original pass ({before['unmatched']} of {before['cellDays']}) vs {pct(after['share'])} after the drift ({after['unmatched']} of {after['cellDays']}); the 90% intervals {'do not overlap' if fingerprint.get('distinguishable') else 'overlap'}.",
            1,
        )

    replays = [r for r in report.get("replays", []) if r.get("support") != "insufficient"]
    for r in report.get("replays", []):
        if r.get("support") == "insufficient":
            facts.add(
                "harmonization",
                f"{label(r['platform'])} is not calibrated: {' '.join(r.get('supportReasons') or [])}",
                3,
            )
            continue
        chosen = next((s for s in r.get("scores", []) if s["model"] == r["decision"]["model"]), None)
        if chosen:
            facts.add(
                "harmonization",
                f"Withheld-Aqua replay with {label(r['platform'])} ({r['support']}, {r['blocks']} {r['blockKind']} blocks, {r['pairedMonths']} paired months): adopted model {r['decision']['model']}, mean error {fmt_num(chosen['mae'], 2)} Aqua-active cell-days per month; largest miss {month_name(r['largestMiss']['month']) if r.get('largestMiss') else 'none'}.",
                2,
            )

    default = max(replays, key=lambda r: (r.get("pairedMonths", 0), r.get("platformCellDays", 0)), default=None)
    for h in report.get("handovers", []):
        scored, alarms = h["scored"], h["alarms"]
        truth = set(alarms["truth"])
        entry = {
            "platform": h["platform"],
            "switch_month": h["switchMonth"],
            "model": h["model"],
            "scored": {k: _r2(v) for k, v in scored.items()},
            "naive_false_alarms": [m for m in alarms["naive"] if m not in truth],
            "bridge_false_alarms": [m for m in alarms["bridged"] if m not in truth],
            "real_alarms": alarms["truth"],
            "real_alarms_caught": [m for m in alarms["truth"] if m in alarms["bridged"]],
        }
        pack.rehearsals.append(entry)
    shown = [h for h in pack.rehearsals if default and h["platform"] == default["platform"]]
    if shown:
        h = max(shown, key=lambda x: x["scored"]["months"])
        s = h["scored"]
        actual = max(1, s["actual"])
        facts.add(
            "rehearsal",
            f"Handover rehearsal: with Aqua switched off in {month_name(h['switch_month'])} and {label(h['platform'])} continuing the record ({s['months']} months scored), the un-harmonized record reads {fmt_num(s['naive'] / actual)}× the true Aqua record and raises {len(h['naive_false_alarms'])} false alarms; FIREBRIDGE reads {fmt_num(s['estimate'] / actual)}× and raises {len(h['bridge_false_alarms'])}, catching {len(h['real_alarms_caught'])} of {len(h['real_alarms'])} real alarms.",
            1,
        )

    _jury(report, facts, pack)

    effect = report.get("driftEffect")
    if effect and effect.get("estimate"):
        facts.add(
            "drift",
            f"In the drifted months Aqua as flown recorded {fmt_int(effect['observed'])} active cell-days; at its original time the VIIRS-based estimate is {fmt_num(effect['estimate']['value'])} (range {fmt_num(effect['estimate']['low'])}–{fmt_num(effect['estimate']['high'])}).",
            3,
        )

    harmonized_years = [(y, info["harmonized"]) for y, info in pack.years.items() if info.get("harmonized")]
    if len(harmonized_years) >= 2:
        text = " · ".join(
            f"{y} {fmt_count(h['value'])}{'*' if (h.get('observed_share') or 0) < 1 else ''}"
            for y, h in harmonized_years
        )
        facts.add(
            "harmonization",
            f"Harmonized activity by year (Aqua-equivalent active cell-days; * partly estimated from VIIRS): {text}.",
            2,
        )

    episodes = report.get("episodes") or {}
    for e in (episodes.get("top") or [])[:8]:
        pack.episodes.append(
            {
                "id": e["id"],
                "start": e["start"],
                "end": e["end"],
                "hours": e["durationHours"],
                "detections": e["detections"],
                "passes": e["passes"],
                "by_sensor": e["bySensor"],
                "max_frp_mw": e.get("maxFrp"),
                "lat": round(e["lat"], 3),
                "lon": round(e["lon"], 3),
            }
        )
    if episodes.get("count"):
        top = pack.episodes[0] if pack.episodes else None
        top_text = (
            f" Largest: {top['start'][:10]} to {top['end'][:10]}, {fmt_int(top['detections'])} detections, peak {fmt_int(top['max_frp_mw'] or 0)} MW."
            if top
            else ""
        )
        facts.add(
            "episodes",
            f"Chaining detections within {episodes.get('linkMetres', 2000) / 1000:g} km and {episodes.get('linkHours', 24)} h gives {fmt_int(episodes['count'])} detection clusters; {pct(episodes['viirsOnly'] / episodes['count'])} were seen by VIIRS only.{top_text} Clusters are a reading aid, not confirmed fire incidents.",
            2,
        )


def _jury(report: dict[str, Any], facts: _Facts, pack: EvidencePack) -> None:
    anomalies = (report.get("anomalies") or {}).get("months", [])
    for a in anomalies:
        month = pack.months.setdefault(a["month"], {"total": None, "by_sensor": {}})
        consensus = a.get("consensus") or {}
        month["jury"] = {
            "kind": a.get("kind"),
            "reason": a.get("reason"),
            "outlier": a.get("outlier"),
            "harmonized": {k: _r2(v) for k, v in (a.get("harmonized") or {}).items()} or None,
            "consensus_z": _r2(consensus.get("z")),
            "agreeing": consensus.get("agreeing"),
            "testifying": consensus.get("n"),
            "witnesses": {
                w["sensor"]: {
                    "state": w["state"],
                    **({k: _r2(v) for k, v in w["score"].items()} if w.get("score") else {}),
                }
                for w in a.get("witnesses", [])
            },
        }

    def strength(a: dict[str, Any]) -> float:
        return max(abs((a.get("consensus") or {}).get("z") or 0), abs((a.get("harmonized") or {}).get("z") or 0))

    unusual = sorted(
        (a for a in anomalies if a.get("kind") in ("unusual-high", "unusual-low")), key=strength, reverse=True
    )
    for i, a in enumerate(unusual[:5]):
        h, c = a.get("harmonized") or {}, a.get("consensus") or {}
        level = "high" if a["kind"] == "unusual-high" else "low"
        value = (
            f"{fmt_count(h['value'])} Aqua-equivalent active cell-days against a usual {fmt_num(h['usual'])}; "
            if h
            else ""
        )
        facts.add(
            "jury",
            f"{month_name(a['month'])} was unusually {level}: {value}{c.get('agreeing', 0)} of {c.get('n', 0)} satellites, each compared only with the same month in its own other years, agree.",
            1 if i < 2 else 2,
        )
    for a in [a for a in anomalies if a.get("kind") == "disputed"][:3]:
        others = [
            label(w["sensor"])
            for w in a.get("witnesses", [])
            if w.get("state") == "scored" and w["sensor"] != a.get("outlier")
        ]
        h = a.get("harmonized") or {}
        direction = "quiet" if (h.get("z") or 0) < 0 else "busy"
        who = (
            f"{label(a['outlier'])} alone recorded an unusually {direction} month; {_join(others)} did not"
            if a.get("outlier")
            else "the harmonized record stands out but the satellites do not confirm it"
        )
        facts.add(
            "jury",
            f"{month_name(a['month'])} is disputed: {who}. The reference record itself may have missed what the others saw.",
            1,
        )
    artifacts = [a for a in anomalies if a.get("kind") == "artifact"]
    if artifacts:
        no_record = [month_name(a["month"]) for a in artifacts if a.get("reason") == "no-record"]
        fleet = [month_name(a["month"]) for a in artifacts if a.get("reason") != "no-record"]
        parts = []
        if fleet:
            parts.append(f"{_join(fleet)} because satellites in the fleet have no record then")
        if no_record:
            parts.append(f"{_join(no_record)} because no satellite record exists (missing, not zero)")
        facts.add(
            "jury",
            f"Raw all-satellite totals would flag {len(artifacts)} months as unusual that are artifacts: {'; '.join(parts)}.",
            1,
        )
    if anomalies and not unusual and not artifacts:
        facts.add("jury", "No month stands out once each satellite is compared with its own past.", 2)


def build_pack(
    analysis_id: str, *, bundle: dict[str, Any] | None = None, report: dict[str, Any] | None = None
) -> EvidencePack:
    if bundle is None and report is None:
        raise ValueError("An evidence pack needs FIREBRIDGE API data, a bridge report, or both.")
    facts = _Facts()
    pack = EvidencePack(
        analysis_id=analysis_id, title="", period_start="", period_end_exclusive="", sources=[], facts=[]
    )
    if bundle is not None:
        pack.period_start = bundle["analysis"]["start_date"]
        pack.period_end_exclusive = bundle["analysis"]["end_date_exclusive"]
        pack.sources.append("firebridge_api")
        _native(bundle, facts, pack)
    if report is not None:
        pack.sources.append("bridge_report")
        _bridge(report.get("report", report), facts, pack, native_loaded=bundle is not None)
    facts.add(
        "method",
        "Definitions: missing data is never zero; the harmonized record estimates what Aqua would report, not ground truth; detections, clusters and FRP are not fire counts or burned area; causes of fires are not established by these data.",
        1,
    )
    pack.facts = facts.items
    return pack
