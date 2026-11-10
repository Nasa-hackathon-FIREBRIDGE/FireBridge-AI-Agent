from firebridge_agent.data.evidence import EvidencePack, build_pack

from .fixtures import ANALYSIS_ID, bridge_report, native_bundle


def texts(pack: EvidencePack, topic: str) -> list[str]:
    return [f.text for f in pack.facts if f.topic == topic]


def test_native_pack_judges_availability_per_satellite():
    pack = build_pack(ANALYSIS_ID, bundle=native_bundle())
    # C6 → C6.1 is one Aqua record; NOAA-21 before launch is not missing data.
    assert pack.satellites["aqua"]["first_day"] == "2021-01-01" and pack.satellites["aqua"]["last_day"] == "2022-12-31"
    assert pack.satellites["aqua"]["missing_days_in_span"] == 7
    assert (
        pack.satellites["noaa21"]["first_day"] == "2022-03-01" and not pack.satellites["noaa21"]["missing_days_in_span"]
    )
    assert pack.days["2022-04-06"]["missing"] == ["aqua"] and pack.days["2022-04-06"]["total"] is not None
    assert pack.days["2021-06-01"]["missing"] == []
    assert any("NOAA-21 record begins 2022-03-01" in t for t in texts(pack, "satellites"))
    assert any("Busiest day: 2022-07-15" in t for t in texts(pack, "activity"))
    assert any("Busiest month: July 2022" in t for t in texts(pack, "activity"))


def test_bridge_report_adds_harmonization_drift_rehearsal_and_jury():
    pack = build_pack(ANALYSIS_ID, bundle=native_bundle(), report=bridge_report())
    assert pack.sources == ["firebridge_api", "bridge_report"]
    jury = " ".join(texts(pack, "jury"))
    assert (
        "July 2022 was unusually high: 30 Aqua-equivalent active cell-days against a usual 1.5; 2 of 2 satellites"
        in jury
    )
    assert "August 2021 is disputed: Aqua alone recorded an unusually quiet month" in jury
    assert "December 2022 because no satellite record exists (missing, not zero)" in jury
    assert any("12% with the original pass" in t for t in texts(pack, "drift"))
    assert any(
        "3.6× the true Aqua record and raises 2 false alarms" in t and "raises 0" in t for t in texts(pack, "rehearsal")
    )
    assert any("NOAA-21 is not calibrated" in t for t in texts(pack, "harmonization"))
    assert pack.months["2022-07"]["jury"]["kind"] == "unusual-high"
    assert pack.rehearsals[0]["naive_false_alarms"] == ["2022-02", "2022-08"]


def test_report_only_pack_and_roundtrip():
    pack = build_pack(ANALYSIS_ID, report=bridge_report())
    assert pack.period_start == "2021-01-01" and pack.months["2022-07"]["total"] == 270
    assert EvidencePack.from_json(pack.to_json()) == pack


def test_digest_puts_priority_facts_first_within_budget():
    pack = build_pack(ANALYSIS_ID, bundle=native_bundle(), report=bridge_report())
    full, small = pack.digest(5000), pack.digest(300)
    assert len(small) < len(full) and len(small) / 3.5 <= 330
    kept = [line for line in small.splitlines()[1:]]
    priorities = [pack.fact(line.split("]")[0][1:]).priority for line in kept]
    assert priorities == sorted(priorities) and priorities[0] == 1
