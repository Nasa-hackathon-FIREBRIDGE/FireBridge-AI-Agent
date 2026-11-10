import json

import pytest

from firebridge_agent.agent.guard import allowed_values, is_degenerate, strip_tool_references, verify
from firebridge_agent.agent.tools import Toolbox, parse_day, parse_month
from firebridge_agent.data.evidence import build_pack

from .fixtures import ANALYSIS_ID, bridge_report, native_bundle

EVIDENCE = "[F2] At least 7,598 detections. [F5] July 2022 with 1,168 detections, 15% of the period. Share 0.27. Ratio 5.34. Drift 13:15 to 15:18 on 2026-08-31."


@pytest.mark.parametrize(
    "answer",
    [
        "There were 7,598 detections and 1,168 in July 2022 [F2][F5].",
        "About 7,600 detections; the ratio is 5.3× and the share 27%.",
        "Il y a eu 7 598 détections, dont 1 168 en juillet 2022 [ F5 ]; ratio 5,3.",
        "Aqua drifted from 13:15 to 15:18; records end 2026-08-31; 3 of 4 satellites agree.",
    ],
)
def test_accepts_numbers_from_the_evidence(answer):
    assert verify(answer, allowed_values(EVIDENCE)).as_dict()["status"] == "verified"


def test_flags_invented_numbers():
    result = verify("Activity was 19 times higher, with 4,500 detections and 42% growth.", allowed_values(EVIDENCE))
    assert result.unsupported == ["19", "4,500", "42"] and result.status == "unverified_numbers"


def test_detects_looping_output_and_strips_tool_citations():
    assert is_degenerate("27–20 juillet 2022, 4 jours, 205, 218, 24, 205. " * 12)
    assert not is_degenerate("July 2022 was unusual. " + "Each satellite agrees for a different reason. " * 2)
    assert (
        strip_tool_references("Up 23% [compare_periods] and see [F3].", ["compare_periods"]) == "Up 23% and see [F3]."
    )


@pytest.mark.parametrize(
    ("text", "month"),
    [
        ("2022-07", "2022-07"),
        ("2022/7", "2022-07"),
        ("July 2022", "2022-07"),
        ("Sept 2026", "2026-09"),
        ("07/2022", "2022-07"),
        ("2022-13", None),
        ("soon", None),
    ],
)
def test_parses_months_leniently(text, month):
    assert parse_month(text) == month


def test_parses_days_leniently():
    assert parse_day("15 July 2022") == parse_day("July 15, 2022") == parse_day("2022-7-15") == "2022-07-15"


@pytest.fixture
def toolbox() -> Toolbox:
    return Toolbox(build_pack(ANALYSIS_ID, bundle=native_bundle(), report=bridge_report()))


def run(toolbox: Toolbox, name: str, **args) -> dict:
    return json.loads(toolbox.run(name, json.dumps(args)))


def test_month_and_day_lookups(toolbox):
    july = run(toolbox, "get_month", month="July 2022")
    assert (
        july["month"] == "July 2022" and july["harmonized"]["value"] == 30 and july["jury"]["verdict"] == "unusual-high"
    )
    assert run(toolbox, "get_day", date="2022-04-06")["satellites_without_record"] == ["Aqua"]
    assert "error" in run(toolbox, "get_month", month="1999-01")


def test_compare_periods_flags_fleet_changes(toolbox):
    out = run(toolbox, "compare_periods", first="2021", second="2022")
    assert out["only_in_second"] == ["NOAA-21"] and "not directly comparable" in out["note"]
    assert out["harmonized"]["change"] == "+77%"


def test_jury_episodes_rehearsal_and_search(toolbox):
    assert [m["month"] for m in run(toolbox, "unusual_months", kind="disputed")["months"]] == ["August 2021"]
    assert run(toolbox, "largest_episodes", limit=1)["episodes"][0]["detections"] == 508
    assert run(toolbox, "handover_rehearsals")["rehearsals"][0]["platform"] == "Suomi-NPP"
    assert run(toolbox, "search_facts", query="disputed August")["facts"][0]["text"].startswith(
        "August 2021 is disputed"
    )
    assert {t["function"]["name"] for t in toolbox.schemas()} >= {
        "unusual_months",
        "largest_episodes",
        "handover_rehearsals",
    }


def test_tool_errors_are_returned_not_raised(toolbox):
    assert "unknown tool" in run(toolbox, "launch_rocket")["error"]
    assert "invalid arguments" in json.loads(toolbox.run("get_month", "{not json"))["error"]
    assert "error" in run(toolbox, "compare_periods", first="2021", second="next year")


def test_normalizes_full_width_citations_and_tool_names():
    text = "Aqua recorded zero【get_month】 while others did not【F29】 and 【F3, F4】; see [compare_periods†L1-L4]."
    assert (
        strip_tool_references(text, ["get_month", "compare_periods"])
        == "Aqua recorded zero while others did not[F29] and [F3, F4]; see."
    )


def test_jury_view_names_each_satellites_instrument(toolbox):
    witnesses = run(toolbox, "get_month", month="2021-08")["jury"]["each_satellite"]
    assert set(witnesses) == {"Aqua (MODIS)", "Suomi-NPP (VIIRS)", "Terra (MODIS)"}
