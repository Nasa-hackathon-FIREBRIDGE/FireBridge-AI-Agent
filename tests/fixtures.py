"""Synthetic FIREBRIDGE inputs exercising the hard cases: a superseded MODIS collection
(C6 → C6.1 on 2022-07-01), a satellite joining mid-window (NOAA-21 on 2022-03-01), one missing
Aqua week, and a summer fire season."""

from datetime import date, timedelta

ANALYSIS_ID = "aaaaaaaa-1111-2222-3333-444444444444"


def _days(start: str, end_exclusive: str):
    day = date.fromisoformat(start)
    while day < date.fromisoformat(end_exclusive):
        yield day
        day += timedelta(days=1)


def native_bundle() -> dict:
    periods = []
    totals = {
        "modis-aqua-c6-standard": 0,
        "modis-aqua-c61-standard": 0,
        "viirs-snpp-c2-standard": 0,
        "viirs-noaa21-c2-standard": 0,
    }
    for day in _days("2021-01-01", "2023-01-01"):
        iso = day.isoformat()
        summer = day.month in (7, 8)
        fires = (6 if summer else 1) + (20 if iso == "2022-07-15" else 0)
        counts = []

        def add(source: str, count: int | None, counts: list = counts) -> None:
            counts.append(
                {
                    "source_id": source,
                    "count": None if count is None else str(count),
                    "count_status": "unknown" if count is None else "complete",
                }
            )
            if count is not None:
                totals[source] += count

        aqua_gap = "2022-04-05" <= iso <= "2022-04-11"
        add("modis-aqua-c6-standard", None if iso >= "2022-07-01" or aqua_gap else fires // 3)
        add("modis-aqua-c61-standard", fires // 3 if iso >= "2022-07-01" else None)
        add("viirs-snpp-c2-standard", fires)
        add("viirs-noaa21-c2-standard", fires if iso >= "2022-03-01" else None)
        periods.append(
            {
                "start": f"{iso}T00:00:00.000Z",
                "end_exclusive": f"{(day + timedelta(days=1)).isoformat()}T00:00:00.000Z",
                "counts_by_source": counts,
            }
        )
    platform = {
        "modis-aqua-c6-standard": "Aqua",
        "modis-aqua-c61-standard": "Aqua",
        "viirs-snpp-c2-standard": "SNPP",
        "viirs-noaa21-c2-standard": "N21",
    }
    return {
        "analysis": {
            "id": ANALYSIS_ID,
            "status": "ready",
            "start_date": "2021-01-01",
            "end_date_exclusive": "2023-01-01",
            "release_id": "rel12345-0000",
            "area_id": "area-1",
        },
        "summary": {
            "native_detections_by_source": [
                {"source_id": s, "platform": platform[s], "count": str(n), "count_status": "complete"}
                for s, n in totals.items()
            ],
            "limitations": ["DETECTIONS_ARE_NOT_UNIQUE_INCIDENTS"],
        },
        "calendar": {"periods": periods},
        "area": {"id": "area-1", "name": "Rif test area", "surface_km2": 1234.5},
    }


def bridge_report() -> dict:
    """A compact sensor-bridge export with every section the evidence builder reads."""
    months = []
    for year in (2021, 2022):
        for m in range(1, 13):
            month = f"{year}-{m:02d}"
            value = 30 if month == "2022-07" else (6 if m in (7, 8) else 1)
            months.append(
                {
                    "month": month,
                    "aqua": value,
                    "referenceDrift": False,
                    "bridge": {"value": value, "low": value, "high": value, "source": "aqua"},
                    "raw": value * 9,
                    "rawBySensor": {"aqua": value, "snpp": value * 8},
                }
            )

    def witness(sensor: str, z: float) -> dict:
        return {"sensor": sensor, "state": "scored", "score": {"value": 10, "usual": 2.0, "z": z, "years": 4}}

    return {
        "analysis_id": ANALYSIS_ID,
        "release_id": "rel12345-0000",
        "report": {
            "meta": {"engine": "fb-bridge/1.1.0", "start": "2021-01-01", "endExclusive": "2023-01-01"},
            "months": months,
            "years": [
                {
                    "year": "2021",
                    "bridge": {"value": 32.0, "low": 32.0, "high": 32.0},
                    "observedShare": 1.0,
                    "partial": False,
                },
                {
                    "year": "2022",
                    "bridge": {"value": 56.5, "low": 50.0, "high": 63.0},
                    "observedShare": 0.9,
                    "partial": False,
                },
            ],
            "headline": {
                "platform": "snpp",
                "ratioDetections": 5.6,
                "ratioCells": 4.6,
                "ratioVisible": 1.5,
                "shares": {"pixel": 0.11, "smallFires": 0.63, "residual": 0.26},
            },
            "sensitivity": {"floor": 7.1},
            "regime": {
                "drift": [
                    {
                        "sensor": "aqua",
                        "fromYear": "2021",
                        "toYear": "2022",
                        "from": 13.25,
                        "to": 14.5,
                        "shiftMinutes": 75,
                    }
                ],
                "orbits": [
                    {
                        "sensor": "aqua",
                        "baseline": 13.25,
                        "baselineFrom": "maintained-orbit",
                        "onset": "2022",
                        "years": ["2022"],
                    }
                ],
                "fingerprint": {
                    "inRegime": {"cellDays": 125, "unmatched": 15, "share": 0.12},
                    "drifted": {"cellDays": 51, "unmatched": 14, "share": 0.27},
                    "distinguishable": True,
                },
            },
            "replays": [
                {
                    "platform": "snpp",
                    "support": "supported",
                    "blocks": 5,
                    "blockKind": "year",
                    "pairedMonths": 59,
                    "platformCellDays": 547,
                    "supportReasons": [],
                    "decision": {"model": "B0"},
                    "scores": [{"model": "B0", "mae": 1.16}],
                    "largestMiss": {"month": "2021-08"},
                },
                {
                    "platform": "noaa21",
                    "support": "insufficient",
                    "supportReasons": ["NOAA-21 overlaps Aqua only after the drift."],
                    "pairedMonths": 0,
                },
            ],
            "handovers": [
                {
                    "platform": "snpp",
                    "switchMonth": "2022-01",
                    "model": "B0",
                    "scored": {
                        "months": 12,
                        "actual": 90,
                        "naive": 325,
                        "estimate": 74.4,
                        "low": 35.8,
                        "high": 130.6,
                        "covered": 10,
                    },
                    "alarms": {
                        "truth": ["2022-07"],
                        "naive": ["2022-02", "2022-07", "2022-08"],
                        "bridged": ["2022-07"],
                    },
                }
            ],
            "anomalies": {
                "months": [
                    {
                        "month": "2022-07",
                        "kind": "unusual-high",
                        "reason": None,
                        "outlier": None,
                        "harmonized": {"value": 30, "usual": 1.5, "z": 4.2, "years": 4},
                        "consensus": {"z": 2.5, "n": 2, "agreeing": 2},
                        "witnesses": [witness("aqua", 3.1), witness("snpp", 1.9)],
                    },
                    {
                        "month": "2021-08",
                        "kind": "disputed",
                        "reason": "outlier",
                        "outlier": "aqua",
                        "harmonized": {"value": 0, "usual": 5.1, "z": -3.0, "years": 4},
                        "consensus": {"z": 0.4, "n": 3, "agreeing": 0},
                        "witnesses": [witness("aqua", -2.8), witness("snpp", 0.6), witness("terra", 0.7)],
                    },
                    {
                        "month": "2022-12",
                        "kind": "artifact",
                        "reason": "no-record",
                        "outlier": None,
                        "harmonized": None,
                        "consensus": None,
                        "witnesses": [{"sensor": "aqua", "state": "no-record", "score": None}],
                    },
                ]
            },
            "episodes": {
                "count": 120,
                "viirsOnly": 100,
                "linkMetres": 2000,
                "linkHours": 24,
                "top": [
                    {
                        "id": "E1",
                        "start": "2022-07-13T21:46:00Z",
                        "end": "2022-07-17T13:46:00Z",
                        "durationHours": 88,
                        "detections": 508,
                        "passes": 38,
                        "bySensor": {"snpp": 400, "aqua": 108},
                        "maxFrp": 1304.7,
                        "lat": 35.05,
                        "lon": -5.63,
                    }
                ],
            },
            "driftEffect": {"observed": 51, "estimate": {"value": 43.8, "low": 6.0, "high": 116.2}},
        },
    }
