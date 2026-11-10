"""Number guard: every figure in a generated answer must trace back to what the model was shown.

Heuristic by design (it checks values, not meaning), so the result is reported alongside the
answer rather than silently blocking it. Rounding and percent/fraction forms are accepted.
"""

import re
from dataclasses import dataclass, field

_CITATION = re.compile(r"\[\s*F\d+(?:\s*[,;]\s*F?\d+)*\s*\]")
# "1 022" / "1 022" (space thousands separators) and "17,9" (decimal comma), as French writes them.
_SPACED_THOUSANDS = re.compile(r"(?<=\d)[\u00a0\u202f ](?=\d{3}(?!\d))")
_DECIMAL_COMMA = re.compile(r"(?<=\d),(?=\d{1,2}(?!\d))")


def normalize_numbers(text: str) -> str:
    return _DECIMAL_COMMA.sub(".", _SPACED_THOUSANDS.sub("", text))


_ISO_DATE = re.compile(r"\b\d{4}-\d{2}(?:-\d{2})?\b")
_CLOCK = re.compile(r"\b\d{1,2}:\d{2}\b")
_NUMBER = re.compile(r"(?<![\w.])[-+−]?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\d])|(?<![\w.,])[-+−]?\d+(?:\.\d+)?(?![\d])")


@dataclass
class Verification:
    checked: int = 0
    unsupported: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        return "verified" if not self.unsupported else "unverified_numbers"

    def as_dict(self) -> dict:
        return {"status": self.status, "numbers_checked": self.checked, "unsupported_numbers": self.unsupported}


def _values(text: str) -> list[tuple[str, float, int]]:
    """(raw, value, decimals) for every number, ignoring citations; dates and clock times are split into parts."""
    text = _CITATION.sub(" ", text)
    out = []
    for match in _NUMBER.finditer(text):
        raw = match.group(0)
        clean = raw.replace(",", "").replace("−", "-").lstrip("+")
        decimals = len(clean.split(".")[1]) if "." in clean else 0
        out.append((raw, float(clean), decimals))
    return out


def strip_tool_references(text: str, tool_names: list[str]) -> str:
    """Models sometimes cite tools like sources ("[compare_periods]", "【get_month】"); evidence
    citations are [F#] only. Full-width brackets around evidence ids become plain [F#]."""
    text = re.sub(r"【\s*(F\d+(?:\s*[,;]\s*F?\d+)*)\s*】", r"[\1]", text)
    names = "|".join(map(re.escape, [*tool_names, "result", "tool"]))
    return re.sub(r"\s?[\[【]\s*(?:" + names + r")(?:[^\]】]{0,40})?\s*[\]】]", "", text)


def is_degenerate(text: str) -> bool:
    """Looping output (the same fragment over and over), which small models produce under pressure."""
    compact = " ".join(text.split())
    if len(compact) < 160:
        return False
    shingles = [compact[i : i + 24] for i in range(0, len(compact) - 24, 12)]
    return len(set(shingles)) / len(shingles) < 0.5


def allowed_values(*sources: str) -> set[float]:
    values: set[float] = set()
    for source in sources:
        for _, value, _ in _values(_CLOCK.sub(lambda m: m.group(0).replace(":", " "), source)):
            values.update({abs(value), abs(value) * 100, abs(value) / 100})
        for date in _ISO_DATE.findall(source):
            values.update(float(p) for p in date.split("-"))
    return values


def verify(answer: str, allowed: set[float]) -> Verification:
    result = Verification()
    text = _CLOCK.sub(
        lambda m: m.group(0).replace(":", " "),
        _ISO_DATE.sub(lambda m: " ".join(m.group(0).split("-")), normalize_numbers(answer)),
    )
    for raw, value, decimals in _values(text):
        value = abs(value)
        result.checked += 1
        if value <= 10 and decimals == 0:
            continue  # small counts and list numbering read as language, not statistics
        tolerance = 0.5 * 10 ** (-decimals)
        if any(abs(value - a) <= max(tolerance, 0.01 * a) for a in allowed):
            continue
        if raw not in result.unsupported:
            result.unsupported.append(raw)
    return result
