LANGUAGES = {
    "en": "English",
    "fr": "French",
    "ar": "Arabic",
    "es": "Spanish",
    "de": "German",
    "pt": "Portuguese",
    "it": "Italian",
}


def language_name(code: str) -> str:
    return LANGUAGES.get(code.lower().split("-")[0], code)


RULES = """You are FireBridge Explainer. You explain one FIREBRIDGE satellite fire analysis (NASA FIRMS active-fire detections from MODIS Terra/Aqua and VIIRS Suomi-NPP/NOAA-20/NOAA-21, harmonized to a common Aqua-equivalent scale) to researchers, analysts, local authorities and students.

Rules:
1. Every number you state must appear in the EVIDENCE below or in a tool result. Do not compute new ratios, percentages, averages or differences yourself; quote the numbers as given (e.g. "29 against a usual 1.5"). If a number is not available, say so plainly.
2. Cite the evidence facts you rely on in square brackets, e.g. [F5]. Tool results need no citation; never put tool names in brackets.
3. Keep these distinctions: a detection is a hot pixel at one pass, not a fire; missing data is not zero; VIIRS sees more small fires than MODIS (an instrument effect, not more burning); the harmonized record estimates what Aqua would report, it is not ground truth; these data do not establish causes.
4. This is not an emergency or warning service: never give evacuation or operational advice.
5. Use the tools for anything not in the evidence: compare_periods for any comparison between years or months (it flags satellite fleet changes that make raw totals incomparable), get_month or get_day for specific dates, unusual_months for the satellite jury.
6. Write in {language}. Lead with the direct answer, then the supporting evidence. Be concise."""


def system_prompt(language: str, digest: str, memories: list[str], conversation_summary: str) -> str:
    parts = [RULES.format(language=language_name(language)), f"EVIDENCE:\n{digest}"]
    if memories:
        parts.append(
            "LONG-TERM MEMORY (what you know about this user and analysis from earlier conversations):\n"
            + "\n".join(f"- {m}" for m in memories)
        )
    if conversation_summary:
        parts.append(f"EARLIER IN THIS CONVERSATION (summary):\n{conversation_summary}")
    return "\n\n".join(parts)


SUMMARY_TASK = """Write the executive summary of this analysis as a JSON object with exactly these keys:
{{"headline": "one sentence, the single most important finding",
  "findings": [{{"title": "short title", "detail": "1-2 sentences with numbers from the evidence", "facts": ["F1"]}}],
  "caveats": ["1 sentence each"],
  "follow_up_questions": ["questions the reader could ask next"]}}
Give 4 to 6 findings ordered by importance: when burning was unusual and whether the satellites agree, how much of the differences between satellites is instrumental, what changed in the observing system (drift, fleet changes, missing records), and what the harmonization tests showed. 2 to 3 caveats, 3 follow-up questions. Every number must come from the evidence; cite fact ids in "facts". Write all text in {language}. Output only the JSON object."""


def summary_task(language: str) -> str:
    return SUMMARY_TASK.format(language=language_name(language))


REPAIR = "These numbers in your answer do not appear in the evidence or tool results: {numbers}. Rewrite the answer using only numbers that do appear there, keeping the same structure and citations."


COMPACT = """Update the running summary of a conversation about a satellite fire analysis.
Keep: the user's goals and questions, conclusions reached, key numbers with their [F#] citations, open questions. Drop pleasantries. At most 120 words. Output only the updated summary.

CURRENT SUMMARY:
{summary}

NEW EXCHANGES:
{exchanges}"""


EXTRACT = """Decide what from this exchange is worth remembering in FUTURE conversations. Output JSON:
{{"memories": [{{"scope": "user" or "analysis", "kind": "preference" | "profile" | "interest" | "conclusion", "content": "...", "importance": 0.0-1.0}}]}}
- user/preference: how the user wants answers (language, length, format). Example: "The user prefers short answers."
- user/profile: the user's role or organisation, only if they stated it. Example: "The user is a forestry analyst at ANEF."
- analysis/interest: a topic the user is investigating in this analysis. Example: "The user is investigating the August 2020 disagreement between satellites."
- analysis/conclusion: an important finding established in the answer, with its key number. Example: "August 2020 is disputed: Aqua recorded no detections while Terra, Suomi-NPP and NOAA-20 recorded 551."
Rules: at most 3 memories; each content is one complete sentence that makes sense without this conversation; skip trivia, greetings and anything not useful later; never store personal data beyond what the user volunteered about their work. Return {{"memories": []}} when nothing qualifies.

USER: {user}
ASSISTANT: {assistant}"""
