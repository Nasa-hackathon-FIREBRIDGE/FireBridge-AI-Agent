"""Scope and prompt-injection defences for the chat.

Three layers, cheapest first:

1. `screen` refuses messages that plainly try to override the rules, change the assistant's role,
   extract its instructions or fake a system turn, in the app's languages, before any model call
   is spent on them.
2. The system prompt limits the assistant to this analysis and the science needed to read it,
   labels memories, the conversation summary and the app context as data, and has the model reply
   `OUT_OF_SCOPE` to anything else. The server swaps that sentinel for a fixed, localized message.
3. Refused turns stay in the session for audit but are never replayed into later prompts, and
   memory candidates that read as instructions are dropped, so an attempt cannot persist.

The screen is a first filter, not a proof: paraphrased, split or obfuscated attempts are left to
layer 2. Its patterns are written against `normalize`d text (lower case, accents and Arabic
diacritics removed, invisible characters dropped, whitespace collapsed) and are kept narrow,
because a false positive refuses a genuine question.
"""

import re
import unicodedata

OUT_OF_SCOPE = "OUT_OF_SCOPE"
_SENTINEL = re.compile(r"\bOUT_OF_SCOPE\b")
# Soft hyphen, Arabic tatweel and letter mark, zero-width and bidi controls, word joiners, BOM.
_INVISIBLE = re.compile(r"[­ـ؜​-‏‪-‮⁠-⁤﻿]")
# A line that opens a fake turn: "System: ...", "### developer:".
_ROLE_LINE = re.compile(r"(?im)^[\s>*#-]*(?:system|developer)\s*:")

_EN_RULES = r"(?:instructions?|rules(?! of thumb)|prompts?|directives|directions|guidelines|guardrails|programming)"
# "above" only where it ends the clause: "repeat everything above." but not "everything above 50 detections".
_ABOVE = r"(?:above|before this)(?: this (?:line|message))?(?:\s*[.?!,:;]|$| and\b)"
_EN_QUALIFIERS = (
    r"(?: (?:all|any|every|of|the|your|my|these|those|previous|prior|above|earlier|preceding|initial|original|"
    r"system|current|existing|safety)){0,4}"
)

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (kind, re.compile(pattern))
    for kind, pattern in [
        # ── English ──
        ("override", rf"\b(?:ignore|disregard|forget|override|bypass|circumvent){_EN_QUALIFIERS} {_EN_RULES}\b"),
        (
            "override",
            rf"\b(?:ignore|disregard|forget) (?:everything|all|anything) (?:{_ABOVE}|you were told|you have been told)",
        ),
        (
            "override",
            r"\b(?:disable|turn off|bypass|lift|remove) (?:all )?(?:your (?:safety |content )?|(?:the )?(?:safety|content) )"
            r"(?:filters?|restrictions|guardrails|limitations|safeguards)\b",
        ),
        ("role", r"\byou are now (?:a|an|my|called|named|in|free|unrestricted|dan)\b|\byou are no longer\b"),
        ("role", r"\bfrom now on,? you (?:are|will be|will act|must act|act)\b|\byou are dan\b"),
        ("role", r"\b(?:dan|developer|god|jailbreak|unrestricted|unfiltered|uncensored) mode\b|\bdo anything now\b"),
        ("role", r"\bjailbr(?:eak|oken)|\b(?:pretend|behave|act as)\b[^.?!]{0,30}\b(?:unrestricted|uncensored)\b"),
        ("prompt_leak", r"\bsystem prompt\b|\byour system message\b"),
        (
            "prompt_leak",
            r"\b(?:your|the) (?:initial|original|hidden|secret|internal|developer) (?:prompt|instructions)\b",
        ),
        ("prompt_leak", r"\byour (?:full|exact|complete|real) (?:prompt|instructions|rules)\b"),
        (
            "prompt_leak",
            r"\b(?:repeat|print|output|reveal|show|write|copy|recite|dump)\b[^.?!]{0,30}"
            rf"\b(?:everything|the text|all the text|the words) {_ABOVE}",
        ),
        (
            "prompt_leak",
            r"\b(?:reveal|repeat|print|recite|dump|leak|show me|tell me|output|copy) (?:all )?(?:of )?your (?:\w+ )?"
            r"(?:prompt|instructions|directives|configuration)\b",
        ),
        ("fake_turn", r"<\|?(?:im_start|im_end|system|endoftext|eot_id|start_header_id|end_header_id)\|?>"),
        ("fake_turn", r"\[/?inst\]|<</?sys>>|</?(?:system|instructions?)>"),
        ("fake_turn", r"(?:^|\s)#{2,} ?(?:system|new instructions|instructions)\b"),
        ("fake_turn", r"\b(?:new|updated|real|actual|revised|additional) (?:system )?instructions ?:"),
        # ── French ──
        (
            "override",
            r"\b(?:ignore[rsz]?|oublie[rsz]?|neglige[rsz]?|contourne[rsz]?)(?: (?:toutes?|tous|les|tes|vos|ces|mes))* "
            r"(?:instructions?|consignes|regles|directives)\b",
        ),
        ("role", r"\b(?:tu es|vous etes) (?:desormais|maintenant) (?:un|une|mon|ma|en mode)\b"),
        ("role", r"\ba partir de maintenant,? (?:tu es|vous etes|tu seras)\b"),
        ("prompt_leak", r"\b(?:prompt|invite|message|instructions?) (?:du )?systeme\b"),
        (
            "prompt_leak",
            r"\b(?:revele|affiche|montre|repete|recite|donne)[sz]?(?:-moi| moi)? (?:tes|vos|ton|votre) "
            r"(?:prompt|instructions|consignes|regles)\b",
        ),
        # ── Spanish ──
        (
            "override",
            r"\b(?:ignora|ignore|ignorar|olvida|olvide|olvidar|omite|omitir|descarta)"
            r"(?: (?:todas|todos|las|los|tus|sus|mis|anteriores|previas))* (?:instrucciones|reglas|indicaciones|directrices)\b",
        ),
        ("role", r"\b(?:ahora|a partir de ahora),? (?:eres|seras) (?:un|una|mi)\b"),
        ("prompt_leak", r"\b(?:prompt|mensaje|instrucciones) del sistema\b"),
        # ── Portuguese ──
        (
            "override",
            r"\b(?:ignore|ignora|ignorar|esqueca|esquecer|desconsidere|desconsiderar)"
            r"(?: (?:todas|todos|as|os|suas|seus|tuas|minhas|anteriores|previas))* (?:instrucoes|regras|diretrizes)\b",
        ),
        ("role", r"\b(?:agora|a partir de agora),? voce (?:e|sera) (?:um|uma|meu|minha)\b"),
        ("prompt_leak", r"\b(?:prompt|mensagem|instrucoes) do sistema\b"),
        # ── Italian ──
        (
            "override",
            r"\b(?:ignora|ignori|ignorare|dimentica|dimentichi|dimenticare)"
            r"(?: (?:tutte|tutti|le|gli|tue|sue|mie|precedenti))* (?:istruzioni|regole|direttive)\b",
        ),
        ("role", r"\b(?:ora|adesso|da ora in poi),? (?:sei|sarai) (?:un|una|il|la|mio|mia)\b"),
        ("prompt_leak", r"\b(?:prompt|messaggio|istruzioni) di sistema\b"),
        # ── German ──
        (
            "override",
            r"\b(?:ignoriere|ignorier|ignorieren|vergiss|vergessen|missachte|missachten|umgehe)"
            r"(?: (?:sie|alle|deine|ihre|die|vorherigen|bisherigen|obigen|vorigen))* "
            r"(?:anweisungen|regeln|instruktionen|vorgaben|richtlinien)\b",
        ),
        (
            "role",
            r"\b(?:du bist|sie sind) (?:jetzt|nun|ab sofort) (?:ein|eine|mein|meine|im)\b|\bab (?:jetzt|sofort) bist du\b",
        ),
        ("prompt_leak", r"\bsystem-?(?:prompt|nachricht|anweisungen)\b"),
        # ── Arabic (hamza forms fold to bare alef under normalize) ──
        (
            "override",
            r"(?<!\S)(?:تجاهل|انس|انسى|اهمل|تخط)(?: \S+){0,3}? (?:التعليمات|تعليماتك|الاوامر|اوامرك|القواعد|قواعدك|التوجيهات)",
        ),
        ("role", r"من الان فصاعدا (?:انت|ستكون)|انت الان (?:في وضع|مساعد|نموذج)"),
        ("prompt_leak", r"(?:موجه|تعليمات|رسالة|اوامر) النظام"),
    ]
]

_MEMORY_DIRECTIVE = re.compile(
    r"\b(?:assistant|chatbot|the ai|the model|system prompt|instructions?|obey|unrestricted|jailbreak|"
    r"must (?:always|never)|(?:should|must|will) (?:answer|reply|respond to) (?:any|every|all))\b"
)

REFUSAL = {
    "en": "I can only help with this FIREBRIDGE fire analysis: the satellite fire detections, unusual months, "
    "agreement between satellites, the harmonized record and the limits of the data. Please ask a question about it.",
    "fr": "Je ne peux répondre qu'à des questions sur cette analyse FIREBRIDGE : les détections de feux par satellite, "
    "les mois inhabituels, l'accord entre satellites, la série harmonisée et les limites des données. "
    "Posez-moi une question à ce sujet.",
    "es": "Solo puedo ayudar con este análisis de incendios de FIREBRIDGE: las detecciones de incendios por satélite, "
    "los meses inusuales, la concordancia entre satélites, el registro armonizado y los límites de los datos. "
    "Hágame una pregunta sobre él.",
    "ar": "يمكنني المساعدة فقط في تحليل الحرائق هذا من FIREBRIDGE: رصد الحرائق بالأقمار الصناعية، والأشهر غير المعتادة، "
    "ومدى اتفاق الأقمار الصناعية، والسجل الموحَّد، وحدود البيانات. يُرجى طرح سؤال عنه.",
    "de": "Ich kann nur bei dieser FIREBRIDGE-Brandanalyse helfen: den Branderkennungen der Satelliten, "
    "ungewöhnlichen Monaten, der Übereinstimmung zwischen den Satelliten, der harmonisierten Zeitreihe und den "
    "Grenzen der Daten. Bitte stellen Sie eine Frage dazu.",
    "pt": "Só posso ajudar com esta análise de incêndios do FIREBRIDGE: as detecções de incêndios por satélite, "
    "os meses atípicos, a concordância entre satélites, o registro harmonizado e os limites dos dados. "
    "Faça uma pergunta sobre ela.",
    "it": "Posso aiutare solo con questa analisi degli incendi di FIREBRIDGE: i rilevamenti di incendi da satellite, "
    "i mesi insoliti, l'accordo tra i satelliti, la serie armonizzata e i limiti dei dati. Fammi una domanda in merito.",
}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", _INVISIBLE.sub("", text))
    return " ".join("".join(c for c in text if not unicodedata.combining(c)).lower().split())


def screen(*texts: str) -> str | None:
    """The kind of injection attempt in any of `texts` ('override', 'role', 'prompt_leak',
    'fake_turn'), or None when none is plain to see."""
    for text in texts:
        if not text:
            continue
        if _ROLE_LINE.search(unicodedata.normalize("NFKC", _INVISIBLE.sub("", text))):
            return "fake_turn"
        flat = normalize(text)
        for kind, pattern in _PATTERNS:
            if pattern.search(flat):
                return kind
    return None


def is_refusal(answer: str) -> bool:
    """The model declined: the message was out of scope or tried to change its rules."""
    return bool(_SENTINEL.search(answer))


def refusal_text(language: str) -> str:
    return REFUSAL.get(language.lower().split("-")[0], REFUSAL["en"])


def is_safe_memory(content: str) -> bool:
    """A memory describes the user; one that addresses the assistant or its rules would be
    replayed into every later prompt, so it is dropped."""
    return screen(content) is None and not _MEMORY_DIRECTIVE.search(normalize(content))
