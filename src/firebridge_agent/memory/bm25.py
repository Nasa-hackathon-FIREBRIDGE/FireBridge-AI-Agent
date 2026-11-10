"""Small dependency-free BM25 ranking for facts and long-term memories."""

import math
import re
from collections import Counter

_WORD = re.compile(r"\w+", re.UNICODE)
_STOP = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "how",
        "i",
        "in",
        "is",
        "it",
        "its",
        "me",
        "my",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "you",
        "your",
        "le",
        "la",
        "les",
        "de",
        "des",
        "du",
        "un",
        "une",
        "et",
        "en",
        "est",
        "pour",
        "que",
        "qui",
        "dans",
        "au",
        "aux",
    ]
)


def tokenize(text: str) -> list[str]:
    return [t for t in (w.lower() for w in _WORD.findall(text)) if t not in _STOP and (len(t) > 1 or t.isdigit())]


def bm25_scores(query: str, documents: list[str], k1: float = 1.4, b: float = 0.75) -> list[float]:
    docs = [tokenize(d) for d in documents]
    terms = set(tokenize(query))
    if not docs or not terms:
        return [0.0] * len(documents)
    avg = sum(len(d) for d in docs) / len(docs) or 1.0
    df = Counter(t for d in docs for t in set(d) if t in terms)
    n = len(docs)
    scores = []
    for doc in docs:
        tf = Counter(doc)
        score = 0.0
        for term in terms:
            if not tf[term]:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            score += idf * tf[term] * (k1 + 1) / (tf[term] + k1 * (1 - b + b * len(doc) / avg))
        scores.append(score)
    return scores


def similarity(a: str, b: str) -> float:
    """Jaccard similarity of token sets, for de-duplicating memories."""
    sa, sb = set(tokenize(a)), set(tokenize(b))
    return len(sa & sb) / len(sa | sb) if sa and sb else 0.0
