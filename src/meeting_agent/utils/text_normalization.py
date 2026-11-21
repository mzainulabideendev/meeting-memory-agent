"""Text normalisation helpers shared by retrieval, extraction and validation.

The most important rule in this project: normalisation is only ever used for
*comparison*.  Any text that is stored or returned as a citation keeps the
original characters from the transcript file.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

_WHITESPACE = re.compile(r"\s+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'\[\]])")
_FILLER_WORDS = {
    "uh", "um", "erm", "ah", "eh", "hmm", "mm", "mmm", "hm", "uhh", "umm",
    "like", "you know", "i mean", "sort of", "kind of",
}

STOPWORDS = frozenset(
    """
    a an the and or but if then than that this these those of to in on at for with
    from by as is are was were be been being do does did doing have has had having
    i you he she it we they me him her us them my your his its our their what which
    who whom whose when where why how all any both each few more most other some
    such no nor not only own same so too very can will just should now about into
    over after before between during through out up down off again further once
    here there also would could may might must shall
    """.split()
)

# Conversational fillers that carry no retrieval signal but appear constantly
# in spoken transcripts ("uh", "um", ...).  Kept separate from STOPWORDS because
# they should never be treated as quoted evidence terms.
FILLERS = frozenset({"uh", "um", "erm", "ah", "eh", "hmm", "mm", "mmm", "hm"})


def collapse_whitespace(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


def normalize_for_match(text: str) -> str:
    """Normalise text for citation comparison.

    The assessment states that case and spacing are ignored.  We additionally
    normalise typographic quotes/dashes and Unicode forms, because transcripts
    mix curly and straight punctuation.  Word order and punctuation marks are
    preserved.
    """
    text = unicodedata.normalize("NFKC", text)
    text = (
        text.replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u00a0", " ")
    )
    text = collapse_whitespace(text)
    return text.casefold()


def contains_quote(haystack: str, quote: str) -> bool:
    """True when ``quote`` occurs verbatim (case/space-insensitive) in ``haystack``."""
    if not quote or len(normalize_for_match(quote).split()) < 3:
        return False
    return normalize_for_match(quote) in normalize_for_match(haystack)


def split_sentences(text: str) -> list[str]:
    """Sentence splitting tuned for spoken transcripts.

    Falls back to smaller units when a 'sentence' is extremely long, which is
    common when a speaker turn contains no punctuation.
    """
    text = collapse_whitespace(text)
    if not text:
        return []
    pieces = [p.strip() for p in _SENTENCE_SPLIT.split(text) if p.strip()]
    out: list[str] = []
    for piece in pieces:
        words = piece.split()
        if len(words) <= 60:
            out.append(piece)
            continue
        # hard split long unpunctuated runs at ~35 words
        for i in range(0, len(words), 35):
            out.append(" ".join(words[i : i + 35]))
    return out


def tokenize_words(text: str) -> list[str]:
    """Tokenise for comparison.

    Transcripts spell "LCD" as ``L_C_D_``; underscores and possessive ``'s``
    are removed so that ``lcd`` and ``l_c_d_`` compare equal.  This never
    changes stored or cited text.
    """
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-_/]*", text.lower())
    normalized: list[str] = []
    for token in tokens:
        token = token.replace("_", "")
        if token.endswith("'s"):
            token = token[:-2]
        token = token.strip("-_/")
        if token:
            normalized.append(token)
    return normalized


def content_terms(text: str, *, drop_fillers: bool = True) -> list[str]:
    stop = STOPWORDS | (FILLERS if drop_fillers else frozenset())
    return [t for t in tokenize_words(text) if t not in stop and len(t) > 1]


def term_overlap(query_terms: Iterable[str], text: str) -> float:
    terms = list(dict.fromkeys(query_terms))
    if not terms:
        return 0.0
    haystack = set(tokenize_words(text))
    hits = sum(1 for t in terms if t in haystack)
    return hits / len(terms)


def clean_quote_for_prompt(text: str, max_words: int = 45) -> str:
    """Trim an evidence sentence to a citation-sized, still-exact quotation.

    Never use this on generated text that will be returned as a citation: the
    result is verbatim only when the input was verbatim.
    """
    text = collapse_whitespace(text)
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).rstrip(",;:-")
