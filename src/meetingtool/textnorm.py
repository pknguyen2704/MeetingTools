"""Light-weight English transcript normalisation.

Whisper already emits punctuation and casing; this module makes the final text
consistent: spacing around punctuation, sentence capitalisation, a terminal
period, ellipsis clean-up, filler-word removal and hallucination filtering.
"""

from __future__ import annotations

import re

_FILLERS = r"(?:um+|uh+|erm+|er|ah+|hmm+|mm+|mhm)"
_FILLER_RE = re.compile(rf"(?<![\w'])({_FILLERS})(?![\w'])[,.]?\s*", re.IGNORECASE)
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?%])")
_MISSING_SPACE_AFTER = re.compile(r"([,;:!?]|(?<!\d)\.)(?=[A-Za-z])")
_DUP_PUNCT = re.compile(r"([,;:])\s*[,;:]+")
_COMMA_BEFORE_END = re.compile(r",\s*([.!?])")
_MULTI_SPACE = re.compile(r"\s{2,}")
_SENTENCE_START = re.compile(r"(^|[.!?]\s+)([a-z])")
_LONE_I = re.compile(r"\bi\b(?=(?:'(?:m|ll|ve|d)\b|\s|[,.!?]|$))")

# Phrases Whisper is known to hallucinate on near-silence / noise.
HALLUCINATIONS = {
    "thank you.",
    "thank you",
    "thanks for watching!",
    "thanks for watching.",
    "thank you for watching.",
    "you",
    "you.",
    "bye.",
    "so.",
    "okay.",
    "subtitles by the amara.org community",
    "please subscribe.",
    "...",
}


def clean_partial(text: str) -> str:
    """Minimal clean-up for the live (not yet final) hypothesis."""
    return _MULTI_SPACE.sub(" ", text).strip()


def normalize(text: str, *, remove_fillers: bool = True) -> str:
    t = text.strip()
    if not t:
        return ""
    t = t.replace("…", "...")
    t = t.strip(" -—")
    if remove_fillers:
        t = _FILLER_RE.sub("", t)
    # Ellipses: trailing -> period, leading -> drop, internal hesitation -> comma.
    t = re.sub(r"^\s*\.{2,}\s*", "", t)
    t = re.sub(r"\s*\.{2,}\s*$", ".", t)
    t = re.sub(r"\s*\.{2,}\s*", ", ", t)
    t = _SPACE_BEFORE_PUNCT.sub(r"\1", t)
    t = _DUP_PUNCT.sub(r"\1", t)
    t = _COMMA_BEFORE_END.sub(r"\1", t)
    t = re.sub(r"([.!?])\1+", r"\1", t)
    t = _MISSING_SPACE_AFTER.sub(r"\1 ", t)
    t = _MULTI_SPACE.sub(" ", t).strip(" ,;:")
    if not t:
        return ""
    t = _LONE_I.sub("I", t)
    t = _SENTENCE_START.sub(lambda m: m.group(1) + m.group(2).upper(), t)
    if t[-1] not in ".!?\"')]":
        t += "."
    elif t[-1] in "\"')]" and len(t) > 1 and t[-2] not in ".!?":
        t += "."
    return t


def is_hallucination(text: str) -> bool:
    return text.strip().lower() in HALLUCINATIONS
