"""Text normalization applied to BOTH references and hypotheses before scoring.

WER is very sensitive to normalization (casing, punctuation, "gonna" vs
"gon'"). We use one function everywhere so every condition is scored the same.
"""
from __future__ import annotations

import re

# Map variant spellings to one canonical form. Extend as you label.
SPELLING_MAP = {
    "gon'": "gonna", "gon": "gonna",
    "finna": "finna", "fixin to": "finna",
    "lil'": "lil", "lil": "lil",
    "'cause": "because", "cause": "because", "cuz": "because",
    "ya": "you", "y'all": "yall",
    "tryna": "trying to",
    "ok": "okay",
    "nothin'": "nothing", "somethin'": "something",
}

_GENIUS_HEADER = re.compile(r"\[[^\]]*\]")          # [Chorus], [Verse 1: Playboi Carti]
_ADLIB = re.compile(r"\([^)]*\)")                    # (what), (slatt)
_PUNCT = re.compile(r"[^\w\s']")


def normalize(text: str, drop_adlibs: bool = False) -> str:
    text = _GENIUS_HEADER.sub(" ", text)
    if drop_adlibs:
        text = _ADLIB.sub(" ", text)
    text = text.lower().replace("’", "'")
    text = _PUNCT.sub(" ", text)
    words = []
    for w in text.split():
        w = SPELLING_MAP.get(w, w)
        # g-dropping: "runnin'" -> "running" (only for -in' endings)
        if w.endswith("in'"):
            w = w[:-1] + "g"
        words.append(w.strip("'"))
    return " ".join(w for w in words if w)
