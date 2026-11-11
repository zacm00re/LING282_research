"""Scoring: WER and CER (jiwer) plus PER (phonemizer + espeak-ng).

- WER/CER: standard orthographic error rates, with substitution/deletion/
  insertion counts kept separately because the error *type* matters for the
  "where" question (deletions on mumbled syllables vs substitutions on
  misheard vowels vs insertions from decoder hallucination).
- PER: reference and hypothesis are both converted to IPA phones with
  espeak-ng, then scored with edit distance over phone tokens. This measures
  how phonetically close Whisper's *words* are to the true words (e.g.
  "sip" for "ship" is 1 phone off, "cat" for "ship" is 3). It does NOT measure
  what Carti actually pronounced; for that see eval/phone_recognizer.py.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import jiwer


@dataclass
class Scores:
    wer: float
    cer: float
    sub: int
    dele: int
    ins: int
    hits: int
    n_ref_words: int

    def as_dict(self):
        return asdict(self)


def word_scores(refs: list[str], hyps: list[str]) -> Scores:
    out = jiwer.process_words(refs, hyps)
    cer = jiwer.cer(refs, hyps)
    n = out.hits + out.substitutions + out.deletions
    return Scores(wer=out.wer, cer=cer, sub=out.substitutions, dele=out.deletions,
                  ins=out.insertions, hits=out.hits, n_ref_words=n)


_phonemizer_backend = None


def to_phones(texts: list[str]) -> list[str]:
    """Space-separated IPA phones for each text (word boundaries dropped)."""
    global _phonemizer_backend
    from phonemizer.backend import EspeakBackend
    from phonemizer.separator import Separator
    if _phonemizer_backend is None:
        _phonemizer_backend = EspeakBackend("en-us", preserve_punctuation=False, with_stress=False)
    sep = Separator(phone=" ", word=" ", syllable="")
    return [" ".join(p.split()) for p in _phonemizer_backend.phonemize(texts, separator=sep, strip=True)]


def phone_error_rate(refs: list[str], hyps: list[str]) -> float:
    # jiwer splits on whitespace, so phone strings are scored phone-by-phone.
    ref_p, hyp_p = to_phones(refs), to_phones(hyps)
    pairs = [(r, h) for r, h in zip(ref_p, hyp_p) if r.strip()]
    return jiwer.wer([r for r, _ in pairs], [h if h.strip() else "<empty>" for _, h in pairs])


def error_pairs(refs: list[str], hyps: list[str]) -> list[tuple[str, str, str]]:
    """List of (op, ref_word, hyp_word) for every non-matching alignment step.
    Feed this to analysis/error_analysis.py for confusion tables."""
    out = jiwer.process_words(refs, hyps)
    pairs = []
    for ref, hyp, chunks in zip(out.references, out.hypotheses, out.alignments):
        for c in chunks:
            if c.type == "equal":
                continue
            r = " ".join(ref[c.ref_start_idx:c.ref_end_idx])
            h = " ".join(hyp[c.hyp_start_idx:c.hyp_end_idx])
            pairs.append((c.type, r, h))
    return pairs
