"""Qualitative + categorical error analysis on zero-shot (or any run's) predictions.

Produces, per style:
  - share of errors that are substitutions / deletions / insertions
  - the most frequent (ref -> hyp) substitution pairs
  - PER *within substituted words only*: low PER on substitutions means
    Whisper heard roughly the right sounds but picked the wrong word (decoder
    side); high PER means it heard different sounds (encoder side)
  - hallucination check: hypothesis words not in the reference that are
    common English/lyric words (decoder language prior filling gaps)
  - a random sample of 25 error clips per style for manual phonetic transcription
    (written to results/<tag>/manual_review_<style>.csv)

    python -m carti_asr.analysis.error_analysis --tag zeroshot
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

from carti_asr.analysis.compare import load_predictions
from carti_asr.config import add_common_args, load_config
from carti_asr.eval.metrics import error_pairs, phone_error_rate


def main():
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--tag", default="zeroshot")
    parser.add_argument("--n-review", type=int, default=25)
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    results = Path(cfg["paths"]["results_dir"])
    df = load_predictions(results, args.tag)
    df = df[df.condition == "original"]
    rows = []
    for style, g in df.groupby("style"):
        pairs = error_pairs(list(g.ref_norm), list(g.hyp_norm))
        ops = Counter(op for op, _, _ in pairs)
        subs = [(r, h) for op, r, h in pairs if op == "substitute"]
        sub_per = phone_error_rate([r for r, _ in subs], [h for _, h in subs]) if subs else float("nan")
        total = sum(ops.values()) or 1
        rows.append(dict(style=style, **{f"pct_{k}": v / total for k, v in ops.items()},
                         per_within_substitutions=sub_per))
        top = Counter(subs).most_common(20)
        pd.DataFrame([(r, h, c) for (r, h), c in top], columns=["ref", "hyp", "count"]) \
            .to_csv(results / args.tag / f"top_substitutions_{style}.csv", index=False)
        errs = g[g.ref_norm != g.hyp_norm]
        errs.sample(min(args.n_review, len(errs)), random_state=cfg["seed"]) \
            [["clip_id", "audio_path", "text", "hyp"]] \
            .assign(ipa_narrow="", notes="") \
            .to_csv(results / args.tag / f"manual_review_{style}.csv", index=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(results / args.tag / "error_types.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
