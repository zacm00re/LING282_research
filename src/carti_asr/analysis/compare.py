"""Collect every results/<tag>/scores_test.csv into one table and test the hypotheses.

Outputs (all small, committed):
  results/summary/all_scores.csv      every run x style x condition
  results/summary/h1_zeroshot.csv     zero-shot WER per condition, with 95% bootstrap CIs
                                      and differences vs normal-voice baseline
  results/summary/h2_finetune.csv     WER reduction per (style, regime), averaged over seeds,
                                      and the "affect-specific" gain = gain(style) - gain(normal)

Bootstrap: resample *songs* (not clips) with replacement, because clips from the
same song are not independent.

    python -m carti_asr.analysis.compare
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from carti_asr.config import add_common_args, load_config
from carti_asr.eval.metrics import word_scores


def load_predictions(results_dir: Path, tag: str, split="test") -> pd.DataFrame:
    with open(results_dir / tag / f"predictions_{split}.jsonl") as f:
        return pd.DataFrame([json.loads(l) for l in f])


def bootstrap_wer(df: pd.DataFrame, n: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    songs = df.song_id.unique()
    by_song = {s: g for s, g in df.groupby("song_id")}
    stats = []
    for _ in range(n):
        g = pd.concat([by_song[s] for s in rng.choice(songs, size=len(songs), replace=True)])
        stats.append(word_scores(list(g.ref_norm), list(g.hyp_norm)).wer)
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def h1_zeroshot(cfg, results_dir: Path) -> pd.DataFrame:
    df = load_predictions(results_dir, "zeroshot")
    rows = []
    for (style, cond), g in df.groupby(["style", "condition"]):
        wer = word_scores(list(g.ref_norm), list(g.hyp_norm)).wer
        lo, hi = bootstrap_wer(g, cfg["eval"]["bootstrap_samples"], cfg["seed"])
        rows.append(dict(style=style, condition=cond, wer=wer, ci_low=lo, ci_high=hi))
    out = pd.DataFrame(rows)
    base = out.query("style == 'normal' and condition == 'original'").wer.item()
    out["delta_vs_normal"] = out.wer - base
    # Share of the affect gap explained by pitch alone:
    for target in ("baby", "deep"):
        gap = out.query(f"style == '{target}' and condition == 'original'").wer
        shifted = out.query(f"condition == 'shift_to_{target}'").wer
        if len(gap) and len(shifted):
            frac = (shifted.item() - base) / (gap.item() - base) if gap.item() != base else np.nan
            print(f"Pitch explains {frac:.0%} of the {target}-voice WER gap")
    return out


def h2_finetune(results_dir: Path) -> pd.DataFrame:
    zs = pd.read_csv(results_dir / "zeroshot" / "scores_test.csv")
    zs = zs[zs.condition == "original"].set_index("style").wer
    rows = []
    for d in results_dir.iterdir():
        parts = d.name.split("_")
        if not (d / "scores_test.csv").exists() or len(parts) < 3 or parts[0] not in zs.index:
            continue
        style, regime, seed = parts[0], "_".join(parts[1:-1]), parts[-1]
        s = pd.read_csv(d / "scores_test.csv")
        wer = s[(s["style"] == style) & (s.condition == "original")].wer.item()
        rows.append(dict(style=style, regime=regime, seed=seed, wer=wer, gain=zs[style] - wer))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    agg = df.groupby(["style", "regime"]).agg(wer=("wer", "mean"), gain=("gain", "mean"),
                                              gain_sd=("gain", "std"), n_seeds=("seed", "count")).reset_index()
    normal_gain = agg[agg["style"] == "normal"].set_index("regime").gain
    agg["affect_specific_gain"] = agg.gain - agg.regime.map(normal_gain)
    return agg


def main():
    args = add_common_args(argparse.ArgumentParser(description=__doc__)).parse_args()
    cfg = load_config(args.config, args.set)
    results = Path(cfg["paths"]["results_dir"])
    out = results / "summary"; out.mkdir(parents=True, exist_ok=True)
    h1 = h1_zeroshot(cfg, results); h1.to_csv(out / "h1_zeroshot.csv", index=False); print(h1)
    h2 = h2_finetune(results)
    if len(h2):
        h2.to_csv(out / "h2_finetune.csv", index=False); print(h2)


if __name__ == "__main__":
    main()
