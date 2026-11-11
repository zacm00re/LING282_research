"""Figures for the presentation and writeup (saved to results/figures/).

  fig1_zeroshot_wer.png   WER by condition (normal, shift_to_baby, baby, shift_to_deep, deep) with CIs
  fig2_finetune_gain.png  WER reduction by style x regime (encoder_only vs decoder_only), seed error bars
  fig3_error_types.png    stacked sub/del/ins shares per style  (TODO)

    python -m carti_asr.analysis.plots
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from carti_asr.config import add_common_args, load_config

ORDER = [("normal", "original"), ("normal", "shift_to_baby"), ("baby", "original"),
         ("normal", "shift_to_deep"), ("deep", "original")]


def fig_zeroshot(summary: Path, out: Path):
    df = pd.read_csv(summary / "h1_zeroshot.csv").set_index(["style", "condition"])
    df = df.loc[[k for k in ORDER if k in df.index]]
    labels = [c if s == "normal" and c != "original" else s for s, c in df.index]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(labels, df.wer, yerr=[df.wer - df.ci_low, df.ci_high - df.wer], capsize=4)
    ax.set_ylabel("WER (zero-shot)"); ax.set_title("whisper-small by vocal style")
    fig.tight_layout(); fig.savefig(out / "fig1_zeroshot_wer.png", dpi=200)


def fig_finetune(summary: Path, out: Path):
    p = summary / "h2_finetune.csv"
    if not p.exists():
        return
    df = pd.read_csv(p).pivot(index="style", columns="regime", values="gain")
    df.plot.bar(figsize=(6, 3.5), ylabel="WER reduction vs zero-shot", rot=0)
    plt.tight_layout(); plt.savefig(out / "fig2_finetune_gain.png", dpi=200)


def main():
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    results = Path(cfg["paths"]["results_dir"])
    out = results / "figures"; out.mkdir(parents=True, exist_ok=True)
    fig_zeroshot(results / "summary", out)
    fig_finetune(results / "summary", out)
    print(f"Figures written to {out}")


if __name__ == "__main__":
    main()
