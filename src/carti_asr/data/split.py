"""Step 4: split clips 60/20/20 into train/dev/test, BY SONG, within each style.

Splitting by song (not by clip) stops the model from seeing other lines of a
test song during fine-tuning (same beat, same hook, repeated lyrics). The split
is greedy on duration so each split gets close to its target share of minutes.

Writes data/manifests/{train,dev,test}.csv and a committed summary at
results/data/split_summary.csv so the split is documented.

    python -m carti_asr.data.split
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

from carti_asr.config import add_common_args, load_config


def split_songs(song_durations: pd.Series, ratios: dict, seed: int) -> dict[str, str]:
    """Assign each song to a split so per-split minutes approach the target ratios."""
    songs = list(song_durations.index)
    random.Random(seed).shuffle(songs)
    total = song_durations.sum()
    filled = {k: 0.0 for k in ratios}
    assign = {}
    for s in sorted(songs, key=lambda s: -song_durations[s]):   # big songs first
        # put the song where it leaves the largest remaining deficit
        k = max(ratios, key=lambda k: ratios[k] * total - filled[k])
        assign[s] = k
        filled[k] += song_durations[s]
    return assign


def main():
    args = add_common_args(argparse.ArgumentParser(description=__doc__)).parse_args()
    cfg = load_config(args.config, args.set)
    mdir = Path(cfg["paths"]["manifests_dir"])
    clips = pd.read_csv(mdir / "all_clips.csv")
    ratios = cfg["split"]
    parts = []
    for style, df in clips.groupby("style"):
        assign = split_songs(df.groupby("song_id")["duration"].sum(), ratios, cfg["seed"])
        parts.append(df.assign(split=df["song_id"].map(assign)))
    clips = pd.concat(parts)
    for name, df in clips.groupby("split"):
        df.to_csv(mdir / f"{name}.csv", index=False)
    summary = (clips.groupby(["style", "split"])
               .agg(songs=("song_id", "nunique"), clips=("clip_id", "count"),
                    minutes=("duration", lambda d: round(d.sum() / 60, 2))))
    out = Path(cfg["paths"]["results_dir"]) / "data"; out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "split_summary.csv")
    print(summary)


if __name__ == "__main__":
    main()
