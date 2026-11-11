"""Step 1: download audio for every song in data/metadata/songs.csv.

Audio is saved to data/raw/<song_id>.<ext> and is git-ignored. We keep the
best available audio stream and do NOT re-encode, so every song goes through
the same downstream conversion (Demucs -> 16 kHz mono WAV).

    python -m carti_asr.data.download
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pandas as pd

from carti_asr.config import add_common_args, load_config


def read_songs(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path, comment="#").dropna(subset=["song_id"])


def download_one(url: str, out_stem: Path) -> None:
    cmd = ["yt-dlp", "-x", "--audio-format", "best", "--no-playlist",
           "-o", f"{out_stem}.%(ext)s", url]
    subprocess.run(cmd, check=True)


def main():
    args = add_common_args(argparse.ArgumentParser(description=__doc__)).parse_args()
    cfg = load_config(args.config, args.set)
    raw = Path(cfg["paths"]["raw_dir"]); raw.mkdir(parents=True, exist_ok=True)
    songs = read_songs(cfg["paths"]["songs_csv"])
    for row in songs.itertuples():
        if any(raw.glob(f"{row.song_id}.*")):
            print(f"[skip] {row.song_id} already downloaded")
            continue
        print(f"[get ] {row.song_id}  {row.title}")
        download_one(row.audio_url, raw / row.song_id)


if __name__ == "__main__":
    main()
