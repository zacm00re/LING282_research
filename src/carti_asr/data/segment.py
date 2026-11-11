"""Step 3: cut labeled spans out of the vocal stems.

Reads data/labels/<song_id>.txt (Audacity label export) and writes
data/clips/<style>/<song_id>_<nnn>.wav at 16 kHz mono, plus a manifest
data/manifests/all_clips.csv with one row per clip:

    clip_id, song_id, style, condition, audio_path, text, duration

`condition` is "original" here; the pitch-shift step adds "shift_to_baby" and
"shift_to_deep" rows.

    python -m carti_asr.data.segment
"""
from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import pandas as pd
import soundfile as sf

from carti_asr.config import add_common_args, load_config
from carti_asr.data.download import read_songs


def read_labels(path: Path) -> list[tuple[float, float, str]]:
    spans = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("\\"):   # "\" lines are Audacity spectral info
            continue
        start, end, text = line.split("\t", 2)
        spans.append((float(start), float(end), text.strip()))
    return spans


def main():
    args = add_common_args(argparse.ArgumentParser(description=__doc__)).parse_args()
    cfg = load_config(args.config, args.set)
    p, a = cfg["paths"], cfg["audio"]
    sr = a["sample_rate"]
    songs = read_songs(p["songs_csv"]).set_index("song_id")
    rows = []
    for label_file in sorted(Path(p["labels_dir"]).glob("*.txt")):
        song_id = label_file.stem
        style = songs.loc[song_id, "style"]
        stem = Path(p["stems_dir"]) / f"{song_id}.wav"
        y, _ = librosa.load(stem, sr=sr, mono=True)
        out_dir = Path(p["clips_dir"]) / style; out_dir.mkdir(parents=True, exist_ok=True)
        for i, (start, end, text) in enumerate(read_labels(label_file)):
            dur = end - start
            if not (a["min_clip_seconds"] <= dur <= a["max_clip_seconds"]):
                print(f"[warn] {song_id} span {i} is {dur:.1f}s; skipped")
                continue
            clip_id = f"{song_id}_{i:03d}"
            out = out_dir / f"{clip_id}.wav"
            sf.write(out, y[int(start * sr):int(end * sr)], sr)
            rows.append(dict(clip_id=clip_id, song_id=song_id, style=style, condition="original",
                             audio_path=str(out), text=text, duration=round(dur, 3)))
    df = pd.DataFrame(rows)
    Path(p["manifests_dir"]).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(p["manifests_dir"]) / "all_clips.csv", index=False)
    print(df.groupby("style")["duration"].agg(["count", "sum"]).assign(minutes=lambda d: d["sum"] / 60))


if __name__ == "__main__":
    main()
