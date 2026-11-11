"""Step 5: pitch-shift control.

Two subcommands:

  measure  Estimate F0 (librosa.pyin) on every original clip, then report the
           median F0 per style and the semitone shift that moves normal-voice
           median F0 to the baby/deep median. Writes results/pitch/f0_summary.csv.

  shift    Pitch-shift every normal-voice TEST clip by the configured semitones,
           preserving formants (rubberband -F), so pitch changes but vocal-tract
           resonances (and so vowel quality) do not. Adds rows with condition
           "shift_to_baby" / "shift_to_deep" to data/manifests/test.csv.

Why formant preservation matters: a plain resample/ffmpeg asetrate shift moves
formants too, which would change vowel quality and confound "pitch" with
"phonetics". The control must isolate F0.

    python -m carti_asr.data.pitch measure
    python -m carti_asr.data.pitch shift
"""
from __future__ import annotations

import argparse
import math
import subprocess
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

from carti_asr.config import add_common_args, load_config


def clip_f0(path: str, sr: int) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr)
    f0, voiced, _ = librosa.pyin(y, fmin=50, fmax=800, sr=sr)
    return f0[voiced & ~np.isnan(f0)]


def measure(cfg):
    mdir = Path(cfg["paths"]["manifests_dir"])
    clips = pd.read_csv(mdir / "all_clips.csv")
    clips = clips[clips.condition == "original"]
    rows = []
    for style, df in clips.groupby("style"):
        f0 = np.concatenate([clip_f0(p, cfg["audio"]["sample_rate"]) for p in df.audio_path])
        rows.append(dict(style=style, median_f0_hz=float(np.median(f0)),
                         iqr_low=float(np.percentile(f0, 25)), iqr_high=float(np.percentile(f0, 75))))
    summary = pd.DataFrame(rows).set_index("style")
    base = summary.loc["normal", "median_f0_hz"]
    summary["semitones_from_normal"] = 12 * np.log2(summary["median_f0_hz"] / base)
    out = Path(cfg["paths"]["results_dir"]) / "pitch"; out.mkdir(parents=True, exist_ok=True)
    summary.round(2).to_csv(out / "f0_summary.csv")
    print(summary.round(2))
    print("\nCopy semitones_from_normal for baby/deep into configs/default.yaml -> pitch_shift.targets")


def rubberband_shift(in_wav: str, out_wav: Path, semitones: float, preserve_formants: bool):
    cmd = ["rubberband", "-p", f"{semitones:.3f}"]
    if preserve_formants:
        cmd.append("-F")
    subprocess.run(cmd + [in_wav, str(out_wav)], check=True, capture_output=True)


def shift(cfg):
    ps = cfg["pitch_shift"]
    mdir = Path(cfg["paths"]["manifests_dir"])
    test = pd.read_csv(mdir / "test.csv")
    test = test[test.condition == "original"]   # idempotent: rebuild shifted rows each time
    normal = test[test.style == "normal"]
    new_rows = []
    for target, semis in ps["targets"].items():
        if semis is None or (isinstance(semis, float) and math.isnan(semis)):
            raise SystemExit(f"pitch_shift.targets.{target} is unset; run `measure` first.")
        out_dir = Path(cfg["paths"]["shifted_dir"]) / f"to_{target}"; out_dir.mkdir(parents=True, exist_ok=True)
        for r in normal.itertuples():
            out = out_dir / f"{r.clip_id}.wav"
            rubberband_shift(r.audio_path, out, semis, ps["preserve_formants"])
            new_rows.append({**r._asdict(), "condition": f"shift_to_{target}", "audio_path": str(out)})
    new = pd.DataFrame(new_rows).drop(columns="Index")
    pd.concat([test, new]).to_csv(mdir / "test.csv", index=False)
    print(f"Added {len(new)} pitch-shifted test clips.")


def main():
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("command", choices=["measure", "shift"])
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    {"measure": measure, "shift": shift}[args.command](cfg)


if __name__ == "__main__":
    main()
