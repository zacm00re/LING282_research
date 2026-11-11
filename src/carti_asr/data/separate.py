"""Step 2: isolate vocals with Demucs.

Writes data/stems/<song_id>.wav (vocals only). Runs on GPU if available;
on BlueHive submit jobs/separate.sbatch instead of running on a login node.

    python -m carti_asr.data.separate
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

from carti_asr.config import add_common_args, load_config


def separate(in_path: Path, out_wav: Path, model: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["python", "-m", "demucs", "-n", model, "--two-stems", "vocals",
                        "-o", tmp, str(in_path)], check=True)
        produced = next(Path(tmp).rglob("vocals.wav"))
        shutil.move(produced, out_wav)


def main():
    args = add_common_args(argparse.ArgumentParser(description=__doc__)).parse_args()
    cfg = load_config(args.config, args.set)
    raw, stems = Path(cfg["paths"]["raw_dir"]), Path(cfg["paths"]["stems_dir"])
    stems.mkdir(parents=True, exist_ok=True)
    for audio in sorted(raw.iterdir()):
        out = stems / f"{audio.stem}.wav"
        if out.exists():
            continue
        print(f"[demucs] {audio.name}")
        separate(audio, out, cfg["separation"]["model"])


if __name__ == "__main__":
    main()
