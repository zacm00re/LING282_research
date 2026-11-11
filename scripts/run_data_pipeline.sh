#!/usr/bin/env bash
# Rebuild the whole dataset from data/metadata/songs.csv + data/labels/*.txt.
# Steps 1 and 2 need internet / a GPU; on BlueHive run step 2 via jobs/separate.sbatch.
set -euo pipefail
python -m carti_asr.data.download          # 1. audio -> data/raw/
python -m carti_asr.data.separate          # 2. Demucs vocals -> data/stems/
python -m carti_asr.data.segment           # 3. labeled spans -> data/clips/ + manifests/all_clips.csv
python -m carti_asr.data.split             # 4. song-level 60/20/20 -> manifests/{train,dev,test}.csv
python -m carti_asr.data.pitch measure     # 5a. median F0 per style -> results/pitch/f0_summary.csv
echo "Now copy the semitone values into configs/default.yaml (pitch_shift.targets), then run:"
echo "  python -m carti_asr.data.pitch shift   # 5b. formant-preserving shifted normal-voice test clips"
