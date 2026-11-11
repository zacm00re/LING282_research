#!/usr/bin/env bash
# Tiny end-to-end run (a few clips, 20 training steps) to check the pipeline before real jobs.
set -euo pipefail
python -m pytest -q tests
python -m carti_asr.eval.transcribe --split dev --tag smoke_zeroshot --no-per
python -m carti_asr.train.finetune --style normal --regime decoder_only \
  --set finetune.max_steps=20 finetune.eval_steps=10 paths.checkpoints_dir=checkpoints/smoke
