#!/usr/bin/env bash
# Submit every experiment to SLURM in dependency order. Run from the repo root on BlueHive.
set -euo pipefail
mkdir -p logs
ZS=$(sbatch --parsable jobs/zeroshot.sbatch)
echo "zero-shot job: $ZS"
FT=$(sbatch --parsable jobs/finetune_array.sbatch)
echo "fine-tune array: $FT"
sbatch --dependency=afterok:$ZS:$FT jobs/analyze.sbatch
