#!/bin/bash
#SBATCH --job-name=carti-ft
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --array=0-17               # 3 styles x 2 regimes x 3 seeds = 18 runs
#SBATCH --output=logs/%x_%A_%a.out
# H2: encoder-only vs decoder-only fine-tuning, one run per array index, then test-set eval.
set -euo pipefail
module load anaconda3 || true
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate carti

STYLES=(normal baby deep)
REGIMES=(encoder_only decoder_only)
SEEDS=(13 42 7)
i=$SLURM_ARRAY_TASK_ID
STYLE=${STYLES[$(( i / 6 ))]}
REGIME=${REGIMES[$(( (i / 3) % 2 ))]}
SEED=${SEEDS[$(( i % 3 ))]}
RUN=${STYLE}_${REGIME}_s${SEED}
echo "Run $RUN"

python -m carti_asr.train.finetune --style "$STYLE" --regime "$REGIME" --seed "$SEED"
python -m carti_asr.eval.transcribe --split test --model "checkpoints/$RUN/best" --tag "$RUN"
