#!/bin/bash
#SBATCH --job-name=carti-zeroshot
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
# H1: zero-shot whisper-small on test (original + pitch-shifted conditions), plus dev for reference.
set -euo pipefail
module load anaconda3 || true
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate carti
python -m carti_asr.eval.transcribe --split test --tag zeroshot
python -m carti_asr.eval.transcribe --split dev  --tag zeroshot
