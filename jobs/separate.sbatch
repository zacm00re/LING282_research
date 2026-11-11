#!/bin/bash
#SBATCH --job-name=carti-demucs
#SBATCH --partition=gpu            # check `sinfo` for BlueHive's GPU partition name
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
module load anaconda3 || true
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate carti
python -m carti_asr.data.separate
