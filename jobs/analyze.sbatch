#!/bin/bash
#SBATCH --job-name=carti-analyze
#SBATCH --partition=standard       # CPU is enough
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
module load anaconda3 || true
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate carti
python -m carti_asr.analysis.compare
python -m carti_asr.analysis.error_analysis --tag zeroshot
python -m carti_asr.analysis.plots
