#!/usr/bin/env bash
# One-time environment setup (BlueHive or local).
# On BlueHive, run this inside an interactive session, not on the login node:
#   interactive -p gpu -t 1:00:00 --gres=gpu:1     (or: srun -p gpu --gres=gpu:1 --pty bash)
set -euo pipefail

# BlueHive: load conda via modules (check `module avail anaconda` for the exact name)
if command -v module >/dev/null 2>&1; then
  module load anaconda3 || true
fi

conda create -y -n carti python=3.11
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate carti

# System tools from conda-forge so no root access is needed
conda install -y -c conda-forge ffmpeg rubberband espeak-ng

pip install -r requirements.txt
pip install -e .

python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
