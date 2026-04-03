#!/bin/bash
#SBATCH --job-name=epicarpo-verl
#SBATCH --partition=b200           # B200 파티션으로 변경 필요
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=/home/jewonyeom/EpiCaRPO/logs/verl_8gpu_%j.log
#SBATCH --error=/home/jewonyeom/EpiCaRPO/logs/verl_8gpu_%j.err

set -euo pipefail

# ── Paths ──
PROJECT_DIR="/home/jewonyeom/EpiCaRPO"
cd "$PROJECT_DIR"
mkdir -p logs

# Activate venv
source .venv/bin/activate
echo "[$(date)] Python: $(which python)"
echo "[$(date)] GPUs:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || echo "N/A"

export WANDB_PROJECT="grpo-epicar"
export WANDB_START_METHOD="thread"

# ── Run VERL GRPO + SFT Training ──
echo ""
echo "============================================="
echo "[$(date)] VERL GRPO + SFT Calibration (8× GPU)"
echo "  Config: verl_config_8gpu.yaml"
echo "============================================="

python -u train_verl_8gpu.py \
    --config-path . \
    --config-name verl_config_8gpu

echo ""
echo "[$(date)] Training complete."
