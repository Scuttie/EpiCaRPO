#!/bin/bash
#SBATCH --job-name=epicarpo-train
#SBATCH --partition=laal_a6000
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=72:00:00
#SBATCH --output=/home/jewonyeom/EpiCaRPO/logs/epicarpo_%j.log
#SBATCH --error=/home/jewonyeom/EpiCaRPO/logs/epicarpo_%j.err

set -euo pipefail

# ── Paths ──
PROJECT_DIR="/home/jewonyeom/EpiCaRPO"
cd "$PROJECT_DIR"
mkdir -p logs

# Activate venv
source .venv/bin/activate
echo "[$(date)] Python: $(which python)"
echo "[$(date)] GPUs: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"

# ── Configuration ──
MODEL_NAME="Qwen/Qwen3-4B-Base"

TRAIN_DATA="/home/jewonyeom/verl/data/math/train.parquet"
TEST_MATH="/home/jewonyeom/verl/data/math/test.parquet"
TEST_AMC23="/home/jewonyeom/verl/data/amc23/test.parquet"
TEST_AIME2025="/home/jewonyeom/verl/data/aime2025/test.parquet"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

export WANDB_PROJECT="grpo-epicar"
export WANDB_START_METHOD="thread"

# GPU 0: vLLM rollout (time-shared with train model)
# GPU 1: ref model (permanent)
VLLM_GPU_UTIL=0.85

# ── Run Fast Training (CPU offload mode) ──
echo ""
echo "============================================="
echo "[$(date)] GRPO + SFT Calibration (CPU Offload Mode)"
echo "  Model:    ${MODEL_NAME}"
echo "  GPU util: ${VLLM_GPU_UTIL}"
echo "============================================="

python -u train_fast.py \
    --model_name "$MODEL_NAME" \
    --train_data "$TRAIN_DATA" \
    --test_math "$TEST_MATH" \
    --test_amc23 "$TEST_AMC23" \
    --test_aime2025 "$TEST_AIME2025" \
    --vllm_gpu_util "$VLLM_GPU_UTIL" \
    --grpo_lr 1e-6 \
    --kl_coeff 0.01 \
    --clip_range 0.2 \
    --group_size 8 \
    --temperature 0.7 \
    --max_new_tokens 2048 \
    --grpo_grad_accum 4 \
    --sft_lr 1e-5 \
    --sft_epochs 1 \
    --sft_batch_size 2 \
    --sft_grad_accum 4 \
    --num_epochs 3 \
    --batch_size 8 \
    --eval_every 50 \
    --save_every 100 \
    --eval_samples 100 \
    --output_dir ./outputs_grpo_epicar \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name "qwen3_4b_fast_${TIMESTAMP}"

echo ""
echo "[$(date)] Training complete."