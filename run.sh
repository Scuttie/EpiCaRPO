#!/bin/bash
# ============================================================
# GRPO + EpiCaR Training Script
# Model: Qwen/Qwen3-4B-Base
# Rollout: vLLM (batched) / GRPO+SFT: PyTorch
# GPUs: 2x A100 80GB
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
export PATH="$HOME/.local/bin:$PATH"

# Activate venv (skip for setup)
if [ "${1}" != "setup" ]; then
    if [ -f "${SCRIPT_DIR}/.venv/bin/activate" ]; then
        source "${SCRIPT_DIR}/.venv/bin/activate"
        echo "Activated venv: $(which python)"
    else
        echo "ERROR: .venv not found. Run: bash run.sh setup"
        exit 1
    fi
fi

# ---- Configuration ----
export CUDA_VISIBLE_DEVICES=0,1
export WANDB_PROJECT="grpo-epicar"
export WANDB_START_METHOD="thread"

MODEL_NAME="Qwen/Qwen3-4B-Base"
OUTPUT_DIR="./outputs_grpo_epicar"

TRAIN_DATA="/mnt/chatbot30TB/jewonyeom/verl_qwen3/verl/data/math/train.parquet"
TEST_MATH="/mnt/chatbot30TB/jewonyeom/verl_qwen3/verl/data/math/test.parquet"
TEST_AMC23="/mnt/chatbot30TB/jewonyeom/verl_qwen3/verl/data/amc23/test.parquet"
TEST_AIME2025="/mnt/chatbot30TB/jewonyeom/verl_qwen3/verl/data/aime2025/test.parquet"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# ============================================================
# Setup
# ============================================================
run_setup() {
    echo "============================================="
    echo "Setting up environment"
    echo "============================================="

    if ! command -v uv &> /dev/null; then
        echo "Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi

    uv venv .venv
    source .venv/bin/activate

    # Step 1: torch + build deps
    uv pip install torch setuptools wheel

    # Step 2: flash-attn (optional, may fail)
    uv pip install flash-attn --no-build-isolation || echo "flash-attn install failed (OK, using SDPA)"

    # Step 3: vLLM + other deps
    uv pip install -r requirements.txt

    echo ""
    echo "Setup complete. Run: bash run.sh standalone"
}

# ============================================================
# Standalone Training (vLLM rollout + PyTorch GRPO/SFT)
# ============================================================
run_standalone() {
    LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"

    echo "============================================="
    echo "GRPO + SFT Calibration (vLLM rollout)"
    echo "GPUs: ${CUDA_VISIBLE_DEVICES} (2x A100)"
    echo "Log:  ${LOG_FILE}"
    echo "============================================="

    nohup python -u train.py \
        --model_name ${MODEL_NAME} \
        --train_data ${TRAIN_DATA} \
        --test_math ${TEST_MATH} \
        --test_amc23 ${TEST_AMC23} \
        --test_aime2025 ${TEST_AIME2025} \
        --vllm_gpu_util 0.45 \
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
        --output_dir ${OUTPUT_DIR} \
        --wandb_project ${WANDB_PROJECT} \
        --wandb_run_name "qwen3_4b_grpo_epicar_${TIMESTAMP}" \
        > "${LOG_FILE}" 2>&1 &

    PID=$!
    echo "${PID}" > "${LOG_DIR}/train.pid"
    echo ""
    echo "Started (PID: ${PID})"
    echo "  Monitor:  tail -f ${LOG_FILE}"
    echo "  Kill:     kill ${PID}"
}

# ============================================================
# VERL Training (Ray distributed GRPO + SFT callback)
# ============================================================
run_verl() {
    LOG_FILE="${LOG_DIR}/train_verl_${TIMESTAMP}.log"

    echo "============================================="
    echo "VERL GRPO + SFT Calibration"
    echo "GPUs: ${CUDA_VISIBLE_DEVICES} (2x A100)"
    echo "Log:  ${LOG_FILE}"
    echo "============================================="

    nohup python -u train_verl.py \
        --config verl_config.yaml \
        --sft_lr 1e-5 \
        --sft_micro_batch_size 2 \
        --sft_max_length 1024 \
        > "${LOG_FILE}" 2>&1 &

    PID=$!
    echo "${PID}" > "${LOG_DIR}/train_verl.pid"
    echo ""
    echo "Started (PID: ${PID})"
    echo "  Monitor:  tail -f ${LOG_FILE}"
    echo "  Kill:     kill ${PID}"
}

# ============================================================
# Evaluate checkpoint
# ============================================================
run_eval() {
    MODEL_PATH=${1:-"${OUTPUT_DIR}/final_model"}
    EVAL_OUTPUT=${2:-"./eval_results"}
    LOG_FILE="${LOG_DIR}/eval_${TIMESTAMP}.log"

    echo "============================================="
    echo "Evaluating: ${MODEL_PATH}"
    echo "Log:  ${LOG_FILE}"
    echo "============================================="

    nohup python -u eval_checkpoint.py \
        --model_path ${MODEL_PATH} \
        --output_dir ${EVAL_OUTPUT} \
        --max_new_tokens 2048 \
        --max_samples 0 \
        --wandb_project ${WANDB_PROJECT} \
        > "${LOG_FILE}" 2>&1 &

    PID=$!
    echo "${PID}" > "${LOG_DIR}/eval.pid"
    echo ""
    echo "Started (PID: ${PID})"
    echo "  Monitor:  tail -f ${LOG_FILE}"
    echo "  Kill:     kill ${PID}"
}

# ============================================================
# Fast Training (optimizer state preserved across steps)
# ============================================================
run_fast() {
    LOG_FILE="${LOG_DIR}/train_fast_${TIMESTAMP}.log"

    echo "============================================="
    echo "GRPO + SFT Calibration (vLLM Server Mode)"
    echo "GPUs: ${CUDA_VISIBLE_DEVICES} (GPU0=train, GPU1=vllm+ref)"
    echo "Log:  ${LOG_FILE}"
    echo "============================================="

    nohup python -u train_fast.py \
        --model_name ${MODEL_NAME} \
        --train_data ${TRAIN_DATA} \
        --test_math ${TEST_MATH} \
        --test_amc23 ${TEST_AMC23} \
        --test_aime2025 ${TEST_AIME2025} \
        --vllm_gpu_util 0.65 \
        --vllm_refresh_every ${2:-5} \
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
        --output_dir ${OUTPUT_DIR} \
        --wandb_project ${WANDB_PROJECT} \
        --wandb_run_name "qwen3_4b_fast_${TIMESTAMP}" \
        > "${LOG_FILE}" 2>&1 &

    PID=$!
    echo "${PID}" > "${LOG_DIR}/train_fast.pid"
    echo ""
    echo "Started (PID: ${PID})"
    echo "  Monitor:  tail -f ${LOG_FILE}"
    echo "  Kill:     kill ${PID}"
}

# ============================================================
# Parse command
# ============================================================
case "${1:-help}" in
    setup)      run_setup ;;
    standalone) run_standalone ;;
    fast)       run_fast ;;
    verl)       run_verl ;;
    eval)       run_eval "$2" "$3" ;;
    *)
        echo "Usage: bash run.sh {setup|standalone|fast|verl|eval}"
        echo ""
        echo "  setup       - Create .venv, install torch + vLLM + deps"
        echo "  standalone  - GRPO + SFT with vLLM rollout [slow] (nohup)"
        echo "  fast        - GRPO + SFT with vLLM server [recommended] (nohup)"
        echo "  verl        - VERL framework GRPO + SFT (nohup)"
        echo "  eval [path] - Evaluate checkpoint (nohup)"
        exit 1
        ;;
esac
