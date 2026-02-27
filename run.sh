#!/bin/bash
# ============================================================
# GRPO + EpiCaR (SFT Calibration) Training Script
# ============================================================
# Model: Qwen/Qwen3-4B-Base
# Method: GRPO + Verbalized Confidence SFT
# GPUs: 2x A100
# ============================================================

set -e

# ---- Environment ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
export PATH="$HOME/.local/bin:$PATH"

# Activate venv (skip for setup command)
if [ "${1}" != "setup" ]; then
    if [ -f "${SCRIPT_DIR}/.venv/bin/activate" ]; then
        source "${SCRIPT_DIR}/.venv/bin/activate"
        echo "Activated venv: $(which python)"
    else
        echo "ERROR: .venv not found. Run setup first:"
        echo "  cd ${SCRIPT_DIR}"
        echo "  bash run.sh setup"
        exit 1
    fi
fi

# ---- Configuration ----
export CUDA_VISIBLE_DEVICES=0,1
export WANDB_PROJECT="grpo-epicar"

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
# Setup: create venv + install deps
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

    # Step 1: Install torch + build deps first
    uv pip install torch setuptools wheel

    # Step 2: Install flash-attn (needs torch+setuptools at build time)
    uv pip install flash-attn --no-build-isolation

    # Step 3: Install remaining packages
    uv pip install \
        transformers \
        accelerate \
        pandas \
        pyarrow \
        scikit-learn \
        matplotlib \
        wandb \
        sympy \
        tqdm

    echo ""
    echo "Setup complete. Now run:"
    echo "  bash run.sh standalone"
}

# ============================================================
# Standalone Training (nohup background)
# ============================================================
run_standalone() {
    LOG_FILE="${LOG_DIR}/train_standalone_${TIMESTAMP}.log"

    echo "============================================="
    echo "Running Standalone GRPO + SFT Calibration"
    echo "GPUs: ${CUDA_VISIBLE_DEVICES} (2x A100)"
    echo "Log:  ${LOG_FILE}"
    echo "============================================="

    nohup python -u train.py \
        --model_name ${MODEL_NAME} \
        --train_data ${TRAIN_DATA} \
        --test_math ${TEST_MATH} \
        --test_amc23 ${TEST_AMC23} \
        --test_aime2025 ${TEST_AIME2025} \
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
    echo "${PID}" > "${LOG_DIR}/train_standalone.pid"
    echo ""
    echo "Started training in background (PID: ${PID})"
    echo ""
    echo "  Monitor:  tail -f ${LOG_FILE}"
    echo "  Kill:     kill ${PID}"
    echo "  Status:   ps -p ${PID}"
}

# ============================================================
# VERL-based Training (nohup background)
# ============================================================
run_verl() {
    LOG_FILE="${LOG_DIR}/train_verl_${TIMESTAMP}.log"

    echo "============================================="
    echo "Running VERL GRPO + SFT Calibration"
    echo "GPUs: ${CUDA_VISIBLE_DEVICES} (2x A100)"
    echo "Log:  ${LOG_FILE}"
    echo "============================================="

    nohup python -u -m verl.trainer.main_ppo \
        --config verl_config.yaml \
        --reward_fn reward_fn.compute_score \
        > "${LOG_FILE}" 2>&1 &

    PID=$!
    echo "${PID}" > "${LOG_DIR}/train_verl.pid"
    echo ""
    echo "Started training in background (PID: ${PID})"
    echo ""
    echo "  Monitor:  tail -f ${LOG_FILE}"
    echo "  Kill:     kill ${PID}"
}

# ============================================================
# Evaluate a checkpoint (nohup background)
# ============================================================
run_eval() {
    MODEL_PATH=${1:-"${OUTPUT_DIR}/final_model"}
    EVAL_OUTPUT=${2:-"./eval_results"}
    LOG_FILE="${LOG_DIR}/eval_${TIMESTAMP}.log"

    echo "============================================="
    echo "Evaluating: ${MODEL_PATH}"
    echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
    echo "Log:  ${LOG_FILE}"
    echo "============================================="

    nohup python -u eval_checkpoint.py \
        --model_path ${MODEL_PATH} \
        --output_dir ${EVAL_OUTPUT} \
        --max_new_tokens 2048 \
        --max_samples 0 \
        --use_flash_attn \
        --wandb_project ${WANDB_PROJECT} \
        > "${LOG_FILE}" 2>&1 &

    PID=$!
    echo "${PID}" > "${LOG_DIR}/eval.pid"
    echo ""
    echo "Started eval in background (PID: ${PID})"
    echo ""
    echo "  Monitor:  tail -f ${LOG_FILE}"
    echo "  Kill:     kill ${PID}"
}

# ============================================================
# Parse command
# ============================================================
case "${1:-help}" in
    setup)
        run_setup
        ;;
    standalone)
        run_standalone
        ;;
    verl)
        run_verl
        ;;
    eval)
        run_eval "$2" "$3"
        ;;
    *)
        echo "Usage: bash run.sh {setup|standalone|verl|eval}"
        echo ""
        echo "  setup       - Create .venv and install dependencies"
        echo "  standalone  - GRPO + SFT training (nohup, 2x A100)"
        echo "  verl        - VERL framework training (nohup)"
        echo "  eval [path] - Evaluate checkpoint (nohup)"
        exit 1
        ;;
esac
