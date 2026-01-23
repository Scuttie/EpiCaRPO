#!/bin/bash
set -e
MODE=off

echo "========================================================"
echo "Running Ablation Study: Verification Task = $MODE"
echo "========================================================"

# 1. Environment Variables
export RAY_DEDUP_LOGS=0
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1
export UVLOOP_DISABLE=1
export RAY_USE_UVLOOP=0

# 2. Paths & Configs
BASE_DIR="/mnt/chatbot30TB/jewonyeom/verl_qwen3"
DATA_DIR="${BASE_DIR}/verl/data"
MATH_TRAIN="${DATA_DIR}/math/train.parquet"

# ★★★ 여러 validation 파일을 리스트로 지정 ★★★
MATH_TEST="${DATA_DIR}/math/test.parquet"
AMC23_TEST="${DATA_DIR}/amc23/test.parquet"
AIME2025_TEST="${DATA_DIR}/aime2025/test.parquet"

MODEL_NAME="Qwen/Qwen3-4B-Base"
N_GPU=2

if [ "$MODE" == "on" ]; then
    EXP_NAME="qwen3_grpo_with_verification"
    USE_VERIF="True"
else
    EXP_NAME="qwen3_grpo_baseline"
    USE_VERIF="False"
fi

# 3. Run Training
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$MATH_TRAIN" \
    "data.val_files=[${MATH_TEST},${AMC23_TEST},${AIME2025_TEST}]" \
    data.train_batch_size=128 \
    data.val_batch_size=64 \
    data.max_prompt_length=2048 \
    data.max_response_length=4096 \
    actor_rollout_ref.model.path="$MODEL_NAME" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.30 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    ++actor_rollout_ref.actor.fsdp_config.param_offload=True \
    ++actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    ++actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    trainer.critic_warmup=0 \
    ++trainer.wandb.entity="jewon0908-seoul-national-university-org" \
    ++trainer.compute_calibration_metrics=True \
    ++actor_rollout_ref.rollout.agent.enable=True \
    "trainer.logger=[console,wandb]" \
    trainer.project_name=verl_grpo_ablation \
    trainer.resume_mode=disable \
    trainer.experiment_name="$EXP_NAME" \
    trainer.default_local_dir="${BASE_DIR}/checkpoints/${EXP_NAME}" \
    trainer.n_gpus_per_node=$N_GPU \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.save_freq=10 \
    trainer.test_freq=1 \
    trainer.total_epochs=10 \
    ++trainer.use_verification_task=$USE_VERIF \
    2>&1 | tee "${EXP_NAME}.log"

echo "Training finished. Log saved to ${EXP_NAME}.log"
