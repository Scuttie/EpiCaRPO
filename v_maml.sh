#!/bin/bash
set -x

# 환경 변수 설정
export RAY_DEDUP_LOGS=0
export PYTHONUNBUFFERED=1
BASE_DIR="/mnt/chatbot30TB/jewonyeom/verl_qwen3"
DATA_DIR="${BASE_DIR}/verl/data"
# 데이터 경로 설정
math_train_path="${DATA_DIR}/math/train.parquet"
math_test_path="${DATA_DIR}/math/test.parquet"
aime2025_test_path="${DATA_DIR}/aime2025/test.parquet"
amc23_test_path="${DATA_DIR}/amc23/test.parquet"

# 실험 설정
model_name="Qwen/Qwen3-4B-Instruct-2507"
N_GPU=2


echo "# Starting V-MAML Training (Verbalized Meta-Learning)"
echo "# Model: $model_name | GPUs: $N_GPU"

python3 v_maml_trainer.py \
    --model "$model_name" \
    --train_data "$math_train_path" \
    --test_data "$math_test_path" "$aime2025_test_path" "$amc23_test_path" \
    --n_gpu $N_GPU