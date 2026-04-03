# EpiCaRPO: GRPO + Verbalized Confidence Calibration

GRPO (Group Relative Policy Optimization) 강화학습과 SFT 기반 Verbalized Confidence Calibration을 결합한 수학 추론 모델 학습 프레임워크.

## Overview

매 학습 step마다 세 단계를 순차 실행:

1. **GRPO Rollout** — 문제당 G개 응답 생성, `\boxed{}` 추출 후 정답 비교로 reward 계산
2. **GRPO Policy Update** — Group-normalized advantage + clipped ratio + KL penalty
3. **SFT Calibration** — 동일 rollout 데이터에 `"Is the answer correct? A) Yes B) No"` 붙여서, 맞힌 문제→A / 틀린 문제→B 학습

평가 시 verbalized confidence `P(A)/(P(A)+P(B))`로 calibration 지표(ECE, AUROC, Brier, F1) 산출.

## Requirements

- **Python**: 3.10+
- **CUDA**: 12.x
- **Package Manager**: `uv`

```bash
# 환경 세팅
bash run.sh setup
```

또는 수동 설치:

```bash
export PATH="$HOME/.local/bin:$PATH"
uv venv .venv
source .venv/bin/activate
uv pip install torch setuptools wheel
uv pip install -r requirements.txt
```

## Data Format

VERL parquet 포맷을 사용합니다.

| 컬럼 | 내용 |
|------|------|
| `prompt` | `[{'role': 'user', 'content': '문제 텍스트'}]` |
| `reward_model` | `{'ground_truth': '정답', 'style': 'rule'}` |

코드 내부에서 `prompt`의 user message를 문제로, `reward_model.ground_truth`를 정답으로 추출합니다.

## Usage

세 가지 학습 모드를 지원합니다.

### 1. Standalone (원본)

vLLM을 매 스텝 생성/파괴하며 학습. 단일 노드 2 GPU에서 동작.

```bash
bash run.sh standalone
tail -f logs/train_standalone_*.log
```

### 2. Fast (optimizer state 보존)

Standalone과 동일한 구조이나, AdamW optimizer state(momentum/variance)를 스텝 간 보존하여 학습 품질 개선.

```bash
bash run.sh fast
tail -f logs/train_fast_*.log
```

### 3. VERL 8-GPU (권장)

VERL 프레임워크 기반 분산 학습. 모델이 GPU에 상주하며 phase만 전환하므로 로드/언로드 오버헤드 없음.

```bash
# Slurm 제출 (파티션명 수정 필요)
sbatch slurm_verl_8gpu.sh

# 또는 직접 실행
python train_verl_8gpu.py --config-path . --config-name verl_config_8gpu
```

| 모드 | GPU | 예상 속도 | 비고 |
|------|-----|-----------|------|
| standalone | 2× A6000/A100 | ~3 min/step | 매 스텝 vLLM 재생성 |
| fast | 2× A6000/A100 | ~3 min/step | optimizer state 보존 |
| VERL 8-GPU | 8× B200 | ~20-25s/step | FSDP + vLLM colocated |

### 체크포인트 평가

```bash
bash run.sh eval ./outputs_grpo_epicar/run_XXXXXX/final_model ./eval_results
```

## Project Structure

```
EpiCaRPO/
├── run.sh                   # 진입점 (setup / standalone / fast / verl / eval)
│
│  ── Standalone / Fast ──
├── train.py                 # Standalone 학습 루프 (vLLM in-process)
├── train_fast.py            # Fast 모드 (optimizer state 보존)
├── rollout_worker.py        # 격리된 vLLM rollout subprocess
├── slurm_epicarpo.sh        # Slurm 제출 (2 GPU)
│
│  ── VERL 8-GPU ──
├── train_verl_8gpu.py       # VERL 기반 trainer + SFT calibration
├── verl_config_8gpu.yaml    # 8-GPU VERL config (FSDP + vLLM hybrid)
├── slurm_verl_8gpu.sh       # Slurm 제출 (8 GPU)
│
│  ── Core ──
├── sft_calibration.py       # Verbalized confidence SFT 트레이너
├── reward_fn.py             # \boxed{} 추출 + sympy 채점
├── evaluate.py              # 생성 + 채점 + calibration 지표
├── eval_checkpoint.py       # 학습 후 전체 평가 + reliability diagram
│
│  ── Legacy ──
├── verl_config.yaml         # VERL config (2 GPU, 기존)
├── train_verl.py            # VERL 연동 wrapper (기존)
├── train_verl_native.py     # VERL worker 확장 (기존)
│
├── requirements.txt
└── README.md
```

## Hyperparameters

| 항목 | 값 |
|------|-----|
| GRPO lr | 1e-6 |
| GRPO KL coeff | 0.01 |
| GRPO clip range | 0.2 |
| Group size | 8 |
| Sampling temperature | 0.7 |
| SFT lr | 1e-5 |
| SFT epochs/step | 1 |
| SFT batch size | 2 |
| SFT max length | 1024 |
| Training epochs | 3 |
| Batch size (problems/step) | 8 (standalone), 256 (VERL) |
| Max new tokens | 2048 |
| Eval frequency | every 50 steps |
| Checkpoint frequency | every 100 steps |
