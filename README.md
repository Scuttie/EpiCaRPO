# EpiCaRO: GRPO + Verbalized Confidence Calibration

GRPO (Group Relative Policy Optimization) 강화학습과 SFT 기반 Verbalized Confidence Calibration을 결합한 수학 추론 모델 학습 프레임워크.

## Overview

매 학습 step마다 세 단계를 순차 실행:

1. **GRPO Rollout** — 문제당 G개 응답 생성, `\boxed{}` 추출 후 정답 비교로 reward 계산
2. **GRPO Policy Update** — Group-normalized advantage + clipped ratio + KL penalty
3. **SFT Calibration** — 동일 rollout 데이터에 `"Is the answer correct? A) Yes B) No"` 붙여서, 맞힌 문제→A / 틀린 문제→B 학습

평가 시 verbalized confidence `P(A)/(P(A)+P(B))`로 calibration 지표(ECE, AUROC, Brier, F1) 산출.

## Requirements

- **GPU**: A100 80GB × 2 (model `cuda:0`, ref_model `cuda:1`)
- **Python**: 3.10+
- **CUDA**: 12.x

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

### 학습 (nohup 백그라운드)

```bash
bash run.sh standalone
```

로그 모니터링:

```bash
tail -f logs/train_standalone_*.log
```

### 체크포인트 평가

```bash
bash run.sh eval ./outputs_grpo_epicar/run_XXXXXX/final_model ./eval_results
```

### VERL 프레임워크 연동

```bash
bash run.sh verl
```

## Project Structure

```
EpiCaRO/
├── run.sh                  # 진입점 (setup / standalone / verl / eval)
├── train.py                # 메인 학습 루프
│   ├── load_parquet_dataset()   # VERL 포맷 파싱
│   ├── load_model()             # SDPA auto-fallback
│   ├── GRPOTrainer              # rollout + policy update
│   └── main()                   # 오케스트레이션
├── sft_calibration.py      # Verbalized confidence SFT 트레이너
├── reward_fn.py            # \boxed{} 추출 + sympy 채점
├── evaluate.py             # 생성 + 채점 + calibration 지표
├── eval_checkpoint.py      # 학습 후 전체 평가 + reliability diagram
├── verl_config.yaml        # VERL 프레임워크 config
├── train_verl.py           # VERL 연동 wrapper
├── train_verl_native.py    # VERL worker 확장
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
| Batch size (problems/step) | 8 |
| Max new tokens | 2048 |
| Eval frequency | every 50 steps |
| Checkpoint frequency | every 100 steps |
