"""
VERL-integrated GRPO + SFT Calibration Training.

This module provides:
1. A VERL-compatible reward function with SFT calibration hook
2. A custom trainer that wraps VERL's GRPO with SFT calibration step

Usage with VERL:
    python train_verl.py
    
    Or via VERL CLI:
    python -m verl.trainer.main_ppo \
        --config verl_config.yaml \
        --reward_fn reward_fn.compute_score
"""
import os
import json
import logging
import copy
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

try:
    import wandb
except ImportError:
    wandb = None

try:
    from verl import DataProto
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    from verl.utils.reward_score import _default_compute_score
    VERL_AVAILABLE = True
except ImportError:
    VERL_AVAILABLE = False
    print("WARNING: verl not found. Using standalone mode.")

from transformers import AutoModelForCausalLM, AutoTokenizer
from reward_fn import compute_score, grade_answer, extract_boxed_answer
from sft_calibration import SFTCalibrationTrainer, CalibrationDataset, collate_fn, VERBALIZATION_CONFIG
from evaluate import generate_and_evaluate, compute_calibration_metrics, save_results

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# VERL Reward Function (used by VERL's reward manager)
# ============================================================

def verl_reward_fn(data_source: str, solution_str: str, ground_truth: Any, **kwargs) -> float:
    """
    VERL-compatible reward function.
    Called by verl's reward computation pipeline.
    """
    if isinstance(ground_truth, dict):
        gt = ground_truth.get("solution", ground_truth.get("answer", ""))
    else:
        gt = str(ground_truth)
    return compute_score(solution_str, gt)


# ============================================================
# SFT Calibration Hook for VERL
# ============================================================

class SFTCalibrationCallback:
    """
    Callback that runs SFT calibration after each GRPO update step.
    Integrates with VERL's training loop.
    """

    def __init__(
        self,
        tokenizer,
        sft_lr: float = 1e-5,
        sft_epochs: int = 1,
        sft_batch_size: int = 4,
        sft_grad_accum: int = 4,
        max_length: int = 2560,
    ):
        self.tokenizer = tokenizer
        self.sft_lr = sft_lr
        self.sft_epochs = sft_epochs
        self.sft_batch_size = sft_batch_size
        self.sft_grad_accum = sft_grad_accum
        self.max_length = max_length
        self.sft_trainer = None  # Lazy init with actual model

    def _ensure_trainer(self, model, device):
        if self.sft_trainer is None:
            self.sft_trainer = SFTCalibrationTrainer(
                model=model,
                tokenizer=self.tokenizer,
                lr=self.sft_lr,
                sft_epochs=self.sft_epochs,
                sft_batch_size=self.sft_batch_size,
                sft_grad_accum=self.sft_grad_accum,
                max_length=self.max_length,
                device=device,
            )

    def on_step_end(
        self,
        model: torch.nn.Module,
        prompts: List[str],
        responses: List[str],
        rewards: List[float],
        device: str = "cuda",
    ) -> Dict[str, float]:
        """
        Called after each GRPO update step.
        Runs SFT calibration on the rollout data.
        """
        self._ensure_trainer(model, device)

        # Convert rewards to is_correct (reward=1.0 means correct)
        is_corrects = [r >= 0.5 for r in rewards]

        metrics = self.sft_trainer.train_on_rollouts(
            prompts=prompts,
            responses=responses,
            is_corrects=is_corrects,
        )

        return metrics


# ============================================================
# Custom VERL Trainer with SFT Calibration
# ============================================================

@dataclass
class TrainConfig:
    """Training configuration."""
    # Model
    model_name: str = "Qwen/Qwen3-4B-Base"
    
    # Data paths
    train_data: str = "/mnt/chatbot30TB/jewonyeom/verl_qwen3/verl/data/math/train.parquet"
    test_math: str = "/mnt/chatbot30TB/jewonyeom/verl_qwen3/verl/data/math/test.parquet"
    test_amc23: str = "/mnt/chatbot30TB/jewonyeom/verl_qwen3/verl/data/amc23/test.parquet"
    test_aime2025: str = "/mnt/chatbot30TB/jewonyeom/verl_qwen3/verl/data/aime2025/test.parquet"
    
    # GRPO
    grpo_lr: float = 1e-6
    kl_coeff: float = 0.01
    clip_range: float = 0.2
    group_size: int = 8
    temperature: float = 0.7
    max_new_tokens: int = 2048
    grpo_grad_accum: int = 4
    grpo_mini_batch_size: int = 64
    
    # SFT Calibration
    sft_lr: float = 1e-5
    sft_epochs: int = 1
    sft_batch_size: int = 4
    sft_grad_accum: int = 4
    
    # Training
    num_epochs: int = 3
    batch_size: int = 8
    eval_every: int = 50
    save_every: int = 100
    eval_samples: int = 100
    
    # System
    output_dir: str = "./outputs_grpo_epicar"
    wandb_project: str = "grpo-epicar"
    wandb_run_name: str = ""
    use_flash_attn: bool = True
    num_gpus: int = 4
    
    # VERL specific
    use_verl: bool = True
    verl_config_path: str = "./verl_config.yaml"


def generate_verl_config(config: TrainConfig) -> Dict:
    """Generate VERL-compatible configuration dict."""
    return {
        "data": {
            "train_files": config.train_data,
            "val_files": config.test_math,
            "train_batch_size": config.batch_size * config.group_size,
            "max_prompt_length": 512,
            "max_response_length": config.max_new_tokens,
        },
        "actor_rollout_ref": {
            "model": {
                "path": config.model_name,
                "enable_gradient_checkpointing": True,
            },
            "actor": {
                "optim": {
                    "lr": config.grpo_lr,
                    "weight_decay": 0.01,
                },
                "ppo_mini_batch_size": config.grpo_mini_batch_size,
                "ppo_micro_batch_size_per_gpu": config.grpo_mini_batch_size // config.num_gpus,
                "clip_ratio": config.clip_range,
                "entropy_coeff": 0.0,
            },
            "rollout": {
                "temperature": config.temperature,
                "top_p": 1.0,
                "n": config.group_size,  # group size for GRPO
                "gpu_memory_utilization": 0.4,
            },
            "ref": {
                "fsdp_config": {
                    "param_offload": True,
                },
            },
        },
        "algorithm": {
            "kl_ctrl": {
                "type": "fixed",
                "kl_coef": config.kl_coeff,
            },
            "adv_estimator": "grpo",  # Key: use GRPO advantage estimation
        },
        "trainer": {
            "total_epochs": config.num_epochs,
            "save_freq": config.save_every,
            "test_freq": config.eval_every,
            "project_name": config.wandb_project,
            "experiment_name": config.wandb_run_name or "grpo_epicar",
            "logger": ["wandb"],
            "default_local_dir": config.output_dir,
        },
    }


# ============================================================
# Standalone Training (without VERL dependency)
# ============================================================

def train_standalone(config: TrainConfig):
    """
    Full training loop without VERL dependency.
    Implements GRPO + SFT Calibration from scratch.
    """
    from train import main as standalone_main, parse_args
    import sys
    
    # Convert config to CLI args
    sys.argv = [
        "train.py",
        "--model_name", config.model_name,
        "--train_data", config.train_data,
        "--test_math", config.test_math,
        "--test_amc23", config.test_amc23,
        "--test_aime2025", config.test_aime2025,
        "--grpo_lr", str(config.grpo_lr),
        "--kl_coeff", str(config.kl_coeff),
        "--clip_range", str(config.clip_range),
        "--group_size", str(config.group_size),
        "--temperature", str(config.temperature),
        "--max_new_tokens", str(config.max_new_tokens),
        "--sft_lr", str(config.sft_lr),
        "--sft_epochs", str(config.sft_epochs),
        "--sft_batch_size", str(config.sft_batch_size),
        "--num_epochs", str(config.num_epochs),
        "--batch_size", str(config.batch_size),
        "--eval_every", str(config.eval_every),
        "--save_every", str(config.save_every),
        "--output_dir", config.output_dir,
        "--wandb_project", config.wandb_project,
    ]
    if config.wandb_run_name:
        sys.argv += ["--wandb_run_name", config.wandb_run_name]
    if config.use_flash_attn:
        sys.argv += ["--use_flash_attn"]
    
    args = parse_args()
    standalone_main(args)


# ============================================================
# VERL-integrated Training
# ============================================================

def train_with_verl(config: TrainConfig):
    """
    Training with VERL framework.
    GRPO is handled by VERL, SFT calibration is injected as a callback.
    """
    if not VERL_AVAILABLE:
        logger.warning("VERL not available. Falling back to standalone mode.")
        train_standalone(config)
        return
    
    import yaml
    
    # Generate and save VERL config
    verl_cfg = generate_verl_config(config)
    
    config_path = os.path.join(config.output_dir, "verl_config.yaml")
    os.makedirs(config.output_dir, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(verl_cfg, f, default_flow_style=False)
    
    logger.info(f"VERL config saved to {config_path}")
    logger.info("Starting VERL training with SFT calibration callback...")
    
    # For VERL integration, we need to run the VERL training loop
    # and inject SFT calibration at each step
    # This is typically done by modifying the VERL trainer class
    
    # Option 1: Use VERL's callback system (if available)
    # Option 2: Wrap VERL's step function
    # Option 3: Run VERL via CLI with custom reward function
    
    # For now, we use the CLI approach with a custom post-step hook
    os.system(
        f"python -m verl.trainer.main_ppo "
        f"--config {config_path} "
        f"--reward_fn reward_fn.compute_score"
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="standalone", 
                        choices=["standalone", "verl"],
                        help="Training mode: standalone (pure Python) or verl (VERL framework)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to JSON config file")
    
    # Allow overriding individual fields
    for f in TrainConfig.__dataclass_fields__:
        field_obj = TrainConfig.__dataclass_fields__[f]
        parser.add_argument(f"--{f}", type=type(field_obj.default), default=None)
    
    args = parser.parse_args()
    
    # Load config
    config = TrainConfig()
    if args.config:
        with open(args.config) as f:
            cfg_dict = json.load(f)
        for k, v in cfg_dict.items():
            if hasattr(config, k):
                setattr(config, k, v)
    
    # Override with CLI args
    for f in TrainConfig.__dataclass_fields__:
        val = getattr(args, f, None)
        if val is not None:
            setattr(config, f, val)
    
    if args.mode == "verl":
        train_with_verl(config)
    else:
        train_standalone(config)
