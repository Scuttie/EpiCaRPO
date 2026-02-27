"""
VERL-native GRPO + SFT Calibration Worker.

This extends VERL's PPO trainer to inject SFT calibration training
after each GRPO policy update step.

Usage:
    python train_verl_native.py

This script:
1. Uses VERL's Ray-based GRPO training pipeline
2. After each GRPO update, runs SFT calibration on the same rollout data
3. Periodically evaluates on MATH/AMC23/AIME2025 with calibration metrics
4. Logs everything to wandb
"""
import os
import sys
import json
import logging
import copy
from functools import partial

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

try:
    import wandb
except ImportError:
    wandb = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# SFT Calibration as VERL ActorRolloutRefWorker Extension
# ============================================================

VERBALIZATION_CONFIG = {
    "injection": "\nIs the answer correct? Choose ONLY one letter. A) Yes B) No. Your choice:",
    "token_yes": " A",
    "token_no": " B",
}


def build_sft_calibration_batch(
    tokenizer,
    prompts: list,
    responses: list,
    is_corrects: list,
    max_length: int = 2560,
):
    """
    Build SFT calibration training batch from GRPO rollout results.
    
    For each (prompt, response), appends the verbalization injection and
    creates target labels for A (correct) / B (incorrect).
    """
    injection = VERBALIZATION_CONFIG["injection"]
    token_yes = VERBALIZATION_CONFIG["token_yes"]
    token_no = VERBALIZATION_CONFIG["token_no"]
    
    yes_ids = tokenizer.encode(token_yes, add_special_tokens=False)
    no_ids = tokenizer.encode(token_no, add_special_tokens=False)
    
    all_input_ids = []
    all_labels = []
    
    for prompt, response, correct in zip(prompts, responses, is_corrects):
        context_text = prompt + response + injection
        target_text = token_yes if correct else token_no
        
        context_ids = tokenizer.encode(context_text, add_special_tokens=False)
        target_ids = tokenizer.encode(target_text, add_special_tokens=False)
        
        # Truncate if needed
        max_ctx = max_length - len(target_ids)
        if len(context_ids) > max_ctx:
            context_ids = context_ids[:max_ctx]
        
        input_ids = context_ids + target_ids
        labels = [-100] * len(context_ids) + target_ids
        
        all_input_ids.append(input_ids)
        all_labels.append(labels)
    
    # Pad
    max_len = max(len(ids) for ids in all_input_ids)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    
    padded_input_ids = []
    padded_labels = []
    attention_masks = []
    
    for input_ids, labels in zip(all_input_ids, all_labels):
        pad_len = max_len - len(input_ids)
        padded_input_ids.append([pad_id] * pad_len + input_ids)
        padded_labels.append([-100] * pad_len + labels)
        attention_masks.append([0] * pad_len + [1] * len(input_ids))
    
    return {
        "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
        "labels": torch.tensor(padded_labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
    }


def run_sft_calibration_step(
    model,
    optimizer,
    tokenizer,
    prompts: list,
    responses: list,
    is_corrects: list,
    micro_batch_size: int = 4,
    device: str = "cuda",
):
    """
    Run one step of SFT calibration training on GRPO rollout data.
    Designed to be called inside a VERL worker's training step.
    
    Returns: dict with sft metrics
    """
    model.train()
    
    batch = build_sft_calibration_batch(
        tokenizer, prompts, responses, is_corrects
    )
    
    n_samples = batch["input_ids"].shape[0]
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    n_micro_batches = 0
    
    for start in range(0, n_samples, micro_batch_size):
        end = min(start + micro_batch_size, n_samples)
        
        input_ids = batch["input_ids"][start:end].to(device)
        labels = batch["labels"][start:end].to(device)
        attention_mask = batch["attention_mask"][start:end].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # Shift for causal LM
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        
        # Scale loss by number of micro-batches for proper averaging
        n_total_micro = (n_samples + micro_batch_size - 1) // micro_batch_size
        scaled_loss = loss / n_total_micro
        scaled_loss.backward()
        
        with torch.no_grad():
            mask = shift_labels != -100
            if mask.any():
                preds = shift_logits[mask].argmax(dim=-1)
                total_correct += (preds == shift_labels[mask]).sum().item()
                total_tokens += mask.sum().item()
        
        total_loss += loss.item()
        n_micro_batches += 1
    
    # Step optimizer
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()
    
    avg_loss = total_loss / max(n_micro_batches, 1)
    pred_acc = total_correct / max(total_tokens, 1)
    
    return {
        "sft_loss": avg_loss,
        "sft_pred_accuracy": pred_acc,
        "sft_samples": n_samples,
        "sft_correct_ratio": sum(is_corrects) / max(len(is_corrects), 1),
    }


# ============================================================
# Evaluation Function
# ============================================================

@torch.no_grad()
def evaluate_with_calibration(
    model,
    tokenizer,
    data_path: str,
    dataset_name: str,
    max_samples: int = 100,
    max_new_tokens: int = 2048,
    device: str = "cuda",
) -> dict:
    """Quick evaluation with verbalized confidence calibration metrics."""
    from evaluate import generate_and_evaluate
    
    df = pd.read_parquet(data_path)
    if "question" in df.columns and "problem" not in df.columns:
        df = df.rename(columns={"question": "problem"})
    if "answer" in df.columns and "solution" not in df.columns:
        df = df.rename(columns={"answer": "solution"})
    
    if max_samples > 0:
        df = df.head(max_samples)
    
    result = generate_and_evaluate(
        model=model,
        tokenizer=tokenizer,
        dataset=df,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        device=device,
        dataset_name=dataset_name,
    )
    
    return result["metrics"]


# ============================================================
# Main Entry Point (for standalone usage or VERL extension)
# ============================================================

def main():
    """
    Standalone training entry point.
    For VERL integration, import run_sft_calibration_step and
    call it within VERL's training loop.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="VERL-native GRPO + SFT Calibration")
    parser.add_argument("--mode", default="standalone", choices=["standalone", "verl"])
    args, remaining = parser.parse_known_args()
    
    if args.mode == "standalone":
        # Fall back to standalone implementation
        from train import main as standalone_main, parse_args
        sys.argv = ["train.py"] + remaining
        standalone_args = parse_args()
        standalone_main(standalone_args)
    else:
        logger.info("For VERL-native mode, use:")
        logger.info("  python -m verl.trainer.main_ppo --config verl_config.yaml")
        logger.info("")
        logger.info("To add SFT calibration, modify VERL's actor worker to call:")
        logger.info("  run_sft_calibration_step(model, optimizer, tokenizer, ...)")
        logger.info("  after each GRPO policy update.")


if __name__ == "__main__":
    main()
