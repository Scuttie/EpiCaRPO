"""
Fast GRPO + SFT Calibration Training

Improvement over train.py: preserves optimizer state (momentum/variance)
across steps instead of resetting each step.

Same GPU layout and vLLM approach as original train.py:
  GPU 0: vLLM (rollout) → train_model (GRPO/SFT), time-multiplexed
  GPU 1: ref_model (permanent)
"""
import os
import gc
import json
import logging
import argparse
import time
from datetime import datetime
from typing import List, Dict

import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import wandb
except ImportError:
    wandb = None

from train import GRPOUpdater, load_parquet_dataset, prepare_prompt, vllm_rollout
from sft_calibration import SFTCalibrationTrainer
from evaluate import generate_and_evaluate, save_results

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d: %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# Optimizer State Save/Restore (to CPU RAM)
# ============================================================

def save_optimizer_state(optimizer) -> dict:
    """Save optimizer state dict to CPU RAM."""
    if len(optimizer.state) == 0:
        return None
    state = optimizer.state_dict()
    for s in state["state"].values():
        for k, v in s.items():
            if isinstance(v, torch.Tensor):
                s[k] = v.cpu()
    return state


def restore_optimizer_state(optimizer, saved_state, device):
    """Restore optimizer state from CPU RAM to GPU."""
    if saved_state is None:
        return
    optimizer.load_state_dict(saved_state)
    for s in optimizer.state.values():
        for k, v in s.items():
            if isinstance(v, torch.Tensor):
                s[k] = v.to(device)


# ============================================================
# Main Training Loop
# ============================================================

def main(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    weights_dir = os.path.join(output_dir, "_latest_weights")
    os.makedirs(weights_dir, exist_ok=True)

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    if wandb is not None and args.wandb_project:
        os.environ["WANDB__SERVICE_WAIT"] = "300"
        os.environ["WANDB_START_METHOD"] = "thread"
        wandb.init(project=args.wandb_project,
                   name=args.wandb_run_name or f"grpo_epicar_fast_{timestamp}",
                   config=vars(args))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_model = "cuda:0"
    device_ref = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    logger.info(f"Devices: train={device_model}, ref={device_ref}, GPUs={torch.cuda.device_count()}")

    attn_impl = "sdpa"
    try:
        import flash_attn; attn_impl = "flash_attention_2"
    except ImportError:
        pass

    # ---- Ref model (permanent on GPU 1) ----
    logger.info(f"Loading ref model → {device_ref} (permanent, frozen)")
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation=attn_impl,
    ).to(device_ref)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    current_model_path = args.model_name

    # ---- Data ----
    train_df = load_parquet_dataset(args.train_data)
    test_datasets = {}
    for name, path in [("MATH", args.test_math), ("AMC23", args.test_amc23),
                       ("AIME2025", args.test_aime2025)]:
        if os.path.exists(path):
            test_datasets[name] = load_parquet_dataset(path)
        else:
            logger.warning(f"Test data not found: {path}, skipping {name}")

    problems = train_df["problem"].tolist()
    solutions = train_df["solution"].tolist()
    n_steps_per_epoch = (len(problems) + args.batch_size - 1) // args.batch_size

    logger.info(f"Training: {len(problems)} problems, {n_steps_per_epoch} steps/epoch, "
                f"{args.num_epochs} epochs = {n_steps_per_epoch * args.num_epochs} total steps")

    # ---- Training Loop ----
    global_step = 0
    train_model = None
    grpo = None
    sft = None
    saved_opt_state = None  # Optimizer state preserved in CPU RAM

    for epoch in range(args.num_epochs):
        logger.info(f"{'='*60}")
        logger.info(f"Epoch {epoch + 1}/{args.num_epochs}")
        logger.info(f"{'='*60}")

        indices = np.random.permutation(len(problems))

        for batch_start in range(0, len(problems), args.batch_size):
            step_start = time.time()

            batch_end = min(batch_start + args.batch_size, len(problems))
            batch_idx = indices[batch_start:batch_end]
            batch_problems = [problems[i] for i in batch_idx]
            batch_solutions = [solutions[i] for i in batch_idx]
            global_step += 1

            logger.info(f"Step {global_step}: {len(batch_problems)} problems × {args.group_size}")

            # ======== Phase 1: Save optimizer state & free GPU 0 for vLLM ========
            t0 = time.time()
            if train_model is not None:
                saved_opt_state = save_optimizer_state(grpo.optimizer)
                del train_model, grpo, sft
                train_model, grpo, sft = None, None, None
                gc.collect()
                torch.cuda.empty_cache()
            t_cleanup = time.time() - t0

            # ======== Phase 2: vLLM Rollout on GPU 0 (same as original) ========
            prompts = [prepare_prompt(p) for p in batch_problems]

            t0 = time.time()
            rollout = vllm_rollout(
                model_path=current_model_path,
                prompts=prompts,
                solutions=batch_solutions,
                group_size=args.group_size,
                temperature=args.temperature,
                max_tokens=args.max_new_tokens,
                gpu_util=args.vllm_gpu_util,
            )
            t_rollout = time.time() - t0

            # ======== Phase 3: Reload model + restore optimizer ========
            t0 = time.time()
            train_model = AutoModelForCausalLM.from_pretrained(
                current_model_path, dtype=torch.bfloat16, trust_remote_code=True,
                attn_implementation=attn_impl,
            ).to(device_model)
            train_model.gradient_checkpointing_enable()

            grpo = GRPOUpdater(
                model=train_model, ref_model=ref_model, tokenizer=tokenizer,
                lr=args.grpo_lr, kl_coeff=args.kl_coeff, clip_range=args.clip_range,
                grad_accum=args.grpo_grad_accum, device=device_model, device_ref=device_ref,
            )

            sft = SFTCalibrationTrainer(
                model=train_model, tokenizer=tokenizer, optimizer=grpo.optimizer,
                lr=args.sft_lr, sft_epochs=args.sft_epochs,
                sft_batch_size=args.sft_batch_size, sft_grad_accum=args.sft_grad_accum,
                max_length=1024, device=device_model,
            )

            # KEY IMPROVEMENT: restore optimizer momentum/variance from previous step
            restore_optimizer_state(grpo.optimizer, saved_opt_state, device_model)
            t_reload = time.time() - t0

            # ======== Phase 4: GRPO Update ========
            t0 = time.time()
            grpo_metrics = grpo.update(
                rollout["prompts"], rollout["responses"],
                rollout["rewards"], args.group_size,
            )
            t_grpo = time.time() - t0
            logger.info(f"GRPO: loss={grpo_metrics['grpo_loss']:.4f}, "
                        f"reward={grpo_metrics['grpo_mean_reward']:.4f} ({t_grpo:.1f}s)")
            torch.cuda.empty_cache()

            # ======== Phase 5: SFT Calibration ========
            t0 = time.time()
            sft_metrics = sft.train_on_rollouts(
                prompts=rollout["prompts"],
                responses=rollout["responses"],
                is_corrects=rollout["is_corrects"],
            )
            t_sft = time.time() - t0
            logger.info(f"SFT: loss={sft_metrics['sft_loss']:.4f}, "
                        f"acc={sft_metrics['sft_pred_accuracy']:.4f} ({t_sft:.1f}s)")

            # ======== Phase 6: Save Weights ========
            train_model.save_pretrained(weights_dir)
            tokenizer.save_pretrained(weights_dir)
            current_model_path = weights_dir

            # Step summary
            step_time = time.time() - step_start
            logger.info(f"Step {global_step}: {step_time:.1f}s total "
                        f"(cleanup={t_cleanup:.1f}s rollout={t_rollout:.1f}s "
                        f"reload={t_reload:.1f}s grpo={t_grpo:.1f}s sft={t_sft:.1f}s)")

            # Logging
            log_dict = {
                **grpo_metrics, **sft_metrics,
                "epoch": epoch, "step": global_step,
                "timing/step_total": step_time,
                "timing/rollout": t_rollout,
                "timing/grpo": t_grpo,
                "timing/sft": t_sft,
            }
            if wandb is not None and args.wandb_project:
                try:
                    if wandb.run is None:
                        wandb.init(project=args.wandb_project,
                                   name=args.wandb_run_name or f"grpo_epicar_fast_{timestamp}",
                                   config=vars(args), resume="allow")
                    wandb.log(log_dict, step=global_step)
                except Exception as e:
                    logger.warning(f"wandb log failed: {e}")

            # ======== Periodic Evaluation ========
            if global_step % args.eval_every == 0 and test_datasets:
                logger.info(f"Evaluation at step {global_step}...")
                train_model.eval()
                for ds_name, ds_df in test_datasets.items():
                    eval_df = ds_df.head(args.eval_samples) if args.eval_samples > 0 else ds_df
                    eval_result = generate_and_evaluate(
                        model=train_model, tokenizer=tokenizer,
                        dataset=eval_df, max_new_tokens=args.max_new_tokens,
                        temperature=0.0, device=device_model, dataset_name=ds_name,
                    )
                    m = eval_result["metrics"]
                    logger.info(f"[{ds_name}] acc={m['acc']:.4f}, ece={m['ece']:.4f}, auroc={m['auroc']:.4f}")
                    save_results(eval_result["results"],
                                 os.path.join(output_dir, f"eval_step{global_step}_{ds_name.lower()}.jsonl"))
                    if wandb is not None and wandb.run is not None:
                        try:
                            wandb.log({f"eval/{ds_name}/{k}": v for k, v in m.items()
                                      if isinstance(v, (int, float))}, step=global_step)
                        except Exception:
                            pass
                train_model.train()
                torch.cuda.empty_cache()

            # ======== Periodic Checkpoint ========
            if global_step % args.save_every == 0:
                ckpt = os.path.join(output_dir, f"checkpoint_step{global_step}")
                train_model.save_pretrained(ckpt)
                tokenizer.save_pretrained(ckpt)
                logger.info(f"Checkpoint saved: {ckpt}")

    # ---- Final ----
    if train_model is not None:
        final_dir = os.path.join(output_dir, "final_model")
        train_model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        logger.info(f"Final model: {final_dir}")

        if test_datasets:
            train_model.eval()
            for ds_name, ds_df in test_datasets.items():
                result = generate_and_evaluate(
                    model=train_model, tokenizer=tokenizer, dataset=ds_df,
                    max_new_tokens=args.max_new_tokens, temperature=0.0,
                    device=device_model, dataset_name=ds_name,
                )
                m = result["metrics"]
                logger.info(f"[FINAL {ds_name}] acc={m['acc']:.4f}, ece={m['ece']:.4f}, "
                            f"auroc={m['auroc']:.4f}, brier={m['brier']:.4f}")
                save_results(result["results"], os.path.join(output_dir, f"final_{ds_name.lower()}.jsonl"))
                if wandb is not None and wandb.run is not None:
                    try:
                        wandb.log({f"final/{ds_name}/{k}": v for k, v in m.items()
                                  if isinstance(v, (int, float))}, step=global_step)
                    except Exception:
                        pass

    if wandb is not None:
        try:
            if wandb.run is not None:
                wandb.finish()
        except Exception:
            pass

    logger.info("Training complete!")


def parse_args():
    p = argparse.ArgumentParser(description="Fast GRPO + SFT Training (Optimizer State Preserved)")

    # Data
    p.add_argument("--train_data", default="/home/jewonyeom/verl/data/math/train.parquet")
    p.add_argument("--test_math", default="/home/jewonyeom/verl/data/math/test.parquet")
    p.add_argument("--test_amc23", default="/home/jewonyeom/verl/data/amc23/test.parquet")
    p.add_argument("--test_aime2025", default="/home/jewonyeom/verl/data/aime2025/test.parquet")

    # Model
    p.add_argument("--model_name", default="Qwen/Qwen3-4B-Base")

    # vLLM
    p.add_argument("--vllm_gpu_util", type=float, default=0.85)

    # GRPO
    p.add_argument("--grpo_lr", type=float, default=1e-6)
    p.add_argument("--kl_coeff", type=float, default=0.01)
    p.add_argument("--clip_range", type=float, default=0.2)
    p.add_argument("--group_size", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--grpo_grad_accum", type=int, default=4)

    # SFT
    p.add_argument("--sft_lr", type=float, default=1e-5)
    p.add_argument("--sft_epochs", type=int, default=1)
    p.add_argument("--sft_batch_size", type=int, default=2)
    p.add_argument("--sft_grad_accum", type=int, default=4)

    # Training
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument("--save_every", type=int, default=100)
    p.add_argument("--eval_samples", type=int, default=100)

    # Output
    p.add_argument("--output_dir", default="./outputs_grpo_epicar")
    p.add_argument("--wandb_project", default="grpo-epicar")
    p.add_argument("--wandb_run_name", default=None)

    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
