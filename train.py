"""
Main Training Script: GRPO + SFT Calibration (EpiCaR-style)

Training Loop:
1. GRPO rollout: Generate solutions for math problems, compute rewards
2. GRPO update: Policy gradient update using group relative policy optimization
3. SFT calibration: Train model to predict A/B (correct/incorrect) for its own solutions
4. Periodic evaluation on test sets with calibration metrics

Usage:
    python train.py
"""
import os
import json
import logging
import argparse
from datetime import datetime
from typing import List, Dict, Optional

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import wandb
except ImportError:
    wandb = None

from reward_fn import compute_score, grade_answer, extract_boxed_answer
from sft_calibration import SFTCalibrationTrainer, VERBALIZATION_CONFIG
from evaluate import generate_and_evaluate, save_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# Data Loading
# ============================================================

def load_parquet_dataset(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df)} samples from {path} | columns={df.columns.tolist()}")

    # Handle VERL format: prompt is chat messages list, ground_truth in reward_model
    if "prompt" in df.columns and "reward_model" in df.columns:
        # Extract problem text from chat messages
        def extract_problem(prompt_msgs):
            if isinstance(prompt_msgs, list):
                for msg in prompt_msgs:
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        return msg.get("content", "")
                # fallback: last message content
                if prompt_msgs:
                    last = prompt_msgs[-1]
                    return last.get("content", str(last)) if isinstance(last, dict) else str(last)
            return str(prompt_msgs)

        def extract_ground_truth(rm):
            if isinstance(rm, dict):
                return rm.get("ground_truth", "")
            return str(rm)

        df["problem"] = df["prompt"].apply(extract_problem)
        df["solution"] = df["reward_model"].apply(extract_ground_truth)
        logger.info(f"  Parsed VERL format: problem from prompt messages, solution from reward_model.ground_truth")
        return df

    # Fallback: try standard column names
    col_map = {}
    for candidate in ["problem", "question", "input", "query", "content"]:
        if candidate in df.columns:
            col_map[candidate] = "problem"
            break
    for candidate in ["solution", "answer", "target", "output", "expected_answer", "ground_truth"]:
        if candidate in df.columns:
            col_map[candidate] = "solution"
            break

    if col_map:
        df = df.rename(columns=col_map)

    if "problem" not in df.columns:
        raise KeyError(
            f"Cannot find problem column in {path}. "
            f"Available columns: {df.columns.tolist()}"
        )

    return df


def prepare_prompt(problem: str) -> str:
    return (
        "<|im_start|>system\nYou are a helpful assistant. The user asks a question, "
        "and you solve it.<|im_end|>\n"
        f"<|im_start|>user\n{problem} Put your final answer within \\boxed{{}}.<|im_end|>\n"
        "<|im_start|>assistant\nLet me solve this step by step."
    )


# ============================================================
# Model Loading Helper
# ============================================================

def load_model(model_name_or_path: str, device: str = "cuda"):
    """Load model with best available attention implementation."""
    # Detect best attention implementation
    attn_impl = "eager"
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
        logger.info("Using FlashAttention2")
    except ImportError:
        try:
            # Check if SDPA is available (PyTorch >= 2.0)
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                attn_impl = "sdpa"
                logger.info("flash-attn not installed, using PyTorch SDPA")
            else:
                logger.info("Using eager attention")
        except:
            logger.info("Using eager attention")

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=attn_impl,
    ).to(device)

    return model


# ============================================================
# GRPO Rollout & Update
# ============================================================

class GRPOTrainer:
    """
    Group Relative Policy Optimization trainer.
    """

    def __init__(
        self,
        model,
        ref_model,
        tokenizer,
        optimizer,  # ✅ Optimizer를 외부에서 주입받도록 수정
        kl_coeff: float = 0.01,
        clip_range: float = 0.2,
        group_size: int = 8,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        device: str = "cuda:0",
        device_ref: str = "cuda:1",
        grad_accum_steps: int = 4,
    ):
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.optimizer = optimizer  # ✅ 주입받은 Optimizer 사용 (내부 생성 삭제)
        self.kl_coeff = kl_coeff
        self.clip_range = clip_range
        self.group_size = group_size
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device
        self.device_ref = device_ref
        self.grad_accum_steps = grad_accum_steps

        self.stop_ids = [151643, 151645]  # Qwen3 stop tokens

    @torch.no_grad()
    def rollout(
        self, problems: List[str], solutions: List[str]
    ) -> Dict:
        self.model.eval()
        all_prompts = []
        all_responses = []
        all_rewards = []
        all_is_corrects = []
        all_prompt_ids = []
        all_response_ids = []

        for problem, solution in tqdm(
            zip(problems, solutions), total=len(problems), desc="GRPO Rollout"
        ):
            prompt = prepare_prompt(problem)
            inputs = self.tokenizer(
                prompt, return_tensors="pt", add_special_tokens=False
            )
            input_ids = inputs["input_ids"].to(self.device)
            prompt_len = input_ids.shape[1]

            for _ in range(self.group_size):
                output = self.model.generate(
                    input_ids,
                    do_sample=True,
                    temperature=self.temperature,
                    max_new_tokens=self.max_new_tokens,
                    eos_token_id=self.stop_ids,
                    pad_token_id=self.tokenizer.pad_token_id
                    or self.tokenizer.eos_token_id,
                )

                response_ids = output[0, prompt_len:]
                response_text = self.tokenizer.decode(
                    response_ids, skip_special_tokens=True
                )

                is_correct = grade_answer(response_text, solution)
                reward = 1.0 if is_correct else 0.0

                all_prompts.append(prompt)
                all_responses.append(response_text)
                all_rewards.append(reward)
                all_is_corrects.append(is_correct)
                all_prompt_ids.append(input_ids[0].tolist())
                all_response_ids.append(response_ids.tolist())

        return {
            "prompts": all_prompts,
            "responses": all_responses,
            "rewards": all_rewards,
            "is_corrects": all_is_corrects,
            "prompt_ids": all_prompt_ids,
            "response_ids": all_response_ids,
        }

    def compute_grpo_loss(
        self,
        prompt_ids: List[int],
        response_ids: List[int],
        advantage: float,
    ) -> torch.Tensor:
        full_ids = prompt_ids + response_ids
        input_ids = torch.tensor([full_ids], device=self.device)
        prompt_len = len(prompt_ids)

        outputs = self.model(input_ids=input_ids)
        logits = outputs.logits[:, prompt_len - 1 : -1, :]
        response_tensor = input_ids[:, prompt_len:]

        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(
            2, response_tensor.unsqueeze(-1)
        ).squeeze(-1)

        with torch.no_grad():
            # ref_model may be on a different device
            ref_input_ids = torch.tensor([full_ids], device=self.device_ref)
            ref_outputs = self.ref_model(input_ids=ref_input_ids)
            ref_logits = ref_outputs.logits[:, prompt_len - 1 : -1, :]
            ref_log_probs = torch.log_softmax(ref_logits, dim=-1)
            ref_response_tensor = ref_input_ids[:, prompt_len:]
            ref_token_log_probs = ref_log_probs.gather(
                2, ref_response_tensor.unsqueeze(-1)
            ).squeeze(-1)
            # Move back to model device
            ref_token_log_probs = ref_token_log_probs.to(self.device)

        ratio = torch.exp(token_log_probs - ref_token_log_probs.detach())
        clipped_ratio = torch.clamp(
            ratio, 1 - self.clip_range, 1 + self.clip_range
        )

        pg_loss1 = -advantage * ratio
        pg_loss2 = -advantage * clipped_ratio
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        kl = (token_log_probs - ref_token_log_probs.detach()).mean()

        total_loss = pg_loss + self.kl_coeff * kl

        return total_loss, pg_loss.item(), kl.item()

    def update(self, rollout_data: Dict) -> Dict[str, float]:
        self.model.train()

        rewards = np.array(rollout_data["rewards"])
        n_samples = len(rewards)
        n_problems = n_samples // self.group_size

        advantages = np.zeros(n_samples)
        for i in range(n_problems):
            start = i * self.group_size
            end = start + self.group_size
            group_rewards = rewards[start:end]
            mean_r = np.mean(group_rewards)
            std_r = np.std(group_rewards) + 1e-8
            advantages[start:end] = (group_rewards - mean_r) / std_r

        total_pg_loss = 0.0
        total_kl = 0.0
        total_loss_val = 0.0
        update_steps = 0

        self.optimizer.zero_grad()

        indices = np.random.permutation(n_samples)

        for step_idx, idx in enumerate(indices):
            adv = advantages[idx]
            if abs(adv) < 1e-8:
                continue

            loss, pg_loss, kl = self.compute_grpo_loss(
                rollout_data["prompt_ids"][idx],
                rollout_data["response_ids"][idx],
                adv,
            )

            scaled_loss = loss / self.grad_accum_steps
            scaled_loss.backward()

            total_pg_loss += pg_loss
            total_kl += kl
            total_loss_val += loss.item()
            update_steps += 1

            if (step_idx + 1) % self.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

        if update_steps % self.grad_accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()

        n = max(update_steps, 1)
        return {
            "grpo_loss": total_loss_val / n,
            "grpo_pg_loss": total_pg_loss / n,
            "grpo_kl": total_kl / n,
            "grpo_mean_reward": float(np.mean(rewards)),
            "grpo_std_reward": float(np.std(rewards)),
            "grpo_update_steps": update_steps,
        }


# ============================================================
# Main Training Loop
# ============================================================

def main(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    if wandb is not None and args.wandb_project:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"grpo_epicar_{timestamp}",
            config=vars(args),
        )

    # ---- Load Model ----
    logger.info(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Distribute across 2 GPUs: model on cuda:0, ref_model on cuda:1
    device_model = "cuda:0"
    device_ref = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    logger.info(f"Using devices: model={device_model}, ref={device_ref}")

    model = load_model(args.model_name, device=device_model)
    model.gradient_checkpointing_enable()
    logger.info("Gradient checkpointing enabled")

    # Reference model (frozen copy for KL) on second GPU
    logger.info("Loading reference model (frozen)")
    ref_model = load_model(args.model_name, device=device_ref)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # ---- Load Data ----
    train_df = load_parquet_dataset(args.train_data)
    test_datasets = {
        "MATH": load_parquet_dataset(args.test_math),
        "AMC23": load_parquet_dataset(args.test_amc23),
        "AIME2025": load_parquet_dataset(args.test_aime2025),
    }

    # ✅ 단일 Optimizer 생성 (GRPO와 SFT가 공유, LR 1e-6 통일)
    shared_optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.grpo_lr, weight_decay=0.01
    )
    logger.info(f"Initialized single shared AdamW optimizer with LR: {args.grpo_lr}")

    # ---- Initialize Trainers ----
    grpo_trainer = GRPOTrainer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        optimizer=shared_optimizer,  # ✅ 통합 Optimizer 주입
        kl_coeff=args.kl_coeff,
        clip_range=args.clip_range,
        group_size=args.group_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device=device_model,
        device_ref=device_ref,
        grad_accum_steps=args.grpo_grad_accum,
    )

    sft_trainer = SFTCalibrationTrainer(
        model=model,
        tokenizer=tokenizer,
        optimizer=shared_optimizer,  # ✅ 통합 Optimizer 주입
        sft_epochs=args.sft_epochs,
        sft_batch_size=args.sft_batch_size,
        sft_grad_accum=args.sft_grad_accum,
        max_length=1024,  # Only need recent context for A/B prediction
        device=device_model,
    )

    # ---- Training Loop ----
    global_step = 0
    problems = train_df["problem"].tolist()
    solutions = train_df["solution"].tolist()

    for epoch in range(args.num_epochs):
        logger.info(f"=== Epoch {epoch + 1}/{args.num_epochs} ===")

        indices = np.random.permutation(len(problems))

        for batch_start in range(0, len(problems), args.batch_size):
            batch_end = min(batch_start + args.batch_size, len(problems))
            batch_indices = indices[batch_start:batch_end]

            batch_problems = [problems[i] for i in batch_indices]
            batch_solutions = [solutions[i] for i in batch_indices]

            global_step += 1
            logger.info(
                f"Step {global_step}: Processing batch {batch_start}-{batch_end} "
                f"({len(batch_problems)} problems)"
            )

            # ---- Step 1: GRPO Rollout ----
            rollout_data = grpo_trainer.rollout(batch_problems, batch_solutions)
            torch.cuda.empty_cache()

            # ---- Step 2: GRPO Policy Update ----
            grpo_metrics = grpo_trainer.update(rollout_data)
            logger.info(
                f"GRPO: loss={grpo_metrics['grpo_loss']:.4f}, "
                f"reward={grpo_metrics['grpo_mean_reward']:.4f}"
            )
            torch.cuda.empty_cache()

            # ---- Step 3: SFT Calibration ----
            sft_metrics = sft_trainer.train_on_rollouts(
                prompts=rollout_data["prompts"],
                responses=rollout_data["responses"],
                is_corrects=rollout_data["is_corrects"],
            )
            logger.info(
                f"SFT: loss={sft_metrics['sft_loss']:.4f}, "
                f"pred_acc={sft_metrics['sft_pred_accuracy']:.4f}"
            )
            torch.cuda.empty_cache()

            # ---- Logging ----
            log_dict = {**grpo_metrics, **sft_metrics, "epoch": epoch, "step": global_step}
            if wandb is not None and wandb.run is not None:
                wandb.log(log_dict, step=global_step)

            # ---- Periodic Evaluation ----
            if global_step % args.eval_every == 0:
                logger.info(f"Running evaluation at step {global_step}...")
                model.eval()

                for ds_name, ds_df in test_datasets.items():
                    eval_df = ds_df.head(args.eval_samples) if args.eval_samples > 0 else ds_df

                    eval_result = generate_and_evaluate(
                        model=model,
                        tokenizer=tokenizer,
                        dataset=eval_df,
                        max_new_tokens=args.max_new_tokens,
                        temperature=0.0,
                        device=device_model,
                        dataset_name=ds_name,
                    )

                    metrics = eval_result["metrics"]
                    logger.info(
                        f"[{ds_name}] acc={metrics['acc']:.4f}, "
                        f"ece={metrics['ece']:.4f}, auroc={metrics['auroc']:.4f}, "
                        f"brier={metrics['brier']:.4f}"
                    )

                    save_results(
                        eval_result["results"],
                        os.path.join(
                            output_dir,
                            f"eval_step{global_step}_{ds_name.lower()}.jsonl",
                        ),
                    )

                    if wandb is not None and wandb.run is not None:
                        eval_log = {
                            f"eval/{ds_name}/{k}": v
                            for k, v in metrics.items()
                            if isinstance(v, (int, float))
                        }
                        wandb.log(eval_log, step=global_step)

                model.train()

            # ---- Periodic Checkpoint ----
            if global_step % args.save_every == 0:
                ckpt_dir = os.path.join(output_dir, f"checkpoint_step{global_step}")
                logger.info(f"Saving checkpoint to {ckpt_dir}")
                model.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)

    # ---- Final Save ----
    final_dir = os.path.join(output_dir, "final_model")
    logger.info(f"Saving final model to {final_dir}")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    # ---- Final Full Evaluation ----
    logger.info("Running final evaluation on full test sets...")
    model.eval()
    for ds_name, ds_df in test_datasets.items():
        eval_result = generate_and_evaluate(
            model=model,
            tokenizer=tokenizer,
            dataset=ds_df,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            device=device_model,
            dataset_name=ds_name,
        )
        metrics = eval_result["metrics"]
        logger.info(
            f"[FINAL {ds_name}] acc={metrics['acc']:.4f}, ece={metrics['ece']:.4f}, "
            f"auroc={metrics['auroc']:.4f}, brier={metrics['brier']:.4f}, f1={metrics['f1']:.4f}"
        )
        save_results(
            eval_result["results"],
            os.path.join(output_dir, f"final_eval_{ds_name.lower()}.jsonl"),
        )
        if wandb is not None and wandb.run is not None:
            eval_log = {
                f"final/{ds_name}/{k}": v
                for k, v in metrics.items()
                if isinstance(v, (int, float))
            }
            wandb.log(eval_log, step=global_step)

    if wandb is not None and wandb.run is not None:
        wandb.finish()

    logger.info("Training complete!")


def parse_args():
    parser = argparse.ArgumentParser(description="GRPO + EpiCaR Training")

    # Data
    parser.add_argument("--train_data", type=str,
                        default="/mnt/chatbot30TB/jewonyeom/verl_qwen3/verl/data/math/train.parquet")
    parser.add_argument("--test_math", type=str,
                        default="/mnt/chatbot30TB/jewonyeom/verl_qwen3/verl/data/math/test.parquet")
    parser.add_argument("--test_amc23", type=str,
                        default="/mnt/chatbot30TB/jewonyeom/verl_qwen3/verl/data/amc23/test.parquet")
    parser.add_argument("--test_aime2025", type=str,
                        default="/mnt/chatbot30TB/jewonyeom/verl_qwen3/verl/data/aime2025/test.parquet")

    # Model
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B-Base")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_flash_attn", action="store_true", default=False,
                        help="(ignored, auto-detects best attention impl)")

    # GRPO
    parser.add_argument("--grpo_lr", type=float, default=1e-6) # ✅ LR 1e-6으로 변경
    parser.add_argument("--kl_coeff", type=float, default=0.01)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--grpo_grad_accum", type=int, default=4)

    # SFT Calibration
    parser.add_argument("--sft_lr", type=float, default=1e-6) # ✅ SFT LR도 1e-6으로 맞춤 (사용 안되지만 argparse 통일성)
    parser.add_argument("--sft_epochs", type=int, default=1)
    parser.add_argument("--sft_batch_size", type=int, default=4)
    parser.add_argument("--sft_grad_accum", type=int, default=4)

    # Training
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Number of unique problems per GRPO batch")
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--eval_samples", type=int, default=100,
                        help="Samples per dataset for periodic eval (0=full)")
    parser.add_argument("--output_dir", type=str, default="./outputs_grpo_epicar")

    # Wandb
    parser.add_argument("--wandb_project", type=str, default="grpo-epicar")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
