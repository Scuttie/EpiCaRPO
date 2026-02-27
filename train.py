"""
Main Training Script: GRPO + SFT Calibration (EpiCaR-style)
Uses vLLM for fast batched rollout generation.

Training Loop per step:
  1. vLLM batch rollout (all problems × group_size in one call)
  2. GRPO policy update (PyTorch)
  3. SFT calibration (verbalized confidence A/B)
  4. Sync updated weights back to vLLM

Usage:
    python train.py
"""
import os
import gc
import json
import logging
import argparse
from datetime import datetime
from typing import List, Dict, Optional

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

try:
    import wandb
except ImportError:
    wandb = None

from reward_fn import compute_score, grade_answer, extract_boxed_answer
from sft_calibration import SFTCalibrationTrainer, VERBALIZATION_CONFIG
from evaluate import generate_and_evaluate, save_results

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# Data Loading (VERL format)
# ============================================================

def load_parquet_dataset(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df)} samples from {path} | columns={df.columns.tolist()}")

    if "prompt" in df.columns and "reward_model" in df.columns:
        def extract_problem(prompt_msgs):
            if isinstance(prompt_msgs, list):
                for msg in prompt_msgs:
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        return msg.get("content", "")
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
        logger.info("  Parsed VERL format")
        return df

    col_map = {}
    for c in ["problem", "question", "input", "query", "content"]:
        if c in df.columns:
            col_map[c] = "problem"
            break
    for c in ["solution", "answer", "target", "output", "expected_answer", "ground_truth"]:
        if c in df.columns:
            col_map[c] = "solution"
            break
    if col_map:
        df = df.rename(columns=col_map)
    if "problem" not in df.columns:
        raise KeyError(f"Cannot find problem column. Available: {df.columns.tolist()}")
    return df


def prepare_prompt(problem: str) -> str:
    return (
        "<|im_start|>system\nYou are a helpful assistant. The user asks a question, "
        "and you solve it.<|im_end|>\n"
        f"<|im_start|>user\n{problem} Put your final answer within \\boxed{{}}.<|im_end|>\n"
        "<|im_start|>assistant\nLet me solve this step by step."
    )


# ============================================================
# Model Loading
# ============================================================

def load_model(model_name_or_path: str, device: str = "cuda:0"):
    attn_impl = "sdpa"
    try:
        import flash_attn
        attn_impl = "flash_attention_2"
    except ImportError:
        pass

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=attn_impl,
    ).to(device)
    return model


# ============================================================
# vLLM Rollout Engine
# ============================================================

class VLLMRolloutEngine:
    """Fast batched generation using vLLM."""

    def __init__(self, model_name: str, gpu_memory_utilization: float = 0.45,
                 tensor_parallel_size: int = 1, max_model_len: int = 3072):
        logger.info(f"Initializing vLLM engine: {model_name}")
        self.llm = LLM(
            model=model_name,
            trust_remote_code=True,
            dtype="bfloat16",
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            enforce_eager=True,  # Avoid CUDA graph issues with weight sync
        )
        self.model_name = model_name

    def generate(self, prompts: List[str], n: int = 8,
                 temperature: float = 0.7, max_tokens: int = 2048) -> List[List[str]]:
        """
        Generate n responses per prompt using vLLM batch inference.
        
        Returns: List[List[str]] - responses[i] = list of n responses for prompts[i]
        """
        sampling_params = SamplingParams(
            n=n,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=["<|im_end|>", "<|endoftext|>"],
        )

        outputs = self.llm.generate(prompts, sampling_params)

        all_responses = []
        for output in outputs:
            responses = [o.text for o in output.outputs]
            all_responses.append(responses)

        return all_responses

    def sync_weights(self, state_dict: Dict):
        """Sync updated model weights to vLLM engine."""
        # vLLM's weight update API
        if hasattr(self.llm, 'llm_engine'):
            model_runner = self.llm.llm_engine.model_executor.driver_worker.model_runner
            if hasattr(model_runner, 'model'):
                vllm_model = model_runner.model
                vllm_model.load_weights(state_dict.items())
                logger.info("Synced weights to vLLM engine")
                return True

        logger.warning("Could not sync weights to vLLM. Will use checkpoint reload.")
        return False


# ============================================================
# HuggingFace Rollout (fallback if vLLM unavailable)
# ============================================================

class HFRolloutEngine:
    """Fallback sequential generation using HuggingFace."""

    def __init__(self, model, tokenizer, device="cuda:0"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.stop_ids = [151643, 151645]

    def generate(self, prompts: List[str], n: int = 8,
                 temperature: float = 0.7, max_tokens: int = 2048) -> List[List[str]]:
        self.model.eval()
        all_responses = []

        for prompt in tqdm(prompts, desc="HF Rollout"):
            inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = inputs["input_ids"].to(self.device)
            prompt_len = input_ids.shape[1]

            responses = []
            for _ in range(n):
                with torch.no_grad():
                    output = self.model.generate(
                        input_ids,
                        do_sample=True,
                        temperature=temperature,
                        max_new_tokens=max_tokens,
                        eos_token_id=self.stop_ids,
                        pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    )
                response_ids = output[0, prompt_len:]
                response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                responses.append(response_text)

            all_responses.append(responses)
            torch.cuda.empty_cache()

        return all_responses

    def sync_weights(self, state_dict):
        """No-op for HF (model is shared)."""
        return True


# ============================================================
# GRPO Update (PyTorch)
# ============================================================

class GRPOUpdater:
    """GRPO policy gradient update using PyTorch model."""

    def __init__(self, model, ref_model, tokenizer,
                 lr=1e-6, kl_coeff=0.01, clip_range=0.2,
                 grad_accum=4, device="cuda:0", device_ref="cuda:1"):
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.kl_coeff = kl_coeff
        self.clip_range = clip_range
        self.grad_accum = grad_accum
        self.device = device
        self.device_ref = device_ref

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    def update(self, prompts: List[str], responses: List[str],
               rewards: List[float], group_size: int) -> Dict[str, float]:
        """Run GRPO update with group-normalized advantages."""
        self.model.train()

        n_samples = len(rewards)
        n_problems = n_samples // group_size
        rewards_arr = np.array(rewards)

        # Group-normalize advantages
        advantages = np.zeros(n_samples)
        for i in range(n_problems):
            s, e = i * group_size, (i + 1) * group_size
            grp = rewards_arr[s:e]
            advantages[s:e] = (grp - grp.mean()) / (grp.std() + 1e-8)

        total_loss = 0.0
        total_pg = 0.0
        total_kl = 0.0
        steps = 0

        self.optimizer.zero_grad()
        indices = np.random.permutation(n_samples)

        for step_i, idx in enumerate(indices):
            adv = advantages[idx]
            if abs(adv) < 1e-8:
                continue

            prompt_text = prompts[idx]
            response_text = responses[idx]

            full_text = prompt_text + response_text
            enc = self.tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
            input_ids = enc["input_ids"].to(self.device)

            prompt_enc = self.tokenizer(prompt_text, add_special_tokens=False)
            prompt_len = len(prompt_enc["input_ids"])

            # Truncate if too long
            if input_ids.shape[1] > 2560:
                input_ids = input_ids[:, :2560]

            # Forward pass
            out = self.model(input_ids=input_ids)
            logits = out.logits[:, prompt_len - 1:-1, :]
            resp_ids = input_ids[:, prompt_len:]

            log_probs = torch.log_softmax(logits, dim=-1)
            token_lp = log_probs.gather(2, resp_ids.unsqueeze(-1)).squeeze(-1)

            # Ref model (may be on different device)
            with torch.no_grad():
                ref_ids = input_ids.to(self.device_ref)
                ref_out = self.ref_model(input_ids=ref_ids)
                ref_logits = ref_out.logits[:, prompt_len - 1:-1, :]
                ref_lp = torch.log_softmax(ref_logits, dim=-1)
                ref_resp = ref_ids[:, prompt_len:]
                ref_token_lp = ref_lp.gather(2, ref_resp.unsqueeze(-1)).squeeze(-1).to(self.device)

            ratio = torch.exp(token_lp - ref_token_lp.detach())
            clipped = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)

            pg_loss = torch.max(-adv * ratio, -adv * clipped).mean()
            kl = (token_lp - ref_token_lp.detach()).mean()
            loss = pg_loss + self.kl_coeff * kl

            (loss / self.grad_accum).backward()

            total_loss += loss.item()
            total_pg += pg_loss.item()
            total_kl += kl.item()
            steps += 1

            if (step_i + 1) % self.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

        # Final step
        if steps % self.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()

        torch.cuda.empty_cache()
        n = max(steps, 1)
        return {
            "grpo_loss": total_loss / n,
            "grpo_pg_loss": total_pg / n,
            "grpo_kl": total_kl / n,
            "grpo_mean_reward": float(rewards_arr.mean()),
            "grpo_std_reward": float(rewards_arr.std()),
            "grpo_update_steps": steps,
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

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Devices ----
    device_model = "cuda:0"
    device_ref = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    logger.info(f"Devices: model={device_model}, ref={device_ref}, GPUs={torch.cuda.device_count()}")

    # ---- Rollout Engine (vLLM on cuda:0) ----
    if VLLM_AVAILABLE and not args.no_vllm:
        logger.info("Using vLLM for fast batched rollout")
        rollout_engine = VLLMRolloutEngine(
            model_name=args.model_name,
            gpu_memory_utilization=args.vllm_gpu_util,
            max_model_len=args.max_new_tokens + 512,
        )
    else:
        logger.warning("vLLM not available, using HuggingFace sequential generation (SLOW)")
        hf_model = load_model(args.model_name, device=device_model)
        rollout_engine = HFRolloutEngine(hf_model, tokenizer, device=device_model)

    # ---- Training Model (PyTorch on cuda:0) ----
    logger.info("Loading training model")
    train_model = load_model(args.model_name, device=device_model)
    train_model.gradient_checkpointing_enable()

    # ---- Ref Model (frozen on cuda:1) ----
    logger.info("Loading reference model")
    ref_model = load_model(args.model_name, device=device_ref)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # ---- GRPO Updater ----
    grpo = GRPOUpdater(
        model=train_model, ref_model=ref_model, tokenizer=tokenizer,
        lr=args.grpo_lr, kl_coeff=args.kl_coeff, clip_range=args.clip_range,
        grad_accum=args.grpo_grad_accum, device=device_model, device_ref=device_ref,
    )

    # ---- SFT Calibration Trainer ----
    sft_optimizer = torch.optim.AdamW(train_model.parameters(), lr=args.sft_lr, weight_decay=0.01)
    sft = SFTCalibrationTrainer(
        model=train_model, tokenizer=tokenizer, optimizer=sft_optimizer,
        lr=args.sft_lr, sft_epochs=args.sft_epochs,
        sft_batch_size=args.sft_batch_size, sft_grad_accum=args.sft_grad_accum,
        max_length=1024, device=device_model,
    )

    # ---- Data ----
    train_df = load_parquet_dataset(args.train_data)
    test_datasets = {
        "MATH": load_parquet_dataset(args.test_math),
        "AMC23": load_parquet_dataset(args.test_amc23),
        "AIME2025": load_parquet_dataset(args.test_aime2025),
    }

    problems = train_df["problem"].tolist()
    solutions = train_df["solution"].tolist()

    # ---- Training ----
    global_step = 0

    for epoch in range(args.num_epochs):
        logger.info(f"=== Epoch {epoch + 1}/{args.num_epochs} ===")
        indices = np.random.permutation(len(problems))

        for batch_start in range(0, len(problems), args.batch_size):
            batch_end = min(batch_start + args.batch_size, len(problems))
            batch_idx = indices[batch_start:batch_end]

            batch_problems = [problems[i] for i in batch_idx]
            batch_solutions = [solutions[i] for i in batch_idx]

            global_step += 1
            n_probs = len(batch_problems)
            logger.info(f"Step {global_step}: {n_probs} problems × {args.group_size} = {n_probs * args.group_size} rollouts")

            # ---- Step 1: vLLM Batch Rollout ----
            prompts = [prepare_prompt(p) for p in batch_problems]

            responses_per_problem = rollout_engine.generate(
                prompts=prompts,
                n=args.group_size,
                temperature=args.temperature,
                max_tokens=args.max_new_tokens,
            )

            # Flatten: prompts[i] × group_size -> all_prompts, all_responses
            all_prompts = []
            all_responses = []
            all_rewards = []
            all_is_corrects = []

            for i, (prompt, sol, resps) in enumerate(
                zip(prompts, batch_solutions, responses_per_problem)
            ):
                for resp in resps:
                    is_correct = grade_answer(resp, sol)
                    all_prompts.append(prompt)
                    all_responses.append(resp)
                    all_rewards.append(1.0 if is_correct else 0.0)
                    all_is_corrects.append(is_correct)

            mean_reward = np.mean(all_rewards)
            logger.info(f"Rollout: reward={mean_reward:.4f}, correct={sum(all_is_corrects)}/{len(all_is_corrects)}")

            # ---- Step 2: GRPO Update ----
            grpo_metrics = grpo.update(
                all_prompts, all_responses, all_rewards, args.group_size
            )
            logger.info(f"GRPO: loss={grpo_metrics['grpo_loss']:.4f}, kl={grpo_metrics['grpo_kl']:.4f}")

            # ---- Step 3: SFT Calibration ----
            sft_metrics = sft.train_on_rollouts(
                prompts=all_prompts,
                responses=all_responses,
                is_corrects=all_is_corrects,
            )
            logger.info(f"SFT: loss={sft_metrics['sft_loss']:.4f}, pred_acc={sft_metrics['sft_pred_accuracy']:.4f}")

            # ---- Step 4: Sync weights to vLLM ----
            if isinstance(rollout_engine, VLLMRolloutEngine):
                rollout_engine.sync_weights(train_model.state_dict())

            torch.cuda.empty_cache()

            # ---- Logging ----
            log_dict = {**grpo_metrics, **sft_metrics, "epoch": epoch, "step": global_step}
            if wandb is not None and wandb.run is not None:
                wandb.log(log_dict, step=global_step)

            # ---- Periodic Evaluation ----
            if global_step % args.eval_every == 0:
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

                    save_results(
                        eval_result["results"],
                        os.path.join(output_dir, f"eval_step{global_step}_{ds_name.lower()}.jsonl"),
                    )
                    if wandb is not None and wandb.run is not None:
                        wandb.log(
                            {f"eval/{ds_name}/{k}": v for k, v in m.items() if isinstance(v, (int, float))},
                            step=global_step,
                        )

                train_model.train()
                torch.cuda.empty_cache()

            # ---- Checkpoint ----
            if global_step % args.save_every == 0:
                ckpt_dir = os.path.join(output_dir, f"checkpoint_step{global_step}")
                logger.info(f"Saving checkpoint: {ckpt_dir}")
                train_model.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)

    # ---- Final Save & Eval ----
    final_dir = os.path.join(output_dir, "final_model")
    train_model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info(f"Final model saved: {final_dir}")

    train_model.eval()
    for ds_name, ds_df in test_datasets.items():
        eval_result = generate_and_evaluate(
            model=train_model, tokenizer=tokenizer,
            dataset=ds_df, max_new_tokens=args.max_new_tokens,
            temperature=0.0, device=device_model, dataset_name=ds_name,
        )
        m = eval_result["metrics"]
        logger.info(f"[FINAL {ds_name}] acc={m['acc']:.4f}, ece={m['ece']:.4f}, auroc={m['auroc']:.4f}, brier={m['brier']:.4f}")
        save_results(eval_result["results"], os.path.join(output_dir, f"final_{ds_name.lower()}.jsonl"))
        if wandb is not None and wandb.run is not None:
            wandb.log({f"final/{ds_name}/{k}": v for k, v in m.items() if isinstance(v, (int, float))}, step=global_step)

    if wandb is not None and wandb.run is not None:
        wandb.finish()
    logger.info("Training complete!")


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="GRPO + EpiCaR Training (vLLM)")

    p.add_argument("--train_data", default="/mnt/chatbot30TB/jewonyeom/verl_qwen3/verl/data/math/train.parquet")
    p.add_argument("--test_math", default="/mnt/chatbot30TB/jewonyeom/verl_qwen3/verl/data/math/test.parquet")
    p.add_argument("--test_amc23", default="/mnt/chatbot30TB/jewonyeom/verl_qwen3/verl/data/amc23/test.parquet")
    p.add_argument("--test_aime2025", default="/mnt/chatbot30TB/jewonyeom/verl_qwen3/verl/data/aime2025/test.parquet")

    p.add_argument("--model_name", default="Qwen/Qwen3-4B-Base")
    p.add_argument("--no_vllm", action="store_true", help="Disable vLLM, use HF generate")
    p.add_argument("--vllm_gpu_util", type=float, default=0.45)

    p.add_argument("--grpo_lr", type=float, default=1e-6)
    p.add_argument("--kl_coeff", type=float, default=0.01)
    p.add_argument("--clip_range", type=float, default=0.2)
    p.add_argument("--group_size", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--grpo_grad_accum", type=int, default=4)

    p.add_argument("--sft_lr", type=float, default=1e-5)
    p.add_argument("--sft_epochs", type=int, default=1)
    p.add_argument("--sft_batch_size", type=int, default=2)
    p.add_argument("--sft_grad_accum", type=int, default=4)

    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument("--save_every", type=int, default=100)
    p.add_argument("--eval_samples", type=int, default=100)
    p.add_argument("--output_dir", default="./outputs_grpo_epicar")

    p.add_argument("--wandb_project", default="grpo-epicar")
    p.add_argument("--wandb_run_name", default=None)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
