"""
GRPO + SFT Calibration Training (EpiCaR-style)

Memory strategy for 2x A100 80GB:
  Phase 1 (Rollout): vLLM on GPU 0 → batch generate → destroy vLLM
  Phase 2 (Training): train model on GPU 0, ref model on GPU 1 → GRPO + SFT
  Phase 3 (Sync):     save weights → next step vLLM reloads them

~2 min/step vs 51 min/step with HF sequential generation.
"""
import os
import gc
import json
import logging
import argparse
from datetime import datetime
from typing import List, Dict

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import wandb
except ImportError:
    wandb = None

from reward_fn import compute_score, grade_answer, extract_boxed_answer
from sft_calibration import SFTCalibrationTrainer, VERBALIZATION_CONFIG
from evaluate import generate_and_evaluate, save_results

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d: %(message)s")
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

        def extract_gt(rm):
            return rm.get("ground_truth", "") if isinstance(rm, dict) else str(rm)

        df["problem"] = df["prompt"].apply(extract_problem)
        df["solution"] = df["reward_model"].apply(extract_gt)
        logger.info("  Parsed VERL format")
        return df

    col_map = {}
    for c in ["problem", "question", "input", "query", "content"]:
        if c in df.columns: col_map[c] = "problem"; break
    for c in ["solution", "answer", "target", "output", "ground_truth"]:
        if c in df.columns: col_map[c] = "solution"; break
    if col_map:
        df = df.rename(columns=col_map)
    if "problem" not in df.columns:
        raise KeyError(f"No problem column. Available: {df.columns.tolist()}")
    return df


def prepare_prompt(problem: str) -> str:
    return (
        "<|im_start|>system\nYou are a helpful assistant. The user asks a question, "
        "and you solve it.<|im_end|>\n"
        f"<|im_start|>user\n{problem} Put your final answer within \\boxed{{}}.<|im_end|>\n"
        "<|im_start|>assistant\nLet me solve this step by step."
    )


# ============================================================
# Phase 1: vLLM Rollout (then destroyed to free GPU)
# ============================================================

def vllm_rollout(model_path: str, prompts: List[str], solutions: List[str],
                 group_size: int = 8, temperature: float = 0.7,
                 max_tokens: int = 2048, gpu_util: float = 0.85) -> Dict:
    """
    Create vLLM engine, generate responses, grade them, then destroy engine.
    Uses nearly full GPU since it's the only thing running.
    """
    from vllm import LLM, SamplingParams

    logger.info(f"[Rollout] Loading vLLM: {model_path}")
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_util,
        max_model_len=max_tokens + 512,
        enforce_eager=True,
    )

    sampling_params = SamplingParams(
        n=group_size,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=["<|im_end|>", "<|endoftext|>"],
    )

    logger.info(f"[Rollout] Generating {len(prompts)} × {group_size} = {len(prompts) * group_size} responses")
    outputs = llm.generate(prompts, sampling_params)

    # Flatten results
    all_prompts, all_responses, all_rewards, all_is_corrects = [], [], [], []

    for output, prompt, sol in zip(outputs, prompts, solutions):
        for resp_obj in output.outputs:
            resp = resp_obj.text
            is_correct = grade_answer(resp, sol)
            all_prompts.append(prompt)
            all_responses.append(resp)
            all_rewards.append(1.0 if is_correct else 0.0)
            all_is_corrects.append(is_correct)

    logger.info(f"[Rollout] Done. reward={np.mean(all_rewards):.4f}, "
                f"correct={sum(all_is_corrects)}/{len(all_is_corrects)}. Destroying vLLM...")

    # Properly shutdown vLLM (kills EngineCore subprocess)
    try:
        if hasattr(llm, 'llm_engine') and hasattr(llm.llm_engine, 'shutdown'):
            llm.llm_engine.shutdown()
        elif hasattr(llm, 'shutdown'):
            llm.shutdown()
    except Exception as e:
        logger.warning(f"vLLM shutdown warning: {e}")

    del llm

    # Force kill any remaining vLLM subprocesses (children of this process)
    import subprocess, signal
    my_pid = os.getpid()
    try:
        # Find child processes, but exclude wandb
        result = subprocess.run(
            ["pgrep", "-P", str(my_pid)], capture_output=True, text=True
        )
        for pid_str in result.stdout.strip().split("\n"):
            pid_str = pid_str.strip()
            if pid_str and pid_str.isdigit():
                pid = int(pid_str)
                # Check if it's a wandb process before killing
                try:
                    cmdline = subprocess.run(
                        ["ps", "-p", str(pid), "-o", "args="],
                        capture_output=True, text=True
                    ).stdout.strip()
                    if "wandb" in cmdline.lower():
                        continue  # Don't kill wandb
                except Exception:
                    pass
                try:
                    os.kill(pid, signal.SIGKILL)
                    logger.info(f"Killed vLLM child process: {pid}")
                except ProcessLookupError:
                    pass
    except Exception:
        pass

    gc.collect()
    torch.cuda.empty_cache()
    import time; time.sleep(2)  # Wait for GPU memory to be freed
    logger.info("[Rollout] vLLM freed.")

    return {
        "prompts": all_prompts,
        "responses": all_responses,
        "rewards": all_rewards,
        "is_corrects": all_is_corrects,
    }


# ============================================================
# Phase 2: GRPO + SFT Update (PyTorch)
# ============================================================

class GRPOUpdater:
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

    def update(self, prompts, responses, rewards, group_size) -> Dict[str, float]:
        self.model.train()
        n = len(rewards)
        rewards_arr = np.array(rewards)

        # Group-normalize advantages
        advantages = np.zeros(n)
        for i in range(n // group_size):
            s, e = i * group_size, (i + 1) * group_size
            g = rewards_arr[s:e]
            advantages[s:e] = (g - g.mean()) / (g.std() + 1e-8)

        total_loss, total_pg, total_kl, steps = 0.0, 0.0, 0.0, 0
        self.optimizer.zero_grad()

        for step_i, idx in enumerate(np.random.permutation(n)):
            adv = advantages[idx]
            if abs(adv) < 1e-8:
                continue

            full_text = prompts[idx] + responses[idx]
            enc = self.tokenizer(full_text, return_tensors="pt", add_special_tokens=False,
                                 truncation=True, max_length=2560)
            input_ids = enc["input_ids"].to(self.device)

            prompt_enc = self.tokenizer(prompts[idx], add_special_tokens=False)
            prompt_len = len(prompt_enc["input_ids"])
            if prompt_len >= input_ids.shape[1]:
                continue

            # Policy forward
            out = self.model(input_ids=input_ids)
            logits = out.logits[:, prompt_len - 1:-1, :]
            resp_ids = input_ids[:, prompt_len:]
            lp = torch.log_softmax(logits, dim=-1)
            token_lp = lp.gather(2, resp_ids.unsqueeze(-1)).squeeze(-1)

            # Ref model forward (different GPU)
            with torch.no_grad():
                ref_ids = input_ids.to(self.device_ref)
                ref_out = self.ref_model(input_ids=ref_ids)
                ref_logits = ref_out.logits[:, prompt_len - 1:-1, :]
                ref_lp = torch.log_softmax(ref_logits, dim=-1)
                ref_token_lp = ref_lp.gather(2, ref_ids[:, prompt_len:].unsqueeze(-1)).squeeze(-1)
                ref_token_lp = ref_token_lp.to(self.device)

            ratio = torch.exp(token_lp - ref_token_lp.detach())
            clipped = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
            pg_loss = torch.max(-adv * ratio, -adv * clipped).mean()
            kl = (token_lp - ref_token_lp.detach()).mean()
            loss = pg_loss + self.kl_coeff * kl

            (loss / self.grad_accum).backward()
            total_loss += loss.item(); total_pg += pg_loss.item(); total_kl += kl.item()
            steps += 1

            if (step_i + 1) % self.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

        if steps % self.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()

        torch.cuda.empty_cache()
        s = max(steps, 1)
        return {
            "grpo_loss": total_loss / s, "grpo_pg_loss": total_pg / s,
            "grpo_kl": total_kl / s, "grpo_mean_reward": float(rewards_arr.mean()),
            "grpo_update_steps": steps,
        }


# ============================================================
# Main
# ============================================================

def main(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    weights_dir = os.path.join(output_dir, "_latest_weights")

    if wandb is not None and args.wandb_project:
        os.environ["WANDB__SERVICE_WAIT"] = "300"
        os.environ["WANDB_START_METHOD"] = "thread"
        wandb.init(project=args.wandb_project,
                   name=args.wandb_run_name or f"grpo_epicar_{timestamp}",
                   config=vars(args))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_model = "cuda:0"
    device_ref = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    logger.info(f"Devices: model={device_model}, ref={device_ref}, GPUs={torch.cuda.device_count()}")

    # Current model path (vLLM loads from here each step)
    current_model_path = args.model_name  # First step: load from HF

    # Data
    train_df = load_parquet_dataset(args.train_data)
    test_datasets = {
        "MATH": load_parquet_dataset(args.test_math),
        "AMC23": load_parquet_dataset(args.test_amc23),
        "AIME2025": load_parquet_dataset(args.test_aime2025),
    }
    problems = train_df["problem"].tolist()
    solutions = train_df["solution"].tolist()

    # Ref model stays loaded (frozen, GPU 1)
    logger.info("Loading reference model (frozen, stays on GPU 1)")
    attn_impl = "sdpa"
    try:
        import flash_attn; attn_impl = "flash_attention_2"
    except ImportError:
        pass
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation=attn_impl,
    ).to(device_ref)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # ---- Training Loop ----
    global_step = 0
    train_model = None
    grpo = None
    sft = None

    for epoch in range(args.num_epochs):
        logger.info(f"=== Epoch {epoch + 1}/{args.num_epochs} ===")
        indices = np.random.permutation(len(problems))

        for batch_start in range(0, len(problems), args.batch_size):
            batch_end = min(batch_start + args.batch_size, len(problems))
            batch_idx = indices[batch_start:batch_end]
            batch_problems = [problems[i] for i in batch_idx]
            batch_solutions = [solutions[i] for i in batch_idx]
            global_step += 1

            logger.info(f"Step {global_step}: {len(batch_problems)} problems × {args.group_size}")

            # ======== Phase 1: vLLM Rollout (GPU 0 only) ========
            # Make sure training model is NOT on GPU 0
            if train_model is not None:
                del train_model, grpo, sft
                train_model, grpo, sft = None, None, None
                gc.collect()
                torch.cuda.empty_cache()

            prompts = [prepare_prompt(p) for p in batch_problems]

            rollout = vllm_rollout(
                model_path=current_model_path,
                prompts=prompts,
                solutions=batch_solutions,
                group_size=args.group_size,
                temperature=args.temperature,
                max_tokens=args.max_new_tokens,
                gpu_util=args.vllm_gpu_util,
            )

            # ======== Phase 2: Load training model + GRPO + SFT (GPU 0 + 1) ========
            logger.info("[Training] Loading model for GRPO + SFT")
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

            # Share GRPO's optimizer with SFT (saves ~16GB of optimizer states)
            sft = SFTCalibrationTrainer(
                model=train_model, tokenizer=tokenizer, optimizer=grpo.optimizer,
                lr=args.sft_lr, sft_epochs=args.sft_epochs,
                sft_batch_size=args.sft_batch_size, sft_grad_accum=args.sft_grad_accum,
                max_length=1024, device=device_model,
            )

            # GRPO update
            grpo_metrics = grpo.update(
                rollout["prompts"], rollout["responses"], rollout["rewards"], args.group_size
            )
            logger.info(f"GRPO: loss={grpo_metrics['grpo_loss']:.4f}, reward={grpo_metrics['grpo_mean_reward']:.4f}")
            torch.cuda.empty_cache()

            # SFT calibration
            sft_metrics = sft.train_on_rollouts(
                prompts=rollout["prompts"],
                responses=rollout["responses"],
                is_corrects=rollout["is_corrects"],
            )
            logger.info(f"SFT: loss={sft_metrics['sft_loss']:.4f}, pred_acc={sft_metrics['sft_pred_accuracy']:.4f}")

            # ======== Phase 3: Save weights for next vLLM step ========
            train_model.save_pretrained(weights_dir)
            tokenizer.save_pretrained(weights_dir)
            current_model_path = weights_dir
            logger.info(f"Weights saved to {weights_dir}")

            # Logging
            log_dict = {**grpo_metrics, **sft_metrics, "epoch": epoch, "step": global_step}
            if wandb is not None and args.wandb_project:
                try:
                    if wandb.run is None:
                        wandb.init(project=args.wandb_project,
                                   name=args.wandb_run_name or f"grpo_epicar_{timestamp}",
                                   config=vars(args), resume="allow")
                    wandb.log(log_dict, step=global_step)
                except Exception as e:
                    logger.warning(f"wandb log failed: {e}. Will retry next step.")

            # ======== Periodic Evaluation ========
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
                    save_results(eval_result["results"],
                                 os.path.join(output_dir, f"eval_step{global_step}_{ds_name.lower()}.jsonl"))
                    if wandb is not None and args.wandb_project:
                        try:
                            if wandb.run is not None:
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
            if wandb is not None and args.wandb_project:
                try:
                    if wandb.run is not None:
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
    p = argparse.ArgumentParser()
    p.add_argument("--train_data", default="/mnt/chatbot30TB/jewonyeom/verl_qwen3/verl/data/math/train.parquet")
    p.add_argument("--test_math", default="/mnt/chatbot30TB/jewonyeom/verl_qwen3/verl/data/math/test.parquet")
    p.add_argument("--test_amc23", default="/mnt/chatbot30TB/jewonyeom/verl_qwen3/verl/data/amc23/test.parquet")
    p.add_argument("--test_aime2025", default="/mnt/chatbot30TB/jewonyeom/verl_qwen3/verl/data/aime2025/test.parquet")
    p.add_argument("--model_name", default="Qwen/Qwen3-4B-Base")
    p.add_argument("--vllm_gpu_util", type=float, default=0.85)
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
    main(parse_args())
