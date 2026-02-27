"""
Standalone evaluation script.
Run after training to evaluate a checkpoint on all test sets.

Usage:
    python eval_checkpoint.py \
        --model_path ./outputs_grpo_epicar/run_xxx/final_model \
        --output_dir ./eval_results
"""
import os
import json
import argparse
import logging

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import wandb
except ImportError:
    wandb = None

from evaluate import (
    generate_and_evaluate,
    compute_calibration_metrics,
    save_results,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_reliability_diagram(metrics: dict, dataset_name: str, save_path: str):
    """Plot reliability diagram from calibration metrics."""
    bin_accs = metrics.get("bin_accs", [])
    bin_confs = metrics.get("bin_confs", [])
    bin_counts = metrics.get("bin_counts", [])

    if not bin_accs:
        return

    n_bins = len(bin_accs)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = 1.0 / n_bins

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 7), height_ratios=[3, 1])

    # Reliability diagram
    ax1.bar(
        bin_edges[:-1], bin_accs, width=bin_width, align="edge",
        edgecolor="black", color="royalblue", alpha=0.7, label="Accuracy",
    )
    ax1.plot([0, 1], [0, 1], "--", color="gray", linewidth=1.5, label="Perfect")

    stats_text = (
        f"Acc: {metrics['acc']:.3f}\n"
        f"ECE: {metrics['ece']:.3f}\n"
        f"AUROC: {metrics['auroc']:.3f}\n"
        f"Brier: {metrics['brier']:.3f}\n"
        f"F1: {metrics['f1']:.3f}"
    )
    ax1.text(
        0.05, 0.95, stats_text, transform=ax1.transAxes,
        fontsize=10, verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Accuracy")
    ax1.set_title(f"Reliability Diagram - {dataset_name}")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.15, linestyle=":")

    # Histogram of confidence values
    ax2.bar(
        bin_edges[:-1], bin_counts, width=bin_width, align="edge",
        edgecolor="black", color="salmon", alpha=0.7,
    )
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Confidence")
    ax2.set_ylabel("Count")
    ax2.grid(True, alpha=0.15, linestyle=":")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved reliability diagram: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./eval_results")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Max samples per dataset (0=all)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_flash_attn", action="store_true", default=True)
    parser.add_argument("--wandb_project", type=str, default="")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if wandb is not None and args.wandb_project:
        wandb.init(project=args.wandb_project, name=f"eval_{os.path.basename(args.model_path)}")

    # Load model
    logger.info(f"Loading model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if args.use_flash_attn else "eager",
    ).to(args.device)
    model.eval()

    # Test datasets
    test_sets = {
        "MATH": "/mnt/chatbot30TB/jewonyeom/verl_qwen3/verl/data/math/test.parquet",
        "AMC23": "/mnt/chatbot30TB/jewonyeom/verl_qwen3/verl/data/amc23/test.parquet",
        "AIME2025": "/mnt/chatbot30TB/jewonyeom/verl_qwen3/verl/data/aime2025/test.parquet",
    }

    all_metrics = {}

    for ds_name, ds_path in test_sets.items():
        if not os.path.exists(ds_path):
            logger.warning(f"Dataset not found: {ds_path}, skipping {ds_name}")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating on {ds_name}: {ds_path}")

        df = pd.read_parquet(ds_path)
        if "question" in df.columns and "problem" not in df.columns:
            df = df.rename(columns={"question": "problem"})
        if "answer" in df.columns and "solution" not in df.columns:
            df = df.rename(columns={"answer": "solution"})

        if args.max_samples > 0:
            df = df.head(args.max_samples)

        result = generate_and_evaluate(
            model=model,
            tokenizer=tokenizer,
            dataset=df,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            device=args.device,
            dataset_name=ds_name,
        )

        metrics = result["metrics"]
        all_metrics[ds_name] = metrics

        logger.info(
            f"[{ds_name}] acc={metrics['acc']:.4f}, ece={metrics['ece']:.4f}, "
            f"auroc={metrics['auroc']:.4f}, brier={metrics['brier']:.4f}, "
            f"f1={metrics['f1']:.4f}, n={metrics['num_samples']}"
        )

        # Save per-sample results
        save_results(
            result["results"],
            os.path.join(args.output_dir, f"{ds_name.lower()}_results.jsonl"),
        )

        # Plot reliability diagram
        plot_reliability_diagram(
            metrics, ds_name,
            os.path.join(args.output_dir, f"{ds_name.lower()}_reliability.png"),
        )

        if wandb is not None and wandb.run is not None:
            log_dict = {
                f"{ds_name}/{k}": v for k, v in metrics.items()
                if isinstance(v, (int, float))
            }
            wandb.log(log_dict)
            wandb.log({
                f"{ds_name}/reliability": wandb.Image(
                    os.path.join(args.output_dir, f"{ds_name.lower()}_reliability.png")
                )
            })

    # Summary table
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'Dataset':<12} {'Acc':>8} {'ECE':>8} {'AUROC':>8} {'Brier':>8} {'F1':>8}")
    logger.info("-" * 56)
    for ds_name, m in all_metrics.items():
        logger.info(
            f"{ds_name:<12} {m['acc']:>8.4f} {m['ece']:>8.4f} "
            f"{m['auroc']:>8.4f} {m['brier']:>8.4f} {m['f1']:>8.4f}"
        )

    # Save summary
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        # Remove non-serializable items
        clean_metrics = {}
        for ds, m in all_metrics.items():
            clean_metrics[ds] = {
                k: v for k, v in m.items() if isinstance(v, (int, float, str, list))
            }
        json.dump(clean_metrics, f, indent=2)
    logger.info(f"Summary saved to {summary_path}")

    if wandb is not None and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
