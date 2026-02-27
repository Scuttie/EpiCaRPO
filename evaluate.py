"""
Evaluation script for GRPO + EpiCaR trained models.
Computes accuracy and calibration metrics using verbalized confidence.
"""
import json
import os
import re
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score, brier_score_loss, f1_score
import logging

logger = logging.getLogger(__name__)

VERBALIZATION_CONFIG = {
    "injection": "\nIs the answer correct? Choose ONLY one letter. A) Yes B) No. Your choice:",
    "token_yes": " A",
    "token_no": " B",
}


def extract_boxed_answer(text: str) -> Optional[str]:
    if not text:
        return None
    idx = text.rfind("\\boxed")
    if idx < 0:
        idx = text.rfind("\\fbox")
    if idx < 0:
        return None
    depth = 0
    start_brace = -1
    for j in range(idx, len(text)):
        if text[j] == "{":
            if depth == 0:
                start_brace = j
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0 and start_brace != -1:
                return text[start_brace + 1 : j]
    return None


def _strip_string(string):
    string = str(string)
    string = string.replace("\n", "").replace("\\!", "").replace("\\\\", "\\")
    string = string.replace("tfrac", "frac").replace("dfrac", "frac")
    string = string.replace("\\left", "").replace("\\right", "")
    string = string.replace(" .", " 0.").replace("{.", "{0.")
    if not string:
        return ""
    if string[0] == ".":
        string = "0" + string
    return string.replace(" ", "")


def mathd_normalize(answer: str) -> str:
    if not answer:
        return ""
    answer = answer.strip()
    try:
        m = re.search(r"^\\text\{(?P<text>.+?)\}$", answer)
        if m:
            answer = m.group("text").strip()
    except:
        pass
    return _strip_string(answer)


def grade_answer(pred: str, solution: str) -> bool:
    if not solution:
        return False
    gt_extracted = extract_boxed_answer(solution) or solution
    pred_extracted = extract_boxed_answer(pred) or pred
    if pred_extracted is None:
        return False
    gt_norm = mathd_normalize(gt_extracted)
    pred_norm = mathd_normalize(pred_extracted)
    if gt_norm == pred_norm:
        return True
    try:
        import sympy
        from sympy.parsing import sympy_parser

        diff = f"({gt_norm})-({pred_norm})".replace("^", "**")
        parsed = sympy_parser.parse_expr(
            diff,
            transformations=(
                sympy_parser.standard_transformations
                + (sympy_parser.implicit_multiplication_application,)
            ),
        )
        if sympy.simplify(parsed) == 0:
            return True
    except:
        pass
    return False


def get_verbalized_confidence(
    model,
    tokenizer,
    prompt: str,
    response: str,
    device: str = "cuda",
) -> float:
    """
    Compute verbalized confidence P(A) / (P(A) + P(B)) for a given response.
    """
    injection = VERBALIZATION_CONFIG["injection"]
    full_text = prompt + response + injection

    inputs = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # Get logits for the next token (after the injection prompt)
        next_token_logits = outputs.logits[:, -1, :]

    # Get token ids for A and B
    yes_ids = tokenizer.encode(VERBALIZATION_CONFIG["token_yes"], add_special_tokens=False)
    no_ids = tokenizer.encode(VERBALIZATION_CONFIG["token_no"], add_special_tokens=False)
    yes_id = yes_ids[0]
    no_id = no_ids[0]

    logit_a = next_token_logits[0, yes_id].item()
    logit_b = next_token_logits[0, no_id].item()

    # Softmax over A and B only
    max_logit = max(logit_a, logit_b)
    prob_a = np.exp(logit_a - max_logit)
    prob_b = np.exp(logit_b - max_logit)
    confidence = prob_a / (prob_a + prob_b)

    return confidence


@torch.no_grad()
def generate_and_evaluate(
    model,
    tokenizer,
    dataset: pd.DataFrame,
    max_new_tokens: int = 2048,
    temperature: float = 0.0,
    batch_size: int = 1,
    device: str = "cuda",
    dataset_name: str = "test",
) -> Dict:
    """
    Generate solutions, grade them, and compute calibration metrics.

    Args:
        dataset: DataFrame with 'problem' and 'solution' columns
        Returns: dict with accuracy, calibration metrics, and per-sample results
    """
    model.eval()

    style = {
        "template": (
            "<|im_start|>system\nYou are a helpful assistant. The user asks a question, and you solve it.<|im_end|>\n"
            "<|im_start|>user\n{question} Put your final answer within \\boxed{{}}.<|im_end|>\n"
            "<|im_start|>assistant\nLet me solve this step by step."
        ),
        "stop_ids": [151643, 151645],
    }

    results = []
    all_confidences = []
    all_corrects = []

    # Resolve columns - handle VERL format
    if "prompt" in dataset.columns and "reward_model" in dataset.columns:
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

        problems = dataset["prompt"].apply(extract_problem).tolist()
        solutions = dataset["reward_model"].apply(extract_ground_truth).tolist()
    else:
        prob_col = None
        for c in ["problem", "question", "prompt", "input", "query", "content"]:
            if c in dataset.columns:
                prob_col = c
                break
        sol_col = None
        for c in ["solution", "answer", "target", "output", "expected_answer", "ground_truth"]:
            if c in dataset.columns:
                sol_col = c
                break
        if prob_col is None:
            raise KeyError(f"No problem column found. Columns: {dataset.columns.tolist()}")
        if sol_col is None:
            raise KeyError(f"No solution column found. Columns: {dataset.columns.tolist()}")
        problems = dataset[prob_col].tolist()
        solutions = dataset[sol_col].tolist()

    for i in tqdm(range(len(problems)), desc=f"Evaluating {dataset_name}"):
        problem = problems[i]
        solution = solutions[i]

        prompt = style["template"].format(question=problem)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs["input_ids"].to(device)

        # Generate
        if temperature == 0.0:
            gen_kwargs = {"do_sample": False, "max_new_tokens": max_new_tokens}
        else:
            gen_kwargs = {
                "do_sample": True,
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
            }

        output_ids = model.generate(
            input_ids,
            **gen_kwargs,
            eos_token_id=style["stop_ids"],
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

        response_ids = output_ids[0, input_ids.shape[1] :]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

        # Grade
        is_correct = grade_answer(response_text, solution)

        # Verbalized confidence
        confidence = get_verbalized_confidence(
            model, tokenizer, prompt, response_text, device
        )

        results.append(
            {
                "problem": problem,
                "solution": solution,
                "response": response_text,
                "pred_answer": extract_boxed_answer(response_text),
                "is_correct": is_correct,
                "verbal_prob": confidence,
            }
        )
        all_confidences.append(confidence)
        all_corrects.append(1.0 if is_correct else 0.0)

    # Compute metrics
    metrics = compute_calibration_metrics(
        np.array(all_confidences), np.array(all_corrects)
    )
    metrics["dataset"] = dataset_name
    metrics["num_samples"] = len(results)

    return {"metrics": metrics, "results": results}


def compute_calibration_metrics(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> Dict[str, float]:
    """Compute comprehensive calibration metrics."""
    mask = np.isfinite(probs)
    probs = probs[mask]
    labels = labels[mask]

    if len(probs) == 0:
        return {"acc": 0, "ece": 0, "auroc": 0, "brier": 0, "f1": 0}

    accuracy = np.mean(labels)

    # Brier Score
    probs_clipped = np.clip(probs, 0, 1)
    brier = brier_score_loss(labels, probs_clipped)

    # ECE
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(n_bins):
        lower, upper = bin_boundaries[i], bin_boundaries[i + 1]
        if i == 0:
            in_bin = (probs_clipped >= lower) & (probs_clipped <= upper)
        else:
            in_bin = (probs_clipped > lower) & (probs_clipped <= upper)

        count = np.sum(in_bin)
        bin_counts.append(count)

        if count > 0:
            acc_in_bin = np.mean(labels[in_bin])
            conf_in_bin = np.mean(probs_clipped[in_bin])
            ece += (count / len(probs)) * np.abs(acc_in_bin - conf_in_bin)
            bin_accs.append(acc_in_bin)
            bin_confs.append(conf_in_bin)
        else:
            bin_accs.append(0.0)
            bin_confs.append(0.0)

    # AUROC
    try:
        auroc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5
    except:
        auroc = 0.5

    # Best F1
    try:
        best_f1 = 0
        thresholds = np.unique(probs_clipped)
        if len(thresholds) > 100:
            thresholds = np.percentile(thresholds, np.linspace(0, 100, 100))
        for t in thresholds:
            preds = (probs_clipped >= t).astype(int)
            f1 = f1_score(labels, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
    except:
        best_f1 = 0.0

    return {
        "acc": float(accuracy),
        "ece": float(ece),
        "auroc": float(auroc),
        "brier": float(brier),
        "f1": float(best_f1),
        "bin_accs": [float(x) for x in bin_accs],
        "bin_confs": [float(x) for x in bin_confs],
        "bin_counts": [int(x) for x in bin_counts],
    }


def save_results(results: List[Dict], output_path: str):
    """Save per-sample results as JSONL."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(results)} results to {output_path}")
