"""
Reward function for VERL GRPO training on math tasks.
Provides binary reward based on answer correctness.
"""
import re
import torch
import numpy as np

try:
    import sympy
    from sympy.parsing import sympy_parser
except ImportError:
    sympy = None

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


def extract_boxed_answer(text: str) -> str:
    if not text:
        return None
    idx = text.rfind("\\boxed")
    if idx < 0:
        idx = text.rfind("\\fbox")
    if idx < 0:
        return None

    i = idx
    depth = 0
    start_brace = -1
    for j in range(i, len(text)):
        if text[j] == "{":
            if depth == 0:
                start_brace = j
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0 and start_brace != -1:
                return text[start_brace + 1 : j]
    return None


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

    if sympy is not None:
        try:
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


def compute_score(solution_str: str, ground_truth: str, **kwargs) -> float:
    """VERL-compatible reward function. Returns 1.0 for correct, 0.0 for incorrect."""
    is_correct = grade_answer(solution_str, ground_truth)
    return 1.0 if is_correct else 0.0
