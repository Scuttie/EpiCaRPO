"""
Isolated vLLM rollout worker.
Runs in a subprocess to avoid CUDA context contamination with the main training process.

Usage: python rollout_worker.py <input.json> <output.json>
"""
import sys
import json
import numpy as np
from reward_fn import grade_answer


def main():
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    with open(input_path) as f:
        cfg = json.load(f)

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=cfg["model_path"],
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=cfg["gpu_util"],
        max_model_len=cfg["max_tokens"] + 512,
        enforce_eager=True,
    )

    sampling_params = SamplingParams(
        n=cfg["group_size"],
        temperature=cfg["temperature"],
        max_tokens=cfg["max_tokens"],
        stop=["<|im_end|>", "<|endoftext|>"],
    )

    outputs = llm.generate(cfg["prompts"], sampling_params)

    all_prompts, all_responses, all_rewards, all_is_corrects = [], [], [], []
    for output, prompt, sol in zip(outputs, cfg["prompts"], cfg["solutions"]):
        for resp_obj in output.outputs:
            resp = resp_obj.text
            is_correct = grade_answer(resp, sol)
            all_prompts.append(prompt)
            all_responses.append(resp)
            all_rewards.append(1.0 if is_correct else 0.0)
            all_is_corrects.append(is_correct)

    result = {
        "prompts": all_prompts,
        "responses": all_responses,
        "rewards": all_rewards,
        "is_corrects": all_is_corrects,
    }

    print(f"[Rollout] reward={np.mean(all_rewards):.4f}, "
          f"correct={sum(all_is_corrects)}/{len(all_is_corrects)}", flush=True)

    with open(output_path, "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
