import pandas as pd
import json
import os
import re
import wandb
import argparse
import numpy as np
from vllm import LLM, SamplingParams

# --- 답변 추출 도우미 ---
def extract_answer(text):
    if not isinstance(text, str):
        text = str(text)
    pattern = r'\\boxed\{(.*?)\}'
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    return text.split()[-1].strip()

class VMAMLTrainer:
    def __init__(self, model_path, train_file, test_files, n_gpu):
        self.llm = LLM(
            model=model_path, 
            tensor_parallel_size=n_gpu, 
            gpu_memory_utilization=0.8,
            max_model_len=8192 
        )
        self.sampling_params = SamplingParams(temperature=0.0, max_tokens=2048)
        self.train_df = pd.read_parquet(train_file)
        self.test_files = test_files
        # MAML의 초기 상태(theta): 모든 작업에 공통적인 기초
        self.theta = "Solve math problems step by step. State your final answer in \\boxed{}."

    def parse_data(self, row):
        prompt_data = row['prompt']
        if isinstance(prompt_data, (list, np.ndarray)):
            question = prompt_data[0]['content']
        else:
            question = str(prompt_data)
        gt = row['reward_model'].get('ground_truth', '')
        return question, str(gt)

    def get_verbal_gradient(self, question, failed_reasoning, ground_truth):
        """Inner Loop: 개별 문제의 손실로부터 교훈(Gradient) 추출"""
        grad_prompt = f"""
        [Thinking Review]
        Analyze the mistake in the reasoning below and extract one key lesson.
        
        Question: {question}
        Failed Reasoning: {failed_reasoning}
        Correct Answer: {ground_truth}
        
        Task: Provide a concise, one-sentence 'Action Rule' to avoid this error. Focus on the logical structure.
        """
        output = self.llm.generate([grad_prompt], self.sampling_params)
        return output[0].outputs[0].text

    def meta_update(self, lessons):
        """Outer Loop: 여러 교훈을 모아 '공통 가이드라인'으로 진화"""
        
        # [수정] 입력되는 교훈의 개수를 최대 8개로 제한 (토큰 폭주 방지)
        if len(lessons) > 8:
            import random
            lessons = random.sample(lessons, 8)

        meta_prompt = f"""
        [Refining the Master Protocol]
        Synthesize these specific lessons into a compact, high-level thinking protocol.
        
        Current Rules to Integrate:
        {chr(10).join(f"- {l}" for l in lessons)}
        
        Task: Update the System Prompt. It must be a unified guide that remains under 1000 tokens. 
        Focus on universal reasoning steps and avoid repeating specific problem data.
        """

        # [수정] 메타 프롬프트 자체도 너무 길면 마지막 6000토큰만 남김 (Safety Crop)
        if len(meta_prompt) > 6000:
            meta_prompt = meta_prompt[-6000:]

        try:
            output = self.llm.generate([meta_prompt], self.sampling_params)
            return output[0].outputs[0].text
        except ValueError as e:
            print(f"Update failed due to length: {e}. Returning previous theta.")
            return self.theta

    def evaluate(self, current_theta, file_path):
        df = pd.read_parquet(file_path)
        test_batch = df.sample(min(50000, len(df)))
        
        test_prompts = []
        gts = []
        for _, row in test_batch.iterrows():
            q, gt = self.parse_data(row)
            # 프롬프트 구성 시 시스템 프롬프트가 너무 길면 잘라내는 안전 장치
            truncated_theta = current_theta[-4000:] if len(current_theta) > 4000 else current_theta
            test_prompts.append(f"System: {truncated_theta}\nUser: {q}")
            gts.append(gt)
            
        outputs = self.llm.generate(test_prompts, self.sampling_params)
        
        correct = 0
        for out, gt in zip(outputs, gts):
            pred = extract_answer(out.outputs[0].text)
            if pred == extract_answer(gt):
                correct += 1
        return correct / len(test_batch)

    def train(self, epochs, batch_size):
        wandb.init(project="V-MAML-Math", name=f"v_maml_{wandb.util.generate_id()}")
        
        for epoch in range(epochs):
            batch = self.train_df.sample(batch_size)
            questions, gts = [], []
            for _, row in batch.iterrows():
                q, gt = self.parse_data(row)
                questions.append(q)
                gts.append(gt)

            # 1. Inference: 현재의 theta로 작업 수행 
            prompts = [f"System: {self.theta}\nUser: {q}" for q in questions]
            results = self.llm.generate(prompts, self.sampling_params)
            
            lessons = []
            for i, res in enumerate(results):
                pred_text = res.outputs[0].text
                if extract_answer(pred_text) != extract_answer(gts[i]):
                    lesson = self.get_verbal_gradient(questions[i], pred_text, gts[i])
                    lessons.append(lesson)

            # 2. Meta-Update: θ 업데이트 
            if lessons:
                self.theta = self.meta_update(lessons)

            # 3. Logging & Evaluation
            log_data = {"epoch": epoch, "master_prompt": self.theta}
            for test_file in self.test_files:
                name = os.path.basename(os.path.dirname(test_file)) or os.path.basename(test_file).split('.')[0]
                acc = self.evaluate(self.theta, test_file)
                log_data[f"test_acc/{name}"] = acc
                print(f"Epoch {epoch} | Accuracy on {name}: {acc:.4f}")
            
            wandb.log(log_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, nargs="+", required=True)
    parser.add_argument("--n_gpu", type=int, default=4)
    args = parser.parse_args()

    trainer = VMAMLTrainer(args.model, args.train_data, args.test_data, args.n_gpu)
    trainer.train(epochs=10, batch_size=16)