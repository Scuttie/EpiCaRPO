"""
SFT Calibration Trainer for Verbalized Confidence.

After GRPO rollouts, constructs calibration data:
  - For each generated solution, append the verbalization prompt
  - Target: "A" if correct, "B" if incorrect
  - Train with cross-entropy loss only on the target token
"""
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

VERBALIZATION_CONFIG = {
    "injection": "\nIs the answer correct? Choose ONLY one letter. A) Yes B) No. Your choice:",
    "token_yes": " A",
    "token_no": " B",
}


class CalibrationDataset(Dataset):
    """Dataset for verbalized confidence SFT training."""

    def __init__(
        self,
        prompts: List[str],
        responses: List[str],
        is_corrects: List[bool],
        tokenizer,
        max_length: int = 2048,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        injection = VERBALIZATION_CONFIG["injection"]
        token_yes = VERBALIZATION_CONFIG["token_yes"]
        token_no = VERBALIZATION_CONFIG["token_no"]

        yes_ids = tokenizer.encode(token_yes, add_special_tokens=False)
        no_ids = tokenizer.encode(token_no, add_special_tokens=False)

        assert len(yes_ids) >= 1 and len(no_ids) >= 1, (
            f"Token encoding issue: yes={yes_ids}, no={no_ids}"
        )

        self.target_yes_id = yes_ids[0]
        self.target_no_id = no_ids[0]

        for prompt, response, correct in zip(prompts, responses, is_corrects):
            # Full context: prompt + response + injection prompt
            context_text = prompt + response + injection
            target_text = token_yes if correct else token_no

            context_ids = tokenizer.encode(context_text, add_special_tokens=False)
            target_ids = tokenizer.encode(target_text, add_special_tokens=False)

            # Truncate from LEFT to keep the answer end visible for calibration
            max_context = self.max_length - len(target_ids)
            if len(context_ids) > max_context:
                context_ids = context_ids[-max_context:]

            input_ids = context_ids + target_ids
            # Labels: -100 for context (no loss), actual ids for target
            labels = [-100] * len(context_ids) + target_ids

            self.samples.append(
                {"input_ids": input_ids, "labels": labels}
            )

        logger.info(
            f"CalibrationDataset: {len(self.samples)} samples, "
            f"correct={sum(is_corrects)}, incorrect={sum(1 for x in is_corrects if not x)}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch, pad_token_id=0):
    """Collate with left-padding for causal LM."""
    max_len = max(len(b["input_ids"]) for b in batch)

    input_ids_list = []
    labels_list = []
    attention_mask_list = []

    for b in batch:
        pad_len = max_len - len(b["input_ids"])
        input_ids_list.append([pad_token_id] * pad_len + b["input_ids"])
        labels_list.append([-100] * pad_len + b["labels"])
        attention_mask_list.append([0] * pad_len + [1] * len(b["input_ids"]))

    return {
        "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
        "labels": torch.tensor(labels_list, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask_list, dtype=torch.long),
    }


class SFTCalibrationTrainer:
    def __init__(
        self,
        model,
        tokenizer,
        optimizer, # ✅ 새롭게 추가된 파라미터
        lr: float = 1e-6,
        sft_epochs: int = 1,
        sft_batch_size: int = 4,
        sft_grad_accum: int = 4,
        max_length: int = 2048,
        device: Optional[str] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer # ✅ 주입받은 Optimizer 할당
        self.lr = lr
        self.sft_epochs = sft_epochs
        self.sft_batch_size = sft_batch_size
        self.sft_grad_accum = sft_grad_accum
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def train_on_rollouts(
        self,
        prompts: List[str],
        responses: List[str],
        is_corrects: List[bool],
    ) -> Dict[str, float]:
        """
        Run SFT calibration training on a batch of GRPO rollouts.

        Args:
            prompts: List of original prompts
            responses: List of model-generated responses
            is_corrects: List of whether each response was correct

        Returns:
            Dict with training metrics
        """
        if len(prompts) == 0:
            return {"sft_loss": 0.0, "sft_samples": 0}

        dataset = CalibrationDataset(
            prompts=prompts,
            responses=responses,
            is_corrects=is_corrects,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )

        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0
        dataloader = DataLoader(
            dataset,
            batch_size=self.sft_batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_fn(b, pad_token_id=pad_id),
            drop_last=False,
        )

        self.model.train()
        total_loss = 0.0
        total_steps = 0
        total_correct_preds = 0
        total_preds = 0

        for epoch in range(self.sft_epochs):
            self.optimizer.zero_grad()
            accum_loss = 0.0

            for step, batch in enumerate(dataloader):
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                logits = outputs.logits

                # Shift for causal LM loss
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()

                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )

                # Compute prediction accuracy on the target token
                with torch.no_grad():
                    mask = shift_labels != -100
                    if mask.any():
                        pred_tokens = shift_logits[mask].argmax(dim=-1)
                        true_tokens = shift_labels[mask]
                        total_correct_preds += (pred_tokens == true_tokens).sum().item()
                        total_preds += mask.sum().item()

                scaled_loss = loss / self.sft_grad_accum
                scaled_loss.backward()
                accum_loss += loss.item()

                if (step + 1) % self.sft_grad_accum == 0 or (step + 1) == len(dataloader):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    total_loss += accum_loss / min(self.sft_grad_accum, step % self.sft_grad_accum + 1)
                    total_steps += 1
                    accum_loss = 0.0

        avg_loss = total_loss / max(total_steps, 1)
        pred_acc = total_correct_preds / max(total_preds, 1)

        metrics = {
            "sft_loss": avg_loss,
            "sft_pred_accuracy": pred_acc,
            "sft_samples": len(dataset),
            "sft_correct_ratio": sum(is_corrects) / max(len(is_corrects), 1),
        }

        logger.info(
            f"SFT Calibration: loss={avg_loss:.4f}, pred_acc={pred_acc:.4f}, "
            f"samples={len(dataset)}"
        )

        return metrics
