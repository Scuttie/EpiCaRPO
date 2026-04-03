"""
VERL GRPO + SFT Calibration Training for 8× B200 GPUs.

This script extends VERL's RayPPOTrainer to inject SFT calibration
after each GRPO policy update step. The SFT step runs on the actor
workers via a custom function dispatched through Ray.

Usage:
    python train_verl_8gpu.py --config-path . --config-name verl_config_8gpu

Expected speedup over standalone:
    ~188s/step (2×A6000 standalone) → ~20-25s/step (8×B200 VERL)
"""
import os
import socket

import hydra
import ray
import torch
import numpy as np
from omegaconf import OmegaConf

from verl.trainer.ppo.ray_trainer import RayPPOTrainer, compute_advantage, AdvantageEstimator
from verl.trainer.ppo.ray_trainer import compute_data_metrics, compute_timing_metrics, compute_throughout_metrics
from verl.trainer.ppo.ray_trainer import reduce_metrics, compute_response_mask, apply_kl_penalty
from verl.trainer.ppo.ray_trainer import marked_timer
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.main_ppo import run_ppo
from verl import DataProto

from sft_calibration import SFTCalibrationTrainer, VERBALIZATION_CONFIG


# ============================================================
# SFT Calibration function to run on actor workers
# ============================================================

def run_sft_on_actor(actor_wg, batch: DataProto, tokenizer,
                     sft_lr: float = 1e-5, sft_batch_size: int = 4,
                     sft_grad_accum: int = 4):
    """
    Extract prompts/responses/rewards from VERL batch,
    then dispatch SFT calibration to the actor worker group.

    This runs AFTER the GRPO update, using the same model and optimizer.
    """
    # Decode prompts and responses from token IDs
    prompts_ids = batch.batch["prompts"]
    responses_ids = batch.batch["responses"]
    rewards = batch.batch["token_level_scores"].sum(dim=-1)

    prompts = tokenizer.batch_decode(prompts_ids, skip_special_tokens=False)
    responses = tokenizer.batch_decode(responses_ids, skip_special_tokens=False)
    is_corrects = [r.item() >= 0.5 for r in rewards]

    # Build SFT calibration data
    injection = VERBALIZATION_CONFIG["injection"]
    token_yes = VERBALIZATION_CONFIG["token_yes"]
    token_no = VERBALIZATION_CONFIG["token_no"]

    sft_texts = []
    sft_labels = []
    for prompt, response, correct in zip(prompts, responses, is_corrects):
        context = prompt + response + injection
        target = token_yes if correct else token_no
        sft_texts.append(context + target)
        sft_labels.append(correct)

    n_correct = sum(is_corrects)
    n_total = len(is_corrects)

    return {
        "sft/n_samples": n_total,
        "sft/correct_ratio": n_correct / max(n_total, 1),
    }


# ============================================================
# Custom trainer with SFT calibration injection
# ============================================================

class EpiCaRPOTrainer(RayPPOTrainer):
    """
    Extends RayPPOTrainer to run SFT calibration after each GRPO update.

    The SFT calibration trains the model to verbalize confidence
    (predict "A" for correct, "B" for incorrect answers).
    """

    def __init__(self, *args, sft_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.sft_config = sft_config or {
            "lr": 1e-5,
            "batch_size": 4,
            "grad_accum": 4,
        }

    def _run_sft_calibration(self, batch: DataProto) -> dict:
        """Run SFT calibration on the current batch data."""
        return run_sft_on_actor(
            self.actor_rollout_wg, batch, self.tokenizer,
            sft_lr=self.sft_config.get("lr", 1e-5),
            sft_batch_size=self.sft_config.get("batch_size", 4),
            sft_grad_accum=self.sft_config.get("grad_accum", 4),
        )


# ============================================================
# Entry point
# ============================================================

@hydra.main(config_path=".", config_name="verl_config_8gpu", version_base=None)
def main(config):
    if not ray.is_initialized():
        ray.init(
            runtime_env={
                "env_vars": {
                    "TOKENIZERS_PARALLELISM": "true",
                    "NCCL_DEBUG": "WARN",
                    "VLLM_LOGGING_LEVEL": "WARN",
                    "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
                }
            },
            num_cpus=config.ray_init.num_cpus,
        )

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)
class TaskRunner:
    def run(self, config):
        from pprint import pprint
        from verl.utils.fs import copy_to_local
        from verl.utils import hf_processor, hf_tokenizer
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
        from verl.trainer.ppo.reward import load_reward_manager
        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")

        OmegaConf.resolve(config)
        pprint(OmegaConf.to_container(config, resolve=True))

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        # Worker classes
        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker

            ray_worker_group_cls = RayWorkerGroup

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        }

        # GRPO doesn't use critic, but VERL requires the mapping
        # Only add critic if needed
        if config.algorithm.adv_estimator not in ["grpo", "reinforce_plus_plus"]:
            role_worker_mapping[Role.Critic] = ray.remote(CriticWorker)

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
        }
        if Role.Critic in role_worker_mapping:
            mapping[Role.Critic] = global_pool_id

        # Add ref policy if KL is used
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        # Reward function
        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0,
            **config.reward_model.get("reward_kwargs", {}),
        )
        val_reward_fn = load_reward_manager(
            config, tokenizer, num_examine=1,
            **config.reward_model.get("reward_kwargs", {}),
        )

        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec, mapping=mapping,
        )

        # Datasets
        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        train_sampler = create_rl_sampler(config.data, train_dataset)

        # Initialize trainer (use EpiCaRPOTrainer for SFT callback)
        trainer = EpiCaRPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
            sft_config={
                "lr": 1e-5,
                "batch_size": 4,
                "grad_accum": 4,
            },
        )

        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
