# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint
from typing import Any, Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from tensordict import TensorDict

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_variance_proxy_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.config import FSDPEngineConfig
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

from sklearn.metrics import roc_auc_score, brier_score_loss

@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, using max_colocate_count=3: actor_critic_ref, rollout, reward model (optional)
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=3, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray._private.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    
    # [Modification] Extract data_source to pass to core_algos (identifies verification tasks)
    data_source = data.non_tensor_batch.get("data_source", None)

    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        # [Modification] Pass data_source
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            data_source=data_source, # <-- Passed
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
            "data_source": data_source, # <-- Passed
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]
        # Add sum_pi_squared for Optimal Token Baseline
        if adv_estimator in (AdvantageEstimator.OPTIMAL_TOKEN_BASELINE, AdvantageEstimator.TIR_OPTIMAL_TOKEN_BASELINE):
            # Check if sum_pi_squared is available
            assert "sum_pi_squared" in data.batch, (
                "Step-dependent optimal baseline requires sum_pi_squared from actor. "
                "Please set actor.calculate_sum_pi_squared=True in config."
            )
            adv_kwargs["sum_pi_squared"] = data.batch["sum_pi_squared"]
            # Get pre-computed rollout IS weights if available
            rollout_is_weights = data.batch.get("rollout_is_weights", None)
            adv_kwargs["rollout_is_weights"] = rollout_is_weights

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping or Role.ActorRolloutRef in role_worker_mapping, (
                f"{role_worker_mapping.keys()=}"
            )

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.config)
        # legacy reward model implementation
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_reward_loop = self.config.reward_model.use_reward_loop

        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self.use_prefix_grouper = self.config.actor_rollout_ref.actor.get("use_prefix_grouper", False)
        self.use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _get_batch_size(self, batch: "DataProto") -> int:
        """Safely get batch size as an integer."""
        batch_size = batch.batch.batch_size
        if hasattr(batch_size, '__getitem__'):
            batch_size = batch_size[0]
        if isinstance(batch_size, torch.Tensor):
            batch_size = batch_size.item()
        return int(batch_size)

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _compute_or_extract_reward(
        self,
        batch: DataProto,
        reward_fn=None,
        return_dict: bool = False,
        sum_reward: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]] | torch.Tensor | dict[str, Any]:
        """
        Compute or extract reward from batch.

        When use_reward_loop=True, rewards are already computed during generate_sequences
        and stored in rm_scores. This method directly extracts them instead of calling
        reward functions which would only perform format conversion.

        Args:
            batch: DataProto containing the batch data
            reward_fn: Reward function to use if rm_scores doesn't exist (for training/validation)
            return_dict: Whether to return dict format with reward_extra_info (for validation)
            sum_reward: Whether to sum reward tensor along last dimension (for REMAX baseline)

        Returns:
            If return_dict=True: dict with "reward_tensor" and "reward_extra_info"
            If return_dict=False and sum_reward=True: summed reward_tensor (1D tensor)
            If return_dict=False and sum_reward=False: reward_tensor (2D tensor)
        """
        # When rm_scores already exists, extract it directly (format conversion only)
        if "rm_scores" in batch.batch.keys():
            reward_tensor = batch.batch["rm_scores"]
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)

            if return_dict:
                # Extract reward_extra_info if available
                reward_extra_keys = batch.meta_info.get("reward_extra_keys", [])
                reward_extra_info = (
                    {key: batch.non_tensor_batch[key] for key in reward_extra_keys} if reward_extra_keys else {}
                )
                return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
            else:
                # If sum_reward=True, only return tensor (for REMAX baseline)
                if sum_reward:
                    return reward_tensor
                # Otherwise, return tuple with reward_extra_info (for training loop)
                reward_extra_keys = batch.meta_info.get("reward_extra_keys", [])
                reward_extra_infos_dict = (
                    {key: batch.non_tensor_batch[key] for key in reward_extra_keys} if reward_extra_keys else {}
                )
                return reward_tensor, reward_extra_infos_dict

        # Otherwise, compute reward using reward_fn
        if reward_fn is None:
            raise ValueError("reward_fn must be provided when rm_scores is not available.")

        if return_dict:
            result = reward_fn(batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)
            reward_extra_info = result.get("reward_extra_info", {})
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        else:
            reward_tensor, reward_extra_infos_dict = compute_reward(batch, reward_fn)
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)
            return reward_tensor, reward_extra_infos_dict

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = []
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _compute_calibration_metrics(self, confidences: np.ndarray, labels: np.ndarray) -> dict:
        """Compute calibration metrics: AUROC, ECE, Brier Score.
        
        Args:
            confidences: Model confidence scores (P(correct)), shape (N,)
            labels: Binary labels (1=correct, 0=incorrect), shape (N,)
        
        Returns:
            Dictionary with calibration metrics
        """
        metrics = {}
        
        # Filter out NaN values
        valid_mask = ~(np.isnan(confidences) | np.isnan(labels))
        confidences = confidences[valid_mask]
        labels = labels[valid_mask]
        
        if len(confidences) == 0:
            return {"val-calibration/auroc": float('nan'), 
                    "val-calibration/ece": float('nan'), 
                    "val-calibration/brier": float('nan')}
        
        # AUROC
        try:
            if len(np.unique(labels)) > 1:
                auroc = roc_auc_score(labels, confidences)
            else:
                auroc = float('nan')
        except Exception:
            auroc = float('nan')
        metrics["val-calibration/auroc"] = auroc
        
        # Brier Score
        try:
            brier = brier_score_loss(labels, confidences)
        except Exception:
            brier = float('nan')
        metrics["val-calibration/brier"] = brier
        
        # ECE (Expected Calibration Error)
        try:
            n_bins = 10
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            ece = 0.0
            for i in range(n_bins):
                bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
                if i == 0:
                    in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
                else:
                    in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
                prop_in_bin = np.mean(in_bin)
                if prop_in_bin > 0:
                    acc_in_bin = np.mean(labels[in_bin])
                    avg_conf_in_bin = np.mean(confidences[in_bin])
                    ece += np.abs(avg_conf_in_bin - acc_in_bin) * prop_in_bin
        except Exception:
            ece = float('nan')
        metrics["val-calibration/ece"] = ece
        
        return metrics


    def _compute_verification_confidence(self, test_batch: DataProto) -> tuple[np.ndarray, np.ndarray]:
        """Compute self-verification confidence for each sample in the batch.
        
        Uses P(Yes) / (P(Yes) + P(No)) formula where the model is asked
        "Is the answer correct? A) Yes B) No"
        
        Args:
            test_batch: DataProto containing prompts, responses, and correctness labels
        
        Returns:
            confidences: Array of confidence scores
            labels: Array of binary correctness labels
        """
        VERBALIZATION_CONFIG = {
            "injection": "\nIs the answer correct? Choose ONLY one letter. A) Yes B) No. Your choice:",
            "token_yes": " A",  # token_id: 362
            "token_no": " B",   # token_id: 425
        }
        
        # Get token IDs
        token_yes_id = self.tokenizer(VERBALIZATION_CONFIG["token_yes"], add_special_tokens=False).input_ids[0]
        token_no_id = self.tokenizer(VERBALIZATION_CONFIG["token_no"], add_special_tokens=False).input_ids[0]
        
        batch_size = test_batch.batch["prompts"].shape[0]
        prompts = test_batch.batch["prompts"]
        responses = test_batch.batch["responses"]
        
        # Get correctness labels from reward scores
        if "token_level_scores" in test_batch.batch.keys():
            scores = test_batch.batch["token_level_scores"].sum(dim=-1).cpu().numpy()
        else:
            # If no scores available, return empty
            return np.array([]), np.array([])
        
        threshold = 0.5
        labels = (scores >= threshold).astype(float)
        
        # Build verification prompts
        verification_input_ids_list = []
        verification_prompts_list = []
        verification_responses_yes_list = []
        verification_responses_no_list = []
        verification_attn_mask_list = []
        
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        
        for i in range(batch_size):
            # Decode question and solution
            q_text = self.tokenizer.decode(prompts[i], skip_special_tokens=True)
            s_text = self.tokenizer.decode(responses[i], skip_special_tokens=True)
            
            # Build verification prompt
            user_content = (
                f"Question: {q_text}\n\n"
                f"Solution: {s_text}\n\n"
                f"{VERBALIZATION_CONFIG['injection']}"
            )
            
            # Apply chat template if available
            if hasattr(self.tokenizer, "apply_chat_template"):
                messages = [{"role": "user", "content": user_content}]
                full_prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                full_prompt = f"User: {user_content}\nAssistant:"
            
            # Tokenize
            prompt_ids = self.tokenizer(full_prompt, add_special_tokens=False).input_ids
            
            # Truncate if too long
            max_len = self.config.data.get("max_prompt_length", 4096) - 1
            if len(prompt_ids) > max_len:
                prompt_ids = prompt_ids[-max_len:]
            
            verification_prompts_list.append(torch.tensor(prompt_ids, dtype=torch.long))
            verification_responses_yes_list.append(torch.tensor([token_yes_id], dtype=torch.long))
            verification_responses_no_list.append(torch.tensor([token_no_id], dtype=torch.long))
        
        if len(verification_prompts_list) == 0:
            return np.array([]), np.array([])
        
        # Pad sequences for batch processing
        from torch.nn.utils.rnn import pad_sequence
        
        # Pad prompts (left padding for causal LM)
        max_prompt_len = max(len(p) for p in verification_prompts_list)
        padded_prompts = []
        for p in verification_prompts_list:
            pad_len = max_prompt_len - len(p)
            padded_prompts.append(torch.cat([torch.full((pad_len,), pad_token_id, dtype=torch.long), p]))
        padded_prompts = torch.stack(padded_prompts)
        
        # Create input_ids for " A" case: prompt + " A"
        responses_yes = torch.stack([torch.tensor([token_yes_id], dtype=torch.long) for _ in range(batch_size)])
        input_ids_yes = torch.cat([padded_prompts, responses_yes], dim=1)
        attn_mask_yes = (input_ids_yes != pad_token_id).long()
        position_ids_yes = torch.cumsum(attn_mask_yes, dim=1) - 1
        position_ids_yes.masked_fill_(attn_mask_yes == 0, 0)
        
        # Create input_ids for " B" case: prompt + " B"
        responses_no = torch.stack([torch.tensor([token_no_id], dtype=torch.long) for _ in range(batch_size)])
        input_ids_no = torch.cat([padded_prompts, responses_no], dim=1)
        attn_mask_no = (input_ids_no != pad_token_id).long()
        position_ids_no = torch.cumsum(attn_mask_no, dim=1) - 1
        position_ids_no.masked_fill_(attn_mask_no == 0, 0)
        
        # Create DataProto for " A" case
        from tensordict import TensorDict
        batch_yes = TensorDict({
            "input_ids": input_ids_yes,
            "attention_mask": attn_mask_yes,
            "position_ids": position_ids_yes,
            "responses": responses_yes,
        }, batch_size=[batch_size])
        
        data_yes = DataProto(
            batch=batch_yes,
            non_tensor_batch={"multi_modal_inputs": np.array([{}] * batch_size, dtype=object)},
        )
        data_yes.meta_info = {
            "micro_batch_size": self.config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu or 8,
            "temperature": 1.0,  # Use temperature=1 for proper probability
            "use_dynamic_bsz": False,
            "pad_token_id": pad_token_id,
        }
        
        # Create DataProto for " B" case
        batch_no = TensorDict({
            "input_ids": input_ids_no,
            "attention_mask": attn_mask_no,
            "position_ids": position_ids_no,
            "responses": responses_no,
        }, batch_size=[batch_size])
        
        data_no = DataProto(
            batch=batch_no,
            non_tensor_batch={"multi_modal_inputs": np.array([{}] * batch_size, dtype=object)},
        )
        data_no.meta_info = {
            "micro_batch_size": self.config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu or 8,
            "temperature": 1.0,
            "use_dynamic_bsz": False,
            "pad_token_id": pad_token_id,
        }
        
        # Compute log probabilities using actor
        try:
            # Pad to divisor for distributed processing
            size_divisor = self.actor_rollout_wg.world_size
            
            data_yes_padded, pad_size_yes = pad_dataproto_to_divisor(data_yes, size_divisor)
            data_no_padded, pad_size_no = pad_dataproto_to_divisor(data_no, size_divisor)
            
            # Compute log_prob for " A"
            output_yes = self.actor_rollout_wg.compute_log_prob(data_yes_padded)
            output_yes = unpad_dataproto(output_yes, pad_size=pad_size_yes)
            log_prob_yes = output_yes.batch["old_log_probs"][:, -1].cpu().numpy()  # Last token log prob
            
            # Compute log_prob for " B"
            output_no = self.actor_rollout_wg.compute_log_prob(data_no_padded)
            output_no = unpad_dataproto(output_no, pad_size=pad_size_no)
            log_prob_no = output_no.batch["old_log_probs"][:, -1].cpu().numpy()
            
            # Compute confidence: P(A) / (P(A) + P(B))
            # Using log-sum-exp for numerical stability
            log_probs = np.stack([log_prob_yes, log_prob_no], axis=1)  # (batch_size, 2)
            max_log_prob = np.max(log_probs, axis=1, keepdims=True)
            exp_log_probs = np.exp(log_probs - max_log_prob)
            probs = exp_log_probs / np.sum(exp_log_probs, axis=1, keepdims=True)
            confidences = probs[:, 0]  # P(A) = P(Yes)
            
            return confidences, labels
            
        except Exception as e:
            print(f"Error computing verification confidence: {e}")
            import traceback
            traceback.print_exc()
            return np.array([]), np.array([])

    def _validate(self, merged: bool = False):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []
        
        # ★★★ 수정: data_source별로 calibration metrics 수집 ★★★
        calibration_data_by_source: dict[str, dict] = defaultdict(lambda: {"confidences": [], "labels": []})

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # Store original inputs
            input_ids = test_batch.batch["prompts"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            # evaluate using reward_function
            result = self._compute_or_extract_reward(test_batch, reward_fn=self.val_reward_fn, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)
            
            # token_level_scores를 test_batch에 저장 (verification confidence 계산용)
            test_batch.batch["token_level_scores"] = reward_tensor

            reward_extra_infos_dict["reward"].extend(scores)
            reward_extra_info = result.get("reward_extra_info", {})
            for key, values in reward_extra_info.items():
                if key not in reward_extra_infos_dict:
                    reward_extra_infos_dict[key] = []
                if isinstance(values, np.ndarray):
                    reward_extra_infos_dict[key].extend(values.tolist())
                else:
                    reward_extra_infos_dict[key].extend(values if isinstance(values, list) else [values])

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            # ★★★ data_source 추출 ★★★
            batch_data_sources = test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0])
            data_source_lst.append(batch_data_sources)
            
            # ★★★ Verification Confidence 계산 (data_source별로 저장) ★★★
            if self.config.trainer.get("compute_calibration_metrics", True):
                try:
                    confidences, labels = self._compute_verification_confidence(test_batch)
                    if len(confidences) > 0:
                        # data_source별로 분리하여 저장
                        for i, (conf, label) in enumerate(zip(confidences, labels)):
                            ds = batch_data_sources[i] if i < len(batch_data_sources) else "unknown"
                            calibration_data_by_source[ds]["confidences"].append(conf)
                            calibration_data_by_source[ds]["labels"].append(label)
                except Exception as e:
                    print(f"Warning: Failed to compute verification confidence: {e}")

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            # score/pred는 calibration용이므로 검증 스킵
            if key_info in ["score", "pred"]:
                continue
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        if merged:
            print("_merge_validation_results validate result will be merged")
            return {
                "data_sources": data_source_lst,
                "sample_uids": sample_uids,
                "sample_turns": sample_turns,
                "reward_extra_infos_dict": reward_extra_infos_dict,
            }
        
        data_sources = np.concatenate(data_source_lst, axis=0)
        reward_extra_infos_dict.pop("score", None)
        reward_extra_infos_dict.pop("pred", None)

        metric_dict = self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)
        
        # ★★★ Calibration metrics 계산 (data_source별로) ★★★
        if self.config.trainer.get("compute_calibration_metrics", True):
            all_confidences = []
            all_labels = []
            
            for data_source, data in calibration_data_by_source.items():
                if len(data["confidences"]) > 0:
                    confidences = np.array(data["confidences"])
                    labels = np.array(data["labels"])
                    
                    # Aggregate for overall metrics
                    all_confidences.extend(data["confidences"])
                    all_labels.extend(data["labels"])
                    
                    # Per-dataset calibration metrics
                    ds_metrics = self._compute_calibration_metrics(confidences, labels)
                    for metric_name, metric_val in ds_metrics.items():
                        # e.g., val-calibration/math/auroc, val-calibration/amc23/ece
                        new_key = metric_name.replace("val-calibration/", f"val-calibration/{data_source}/")
                        metric_dict[new_key] = metric_val
                    
                    print(f"Calibration [{data_source}]: AUROC={ds_metrics.get('val-calibration/auroc', 'N/A'):.4f}, "
                        f"ECE={ds_metrics.get('val-calibration/ece', 'N/A'):.4f}, "
                        f"Brier={ds_metrics.get('val-calibration/brier', 'N/A'):.4f}")
            
            # Overall (aggregated) calibration metrics
            if len(all_confidences) > 0:
                overall_metrics = self._compute_calibration_metrics(
                    np.array(all_confidences), 
                    np.array(all_labels)
                )
                for metric_name, metric_val in overall_metrics.items():
                    new_key = metric_name.replace("val-calibration/", "val-calibration/overall/")
                    metric_dict[new_key] = metric_val
                
                print(f"Calibration [overall]: AUROC={overall_metrics.get('val-calibration/auroc', 'N/A'):.4f}, "
                    f"ECE={overall_metrics.get('val-calibration/ece', 'N/A'):.4f}, "
                    f"Brier={overall_metrics.get('val-calibration/brier', 'N/A'):.4f}")
        
        return metric_dict

    def _val_metrics_update(self, data_sources, sample_uids, reward_extra_infos_dict, sample_turns):
        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def _merge_validation_results(self, result_a, result_b):
        if result_a is None and result_b is None:
            return {}
        if result_a is None:
            result_a = {"data_sources": [], "sample_uids": [], "sample_turns": [], "reward_extra_infos_dict": {}}
        if result_b is None:
            result_b = {"data_sources": [], "sample_uids": [], "sample_turns": [], "reward_extra_infos_dict": {}}

        if not result_a.get("data_sources") and not result_b.get("data_sources"):
            return {}

        data_sources = np.concatenate(result_a["data_sources"] + result_b["data_sources"], axis=0)
        sample_uids = result_a["sample_uids"] + result_b["sample_uids"]
        sample_turns = result_a["sample_turns"] + result_b["sample_turns"]

        reward_extra_infos_dict = {}
        all_keys = set(result_a["reward_extra_infos_dict"].keys()) | set(result_b["reward_extra_infos_dict"].keys())
        for key in all_keys:
            list_a = result_a["reward_extra_infos_dict"].get(key, [])
            list_b = result_b["reward_extra_infos_dict"].get(key, [])
            reward_extra_infos_dict[key] = list_a + list_b

        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    def init_workers(self):
        """Initialize distributed training workers using Ray backend."""
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # 1. Actor/Rollout 워커 설정
        # use_reference_policy가 True이고 LoRA가 아니면 role을 "actor_rollout_ref"로 설정해야 함
        if Role.ActorRolloutRef in self.role_worker_mapping:
            actor_role = Role.ActorRolloutRef
        elif Role.ActorRollout in self.role_worker_mapping:
            actor_role = Role.ActorRollout
        else:
            raise ValueError("Neither ActorRolloutRef nor ActorRollout in role_worker_mapping")
        
        # ★★★ 핵심 수정: use_reference_policy && !ref_in_actor 일 때 role 결정 ★★★
        if self.use_reference_policy and not self.ref_in_actor:
            # 별도 ref 워커가 필요한 경우 -> actor_rollout_ref role 사용
            worker_role_str = "actor_rollout_ref"
        else:
            worker_role_str = "actor_rollout"
        
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[actor_role],
                config=self.config.actor_rollout_ref,
                role=worker_role_str,  # ★ 동적으로 결정된 role
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # 2. Critic 워커 설정
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], 
                config=self.config.critic,
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # 3. Reference Policy 워커 설정은 불필요 (actor_rollout_ref가 ref도 처리)
        # use_legacy_worker_impl="auto"에서는 ActorRolloutRefWorker가 ref 역할도 수행

        # 4. Ray Worker Group 생성
        all_wg = {}
        wg_kwargs = {"device_name": self.device_name}
        
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool, 
                ray_cls_with_init=worker_dict_cls, 
                **wg_kwargs
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        # 5. 각 워커 할당 및 모델 초기화
        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

        # ★★★ use_reference_policy=True && !ref_in_actor 일 때, 
        # actor_rollout_wg가 ref 역할도 수행하므로 ref_policy_wg = actor_rollout_wg ★★★
        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = self.actor_rollout_wg

        # 6. Async Rollout Manager 설정
        self.async_rollout_mode = False
        agent_config = self.config.actor_rollout_ref.rollout.get("agent", None)
        if agent_config is not None and agent_config.get("enable", False):
            self.async_rollout_mode = True
            
        manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        if manager_class_fqn:
            AgentLoopManager = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
        else:
            from verl.experimental.agent_loop import AgentLoopManager

        rm_rp = self.resource_pool_manager.get_resource_pool(Role.RewardModel) if (
            self.config.reward_model.enable and self.config.reward_model.enable_resource_pool
        ) else None
        self.async_rollout_manager = AgentLoopManager(
            config=self.config, 
            worker_group=self.actor_rollout_wg, 
            rm_resource_pool=rm_rp
        )
        
    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        if (
            hasattr(self.config.actor_rollout_ref.actor.checkpoint, "async_save")
            and self.config.actor_rollout_ref.actor.checkpoint.async_save
        ) or (
            "async_save" in self.config.actor_rollout_ref.actor.checkpoint
            and self.config.actor_rollout_ref.actor.checkpoint["async_save"]
        ):
            print("skip write latest_checkpointed_iteration.txt when async_save is True")
            return
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm and not self.use_reward_loop:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm and not self.use_reward_loop:
                self.rm_wg.stop_profile()

    def _get_dp_size(self, worker_group, role: str) -> int:
        """Get data parallel size from worker group dispatch info.

        This method retrieves the data parallel size by querying the dispatch info
        for the specified role. The dispatch info is cached for subsequent calls.

        Args:
            worker_group: The worker group to query dispatch info from.
            role: The role name (e.g., "actor", "critic") to get DP size for.

        Returns:
            The data parallel size (number of DP ranks).
        """
        if role not in worker_group._dispatch_info:
            dp_rank_mapping = worker_group._query_dispatch_info(role)
            worker_group._dispatch_info[role] = dp_rank_mapping
        else:
            dp_rank_mapping = worker_group._dispatch_info[role]
        return max(dp_rank_mapping) + 1

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens.

        When use_prefix_grouper is enabled, uses group-level balancing to keep samples with
        the same uid together on the same rank for prefix sharing optimization.
        """
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)  # (train_batch_size,)
        workload_lst = calculate_workload(global_seqlen_lst)
        # Get dp_size from dispatch info to correctly balance across data parallel ranks
        # Note: world_size may include tensor/pipeline parallel dimensions, but we only want DP
        dp_size = self._get_dp_size(self.actor_rollout_wg, "actor")

        # Use group-level balancing for PrefixGrouper to keep same-uid samples together
        if getattr(self, "use_prefix_grouper", False) and "uid" in batch.non_tensor_batch:
            from verl.utils.seqlen_balancing import get_group_balanced_partitions

            uid_list = list(batch.non_tensor_batch["uid"])
            seqlen_list = global_seqlen_lst.tolist()

            # Count number of uid groups
            num_groups = len(set(uid_list))

            if num_groups % dp_size != 0:
                raise ValueError(
                    f"PrefixGrouper with balance_batch requires num_uid_groups ({num_groups}) "
                    f"% dp_size ({dp_size}) == 0. "
                    f"This ensures each rank gets equal number of groups. "
                    f"Current batch_size={batch_size}, adjust batch_size to be a multiple of "
                    f"dp_size * rollout.n."
                )

            global_partition_lst = get_group_balanced_partitions(
                seqlen_list=seqlen_list,
                uid_list=uid_list,
                k_partitions=dp_size,
            )

        elif keep_minibatch:
            # Decouple the DP balancing and mini-batching.
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(workload_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(dp_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    workload_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=dp_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(workload_lst, k_partitions=dp_size, equal_size=True)
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        # Skip reordering within partitions for PrefixGrouper to maintain uid grouping
        if not getattr(self, "use_prefix_grouper", False):
            for idx, partition in enumerate(global_partition_lst):
                partition.sort(key=lambda x: (workload_lst[x], x))
                ordered_partition = partition[::2] + partition[1::2][::-1]
                global_partition_lst[idx] = ordered_partition

        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst.tolist(), partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _compute_values(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            tu.assign_non_tensor(batch_td, compute_loss=False)
            output = self.critic_wg.infer_batch(batch_td)
            output = output.get()
            values = tu.get(output, "values")
            values = no_padding_2_padding(values, batch_td)
            values = tu.get_tensordict({"values": values.float()})
            values = DataProto.from_tensordict(values)
        else:
            values = self.critic_wg.compute_values(batch)
        return values

    def _compute_ref_log_prob(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            # step 1: convert dataproto to tensordict.
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            metadata = {"calculate_entropy": False, "compute_loss": False}
            if self.ref_in_actor:
                metadata["no_lora_adapter"] = True
            tu.assign_non_tensor(batch_td, **metadata)
            if self.ref_in_actor:
                output = self.actor_rollout_wg.compute_log_prob(batch_td)
            else:
                output = self.ref_policy_wg.compute_ref_log_prob(batch_td)
            # gather output
            log_probs = tu.get(output, "log_probs")
            # step 4. No padding to padding
            log_probs = no_padding_2_padding(log_probs, batch_td)
            # step 5: rebuild a tensordict and convert to dataproto
            ref_log_prob = tu.get_tensordict({"ref_log_prob": log_probs.float()})
            ref_log_prob = DataProto.from_tensordict(ref_log_prob)
        else:
            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)

        return ref_log_prob

    def _compute_old_log_prob(self, batch: DataProto):
        if self.use_legacy_worker_impl == "disable":
            # TODO: remove step 1, 2, 4 after we make the whole training tensordict and padding free
            # step 1: convert dataproto to tensordict.
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            tu.assign_non_tensor(batch_td, calculate_entropy=True, compute_loss=False)
            output = self.actor_rollout_wg.compute_log_prob(batch_td)
            # gather output
            entropy = tu.get(output, "entropy")
            log_probs = tu.get(output, "log_probs")
            old_log_prob_mfu = tu.get(output, "metrics")["mfu"]
            # step 4. No padding to padding
            entropy = no_padding_2_padding(entropy, batch_td)
            log_probs = no_padding_2_padding(log_probs, batch_td)
            # step 5: rebuild a tensordict and convert to dataproto
            old_log_prob = tu.get_tensordict({"old_log_probs": log_probs.float(), "entropys": entropy.float()})
            old_log_prob = DataProto.from_tensordict(old_log_prob)
        else:
            print(f"[DEBUG _compute_old_log_prob] Calling actor_rollout_wg.compute_log_prob")
            print(f"[DEBUG _compute_old_log_prob] batch size: {batch.batch.batch_size}")
            print(f"[DEBUG _compute_old_log_prob] batch keys: {list(batch.batch.keys())}")
            
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            
            print(f"[DEBUG _compute_old_log_prob] old_log_prob: {old_log_prob}")
            print(f"[DEBUG _compute_old_log_prob] old_log_prob type: {type(old_log_prob)}")
            
            old_log_prob_mfu = 0
        return old_log_prob, old_log_prob_mfu

    def _update_actor(self, batch: DataProto) -> DataProto:
        rollout_config = self.config.actor_rollout_ref.rollout
        batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
        # TODO: Make "temperature" single source of truth from generation.
        batch.meta_info["temperature"] = rollout_config.temperature
        # update actor
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to no-padding
            batch_td = left_right_2_no_padding(batch_td)
            calculate_entropy = self.config.actor_rollout_ref.actor.entropy_coeff != 0.0
            ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
            ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            ppo_epochs = self.config.actor_rollout_ref.actor.ppo_epochs
            seed = self.config.actor_rollout_ref.actor.data_loader_seed
            shuffle = self.config.actor_rollout_ref.actor.shuffle
            tu.assign_non_tensor(
                batch_td,
                calculate_entropy=calculate_entropy,
                global_batch_size=ppo_mini_batch_size,
                mini_batch_size=ppo_mini_batch_size,
                epochs=ppo_epochs,
                seed=seed,
                dataloader_kwargs={"shuffle": shuffle},
            )

            actor_output = self.actor_rollout_wg.update_actor(batch_td)
            actor_output = tu.get(actor_output, "metrics")
            actor_output = rename_dict(actor_output, "actor/")
            # modify key name
            actor_output["perf/mfu/actor"] = actor_output.pop("actor/mfu")
            actor_output = DataProto.from_single_dict(data={}, meta_info={"metrics": actor_output})
        else:
            actor_output = self.actor_rollout_wg.update_actor(batch)
        return actor_output

    def _update_critic(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to no-padding
            batch_td = left_right_2_no_padding(batch_td)
            ppo_mini_batch_size = self.config.critic.ppo_mini_batch_size
            ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            ppo_epochs = self.config.critic.ppo_epochs
            seed = self.config.critic.data_loader_seed
            shuffle = self.config.critic.shuffle
            tu.assign_non_tensor(
                batch_td,
                global_batch_size=ppo_mini_batch_size,
                mini_batch_size=ppo_mini_batch_size,
                epochs=ppo_epochs,
                seed=seed,
                dataloader_kwargs={"shuffle": shuffle},
            )

            output = self.critic_wg.train_mini_batch(batch_td)
            output = output.get()
            output = tu.get(output, "metrics")
            output = rename_dict(output, "critic/")
            # modify key name
            output["perf/mfu/critic"] = output.pop("critic/mfu")
            critic_output = DataProto.from_single_dict(data={}, meta_info={"metrics": output})
        else:
            critic_output = self.critic_wg.update_critic(batch)
        return critic_output

    def _pad_and_concat(self, batch1: DataProto, batch2: DataProto) -> DataProto:
        """Concatenates two DataProtos with padding for different sequence lengths."""
        pad_token_id = self.tokenizer.pad_token_id or 0

        def pad_tensor(t, target_len, pad_val):
            if t is None or not isinstance(t, torch.Tensor):
                return t
            if t.dim() < 2:
                return t
            curr_len = t.shape[1]
            diff = target_len - curr_len
            if diff <= 0:
                return t
            
            if t.dim() == 2:
                return torch.nn.functional.pad(t, (0, diff), value=pad_val)
            elif t.dim() == 3:
                return torch.nn.functional.pad(t, (0, 0, 0, diff), value=pad_val)
            return t

        batch1_size = self._get_batch_size(batch1)
        batch2_size = self._get_batch_size(batch2)
        
        def get_seq_len(batch, key):
            if key in batch.batch.keys():
                t = batch.batch[key]
                if isinstance(t, torch.Tensor) and t.dim() >= 2:
                    return t.shape[1]
            return 0
        
        # ★★★ 핵심: response 관련 키와 total sequence 키 분리 ★★★
        response_keys = ['responses', 'response_mask', 'token_level_scores', 
                         'token_level_rewards', 'old_log_probs', 'ref_log_prob', 
                         'advantages', 'returns']
        total_seq_keys = ['input_ids', 'attention_mask', 'position_ids']
        prompt_keys = ['prompts']
        
        # 각 카테고리별 최대 길이 계산
        max_response_len = 0
        for key in response_keys:
            max_response_len = max(max_response_len, get_seq_len(batch1, key), get_seq_len(batch2, key))
        
        max_total_len = 0
        for key in total_seq_keys:
            max_total_len = max(max_total_len, get_seq_len(batch1, key), get_seq_len(batch2, key))
        
        max_prompt_len = 0
        for key in prompt_keys:
            max_prompt_len = max(max_prompt_len, get_seq_len(batch1, key), get_seq_len(batch2, key))
        
        keys = set(batch1.batch.keys()) | set(batch2.batch.keys())

        for k in keys:
            # 패딩 값 결정
            if k in ['input_ids', 'responses', 'prompts']:
                pad_val = pad_token_id
            elif k in ['attention_mask', 'response_mask']:
                pad_val = 0
            elif k in ['token_level_scores', 'token_level_rewards', 'advantages', 
                       'returns', 'old_log_probs', 'ref_log_prob']:
                pad_val = 0.0
            else:
                pad_val = 0
            
            # target 길이 결정
            if k in response_keys:
                target_len = max_response_len
            elif k in total_seq_keys:
                target_len = max_total_len
            elif k in prompt_keys:
                target_len = max_prompt_len
            else:
                target_len = max(get_seq_len(batch1, k), get_seq_len(batch2, k))
                if target_len == 0:
                    continue
            
            # batch1 처리
            if k not in batch1.batch.keys():
                ref = batch2.batch.get(k)
                if ref is not None and isinstance(ref, torch.Tensor):
                    if ref.dim() >= 2:
                        shape = (batch1_size, target_len) + ref.shape[2:]
                    else:
                        shape = (batch1_size,) + ref.shape[1:]
                    batch1.batch[k] = torch.full(shape, pad_val, dtype=ref.dtype, device=ref.device)
            else:
                batch1.batch[k] = pad_tensor(batch1.batch[k], target_len, pad_val)
            
            # batch2 처리
            if k not in batch2.batch.keys():
                ref = batch1.batch.get(k)
                if ref is not None and isinstance(ref, torch.Tensor):
                    if ref.dim() >= 2:
                        shape = (batch2_size, target_len) + ref.shape[2:]
                    else:
                        shape = (batch2_size,) + ref.shape[1:]
                    batch2.batch[k] = torch.full(shape, pad_val, dtype=ref.dtype, device=ref.device)
            else:
                batch2.batch[k] = pad_tensor(batch2.batch[k], target_len, pad_val)

        # Sync non-tensor batch keys with proper default values
        for k in set(batch1.non_tensor_batch.keys()) - set(batch2.non_tensor_batch.keys()):
            ref = batch1.non_tensor_batch[k]
            if k == 'num_turns':
                # num_turns는 1로 기본값 설정
                batch2.non_tensor_batch[k] = np.array([1] * batch2_size, dtype=ref.dtype if hasattr(ref, 'dtype') else np.int64)
            else:
                batch2.non_tensor_batch[k] = np.array([None] * batch2_size, dtype=object)
                
        for k in set(batch2.non_tensor_batch.keys()) - set(batch1.non_tensor_batch.keys()):
            ref = batch2.non_tensor_batch[k]
            if k == 'num_turns':
                batch1.non_tensor_batch[k] = np.array([1] * batch1_size, dtype=ref.dtype if hasattr(ref, 'dtype') else np.int64)
            else:
                batch1.non_tensor_batch[k] = np.array([None] * batch1_size, dtype=object)
                    
        return DataProto.concat([batch1, batch2])
                    
    def _generate_verification_batch(self, batch: DataProto) -> Optional[DataProto]:
        """
        Generates 'Is it correct?' verification data based on the rollout results.
        Format: Question + Solution + 'Is the answer correct?' -> Yes/No
        """
        prompts = batch.batch['prompts']
        responses = batch.batch['responses']

        n_samples = batch.batch['input_ids'].shape[0]
        
        # Assume Outcome Reward: sum of token_level_scores indicates correctness
        if "token_level_scores" not in batch.batch.keys():
            return None
        
        scores = batch.batch["token_level_scores"].sum(dim=-1).detach().cpu().numpy()
        
        new_input_ids = []
        new_prompts_ids = []
        new_attn_masks = []
        new_resp_masks = []
        new_responses_ids = []
        
        found_count = 0
        max_len = self.config.data.get("max_prompt_length", 4096)
        threshold = 0.5 # Treat as correct if reward >= 0.5

        for i in range(len(prompts)):
            q_text = self.tokenizer.decode(prompts[i], skip_special_tokens=True)
            s_text = self.tokenizer.decode(responses[i], skip_special_tokens=True)
            is_correct = scores[i] >= threshold

            # Construct Verification Prompt
            user_content = (
                f"Question: {q_text}\n\n"
                f"Solution: {s_text}\n\n"
                f"Is the answer correct? (a) Yes (b) No"
            )
            
            target_text = " (a) Yes" if is_correct else " (b) No"

            # Apply Chat Template if available
            if hasattr(self.tokenizer, "apply_chat_template"):
                messages = [{"role": "user", "content": user_content}]
                full_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                # Fallback for Base models or simple tokenizers
                full_prompt = f"User: {user_content}\nAssistant:"

            # Tokenize
            p_ids = self.tokenizer(full_prompt, add_special_tokens=False).input_ids
            t_ids = self.tokenizer(target_text, add_special_tokens=False).input_ids
            
            # Truncate prompt from the front if too long
            total_len = len(p_ids) + len(t_ids)
            if total_len > max_len:
                p_ids = p_ids[-(max_len - len(t_ids)):]
            
            p_tensor = torch.tensor(p_ids, dtype=torch.long)
            t_tensor = torch.tensor(t_ids, dtype=torch.long)
            full_tensor = torch.cat([p_tensor, t_tensor])
            
            # Create Masks
            attn_mask = torch.ones_like(full_tensor)
            resp_mask = torch.zeros_like(full_tensor)
            resp_mask[len(p_ids):] = 1  # Mask only the target response (Yes/No)

            new_input_ids.append(full_tensor)
            new_prompts_ids.append(p_tensor)
            new_responses_ids.append(t_tensor) # Just the target tokens
            new_attn_masks.append(attn_mask)
            new_resp_masks.append(resp_mask)
            found_count += 1

        if found_count == 0:
            return None

        # Pad sequences
        device = batch.batch['input_ids'].device
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        
        padded_input_ids = pad_sequence(new_input_ids, batch_first=True, padding_value=pad_id).to(device)
        padded_prompts = pad_sequence(new_prompts_ids, batch_first=True, padding_value=pad_id).to(device)
        # For responses key, align with new_responses_ids (the targets)
        padded_responses = pad_sequence(new_responses_ids, batch_first=True, padding_value=pad_id).to(device)
        
        padded_attn = pad_sequence(new_attn_masks, batch_first=True, padding_value=0).to(device)
        padded_resp_mask = pad_sequence(new_resp_masks, batch_first=True, padding_value=0).to(device)

        # Recalculate Position IDs
        position_ids = torch.cumsum(padded_attn, dim=1) - 1
        position_ids.masked_fill_(padded_attn == 0, 0)

        # Assign fixed reward 1.0 to the correct token positions for SFT-like training
        token_level_scores = padded_resp_mask.float() * 1.0

        batch_dict = TensorDict({
            "input_ids": padded_input_ids,
            "attention_mask": padded_attn,
            "position_ids": position_ids,
            "prompts": padded_prompts,
            "responses": padded_responses,
            "response_mask": padded_resp_mask,
            "token_level_scores": token_level_scores
        }, batch_size=[n_samples])

        # non_tensor_batch 생성
        non_tensor_batch = {
            'data_source': np.array(['verification'] * n_samples, dtype=object),
        }
        
        # 원본 batch의 모든 non_tensor_batch 키에 대해 기본값 설정
        for key in batch.non_tensor_batch.keys():
            if key == 'data_source':
                continue
            elif key == 'num_turns':
                non_tensor_batch[key] = np.array([1] * n_samples, dtype=np.int64)
            elif key in ['uid', 'index', 'extra_info']:
                non_tensor_batch[key] = np.array([f'verif_{i}' for i in range(n_samples)], dtype=object)
            else:
                ref = batch.non_tensor_batch[key]
                if hasattr(ref, 'dtype'):
                    if np.issubdtype(ref.dtype, np.integer):
                        non_tensor_batch[key] = np.zeros(n_samples, dtype=ref.dtype)
                    elif np.issubdtype(ref.dtype, np.floating):
                        non_tensor_batch[key] = np.zeros(n_samples, dtype=ref.dtype)
                    else:
                        non_tensor_batch[key] = np.array([None] * n_samples, dtype=object)
                else:
                    non_tensor_batch[key] = np.array([None] * n_samples, dtype=object)
        
        verification_batch = DataProto(
            batch=batch_dict,
            non_tensor_batch=non_tensor_batch,
        )
        
        return verification_batch

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        current_epoch = self.global_steps // len(self.train_dataloader)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            # compute reward model score on batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                if not self.use_reward_loop:
                                    rm_scores = self.rm_wg.compute_rm_score(batch)
                                else:
                                    assert self.reward_loop_manager is not None, "RewardLoopManager is None"
                                    rm_scores = self.reward_loop_manager.compute_rm_score(batch)
                                batch = batch.union(rm_scores)

                            # Compute or extract reward for REMAX baseline
                            reward_baseline_tensor = self._compute_or_extract_reward(
                                batch, reward_fn=self.reward_fn, sum_reward=True
                            )

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            batch.pop(batch_keys=list(keys_to_pop))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    # get images_seqlens
                    images_seqlens_all = []
                    for multi_modal_input in batch.non_tensor_batch["multi_modal_inputs"]:
                        if "image_grid_thw" not in multi_modal_input.keys():
                            continue
                        images_seqlens_all.extend(multi_modal_input["images_seqlens"].tolist())
                    batch.meta_info["images_seqlens"] = images_seqlens_all
                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            if not self.use_reward_loop:
                                reward_tensor = self.rm_wg.compute_rm_score(batch)
                            else:
                                assert self.reward_loop_manager is not None, "RewardLoopManager is None"
                                reward_tensor = self.reward_loop_manager.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # Compute or extract reward for training
                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(
                                data=batch, config=self.config, tokenizer=self.tokenizer
                            )
                        else:
                            reward_tensor, reward_extra_infos_dict = self._compute_or_extract_reward(
                                batch, reward_fn=self.reward_fn, return_dict=False
                            )

                        # Ensure token_level_scores is set
                        batch.batch["token_level_scores"] = reward_tensor

                        # =============================================================================
                        # [CUSTOM IMPLEMENTATION] Self-Verification Task Injection
                        # =============================================================================
                        batch_size = self._get_batch_size(batch)

                        if "data_source" not in batch.non_tensor_batch:
                            batch.non_tensor_batch["data_source"] = np.array(["rl"] * batch_size, dtype=object)

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout correction (only in decoupled mode)
                        rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                        bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            metrics.update(is_metrics)

                        # ★★★ 핵심 수정: Advantage 계산 순서 변경 ★★★
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)

                        if self.config.trainer.get("use_verification_task", False):
                            # 1. Original batch의 advantage 먼저 계산 (GRPO)
                            batch = compute_advantage(
                                batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                num_repeat=self.config.actor_rollout_ref.rollout.n,
                                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                config=self.config.algorithm,
                            )
                            
                            # 2. Verification batch 생성
                            verif_batch = self._generate_verification_batch(batch)
                            
                            if verif_batch is not None:
                                verif_count = self._get_batch_size(verif_batch)
                                
                                # 3. Verification batch의 advantages 직접 설정 (SFT-like: 고정 1.0)
                                verif_batch.batch['token_level_rewards'] = verif_batch.batch['token_level_scores'].clone()
                                verif_batch.batch['advantages'] = verif_batch.batch['token_level_scores'].clone()
                                verif_batch.batch['returns'] = verif_batch.batch['token_level_scores'].clone()
                                
                                # 4. 합치기 (둘 다 advantage 계산 완료 상태)
                                batch = self._pad_and_concat(batch, verif_batch)
                                metrics["verification/count"] = verif_count
                            else:
                                metrics["verification/count"] = 0
                        else:
                            # Baseline: 기존대로 advantage 계산
                            batch = compute_advantage(
                                batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                num_repeat=self.config.actor_rollout_ref.rollout.n,
                                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                config=self.config.algorithm,
                            )

                        # reward_extra_infos_dict 업데이트
                        if reward_extra_infos_dict:
                            current_bsz = batch.batch.batch_size[0]
                            dict_bsz = len(next(iter(reward_extra_infos_dict.values())))
                            
                            if dict_bsz == current_bsz:
                                batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})
                            elif dict_bsz < current_bsz:
                                for k, v in reward_extra_infos_dict.items():
                                    full_arr = batch.non_tensor_batch.get(k, np.array([None]*current_bsz, dtype=object))
                                    full_arr[:dict_bsz] = np.array(v)
                                    batch.non_tensor_batch[k] = full_arr

                    # Operating Mode Selection:
                    # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                    # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                    #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                    if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                        from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                        apply_bypass_mode(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:  # Recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            
                            # ★★★ Shape 맞추기 추가 ★★★
                            if entropys.shape[1] != response_masks.shape[1]:
                                target_len = response_masks.shape[1]
                                curr_len = entropys.shape[1]
                                if curr_len < target_len:
                                    # 패딩 추가
                                    entropys = torch.nn.functional.pad(entropys, (0, target_len - curr_len), value=0.0)
                                    old_log_prob.batch["old_log_probs"] = torch.nn.functional.pad(
                                        old_log_prob.batch["old_log_probs"], (0, target_len - curr_len), value=0.0
                                    )
                                else:
                                    # 트렁케이션
                                    entropys = entropys[:, :target_len]
                                    old_log_prob.batch["old_log_probs"] = old_log_prob.batch["old_log_probs"][:, :target_len]
                            
                            actor_config = self.config.actor_rollout_ref.actor
                            entropy_agg = agg_loss(
                                loss_mat=entropys,
                                loss_mask=response_masks,
                                loss_agg_mode=actor_config.loss_agg_mode,
                                loss_scale_factor=actor_config.loss_scale_factor,
                            )
                            old_log_prob_metrics = {
                                "actor/entropy": entropy_agg.detach().item(),
                                "perf/mfu/actor_infer": old_log_prob_mfu,
                            }
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            ref_log_prob = self._compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self._compute_values(batch)
                            batch = batch.union(values)

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self._update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            actor_output = self._update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # compute variance proxy metrics
                gradient_norm = metrics.get("actor/grad_norm", None)
                metrics.update(compute_variance_proxy_metrics(batch=batch, gradient_norm=gradient_norm))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
