
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.algorithms.dqn import DQNTorchPolicy
from ray.rllib.evaluation import Episode
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.sample_batch import SampleBatch

from ray.rllib.utils.typing import TensorType

from model.utils import ModelWrapper
from ray import cloudpickle
from pathlib import Path
import torch
from ray.rllib.evaluation.postprocessing import Postprocessing

from typing import Dict, List, Optional, Tuple, Type, Union
from ray.rllib.utils.typing import TensorType
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
import os
from ray.rllib.utils.torch_utils import (
    explained_variance,
    sequence_mask,
    warn_if_infinite_kl_divergence,
)
import numpy as np
from itertools import permutations


class AgentPolicy(PPOTorchPolicy):
    '''
    PPO policy built based on the PPOTorchPolicy class \n
    reserved necessary interfaces for customization
    '''
    def __init__(self, observation_space, action_space, config):
        self.policy_name = config.get("name")
        self.evaluation = False
        self.agent_id = int(self.policy_name[-1]) -1 # pol1 --> 0; pol2 --> 1; pol3 --> 2

        super().__init__(observation_space, action_space, config)
        PPOTorchPolicy.__init__(self, observation_space, action_space, config)

    def postprocess_trajectory(self, sample_batch, other_agent_batches=None, episode=None):
        '''
        This function is called after one trajectory (segment) is collected from one environment \n
        This function processes the collected sample (s,a,s',r) trajectories, and computes advantages \n
        The processed trajectory is ready to be used for policy update \n
        For on-policy algorithms, the computed logits are also included in the sample \n
        Logic can be added here to customize sample
        '''
        batch = super().postprocess_trajectory(sample_batch, other_agent_batches, episode)    
        return batch
    

    def compute_actions_from_input_dict(self,
                                        input_dict: Dict[str, TensorType],
                                        explore: bool = None,
                                        timestep: int = None,
                                        episodes: Optional[List["Episode"]] = None,
                                        **kwargs) \
                                        -> Tuple[TensorType, List[TensorType], Dict[str, TensorType]]:
        '''
        `input_dict`: {'obs':a batch of observation}, batch size depending on the environments running in parallel
        '''
        return super().compute_actions_from_input_dict(input_dict, explore, timestep, episodes, **kwargs)
    

    def stats_fn(self, train_batch):
        '''
        Update the stats of local worker policies using train batch \n
        The function is called after each time loss function is called \n
        e.g. stats['var1'] = 1 \n
        The updated stats will be collected into the results with key 'learner_stats/policy_name/var1'
        '''
        stats = super().stats_fn(train_batch)
        return convert_to_numpy(stats)


    def loss(
            self,
            model: ModelV2,
            dist_class: Type[ActionDistribution],
            train_batch: SampleBatch,
        ) -> Union[TensorType, List[TensorType]]:
        """Compute loss for Proximal Policy Objective.
        Args:
            model: The Model to calculate the loss for.
            dist_class: The action distr. class.
            train_batch: The training data.
        Returns:
            The PPO loss tensor given the input batch.
        """
        logits, state = model(train_batch)
        curr_action_dist = dist_class(logits, model)

        # RNN case: Mask away 0-padded chunks at end of time axis.
        if state:
            B = len(train_batch[SampleBatch.SEQ_LENS])
            max_seq_len = logits.shape[0] // B
            mask = sequence_mask(
                train_batch[SampleBatch.SEQ_LENS],
                max_seq_len,
                time_major=model.is_time_major(),
            )
            mask = torch.reshape(mask, [-1])
            num_valid = torch.sum(mask)

            def reduce_mean_valid(t):
                return torch.sum(t[mask]) / num_valid

        # non-RNN case: No masking.
        else:
            mask = None
            reduce_mean_valid = torch.mean

        prev_action_dist = dist_class(
            train_batch[SampleBatch.ACTION_DIST_INPUTS], model
        )

        logp_ratio = torch.exp(
            curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
            - train_batch[SampleBatch.ACTION_LOGP]
        )

        # Only calculate kl loss if necessary (kl-coeff > 0.0).
        if self.config["kl_coeff"] > 0.0:
            action_kl = prev_action_dist.kl(curr_action_dist)
            mean_kl_loss = reduce_mean_valid(action_kl)
            # TODO smorad: should we do anything besides warn? Could discard KL term
            # for this update
            warn_if_infinite_kl_divergence(self, mean_kl_loss)
        else:
            mean_kl_loss = torch.tensor(0.0, device=logp_ratio.device)

        curr_entropy = curr_action_dist.entropy()
        mean_entropy = reduce_mean_valid(curr_entropy)

        advantages = train_batch[Postprocessing.ADVANTAGES]

        surrogate_loss = torch.min(
            advantages * logp_ratio,
            advantages
            * torch.clamp(
                logp_ratio, 1 - self.config["clip_param"], 1 + self.config["clip_param"]
            ),
        )

        # Compute a value function loss.
        if self.config["use_critic"]:
            value_fn_out = model.value_function()
            vf_loss = torch.pow(
                value_fn_out - train_batch[Postprocessing.VALUE_TARGETS], 2.0
            )
            # Scale the VF loss by the student's IS ratio. Note that this only affects actions coming from the teacher.
            if self.advice_mode and self.off_policy_correction and "student_is_ratio" in train_batch:
                vf_loss = vf_loss * train_batch["student_is_ratio"]

            vf_loss_clipped = torch.clamp(vf_loss, 0, self.config["vf_clip_param"])
            mean_vf_loss = reduce_mean_valid(vf_loss_clipped)
        # Ignore the value function.
        else:
            value_fn_out = torch.tensor(0.0).to(surrogate_loss.device)
            vf_loss_clipped = mean_vf_loss = torch.tensor(0.0).to(surrogate_loss.device)

        total_loss = reduce_mean_valid(
            -surrogate_loss
            + self.config["vf_loss_coeff"] * vf_loss_clipped
            - self.entropy_coeff * curr_entropy
        )

        # Add mean_kl_loss (already processed through `reduce_mean_valid`),
        # if necessary.
        if self.config["kl_coeff"] > 0.0:
            total_loss += self.kl_coeff * mean_kl_loss

        # Store values for stats function in model (tower), such that for
        # multi-GPU, we do not override them during the parallel loss phase.
        model.tower_stats["total_loss"] = total_loss
        model.tower_stats["mean_policy_loss"] = reduce_mean_valid(-surrogate_loss)
        model.tower_stats["mean_vf_loss"] = mean_vf_loss
        model.tower_stats["vf_explained_var"] = explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS], value_fn_out
        )
        model.tower_stats["mean_entropy"] = mean_entropy
        model.tower_stats["mean_kl_loss"] = mean_kl_loss

        return total_loss