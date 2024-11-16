import ray
import pandas as pd
import matplotlib.pyplot as plt
import os
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.typing import PolicyID
from ray.tune.callback import Callback
import shutil
import csv
import gym.spaces
import numpy as np
import torch

from typing import Dict, Optional, Union
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID, SampleBatch
from overcooked.rllib_utils import TrainingCallbacks


class CustomCheckpointCallback(Callback):
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir

    def on_checkpoint(self, iteration, trials, trial, checkpoint, **info):
        # Checkpoint is a Checkpoint object which contains metadata
        checkpoint_path = checkpoint.dir_or_data
        if checkpoint_path:
            # Construct new checkpoint path
            new_checkpoint_path = os.path.join(self.checkpoint_dir, os.path.basename(checkpoint_path))
            # Copy or move the checkpoint to the new location
            shutil.copytree(checkpoint_path, new_checkpoint_path)
            print(f"Checkpoint saved to {new_checkpoint_path}")


class LoggingCallbacks(TrainingCallbacks):
    def __init__(self):
        self.in_evaluation = False
        self.posfix = '_beginning_mixed_10k1.csv'
        # self.teacher_policy = "pol1"
        # self.eat_agent = ray.get_actor(name="EAT_agent", namespace="1")
        self.bandit_result = None
        """
        flag variable to disable advising in evaluation. This should work when there are only 1 envrionment running
        in the worker, since no other parallel evaluations or trainings are happening on this worker who holds only
        1 this callback instance.

        Currently however not working, seems to because the 
        """
    
    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        super().on_episode_end(worker, base_env, policies, episode)

    
    def on_episode_start(
        self, 
        *, 
        worker: RolloutWorker, 
        base_env: BaseEnv, 
        policies: Dict[PolicyID, Policy], 
        episode: Episode, 
        **kwargs
    ):
        super().on_episode_start(worker, base_env, policies, episode)


    def on_evaluate_start(self, *, algorithm: Algorithm, **kwargs) -> None:
        super().on_evaluate_start(algorithm=algorithm, **kwargs)