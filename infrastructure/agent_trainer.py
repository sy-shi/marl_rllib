import logging
import functools

from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.util.debug import log_once

from ray.rllib.execution.rollout_ops import (
    standardize_fields,
)
from ray.rllib.execution.train_ops import (
    train_one_step,
    multi_gpu_train_one_step,
)
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.typing import ResultDict
from ray.rllib.execution.rollout_ops import synchronous_parallel_sample
from ray.rllib.utils.metrics import (
    NUM_AGENT_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED,
    SYNCH_WORKER_WEIGHTS_TIMER,
)

from ray.rllib.evaluation.metrics import (
    collect_episodes,
)
from ray.rllib.evaluation.worker_set import WorkerSet


logger = logging.getLogger(__name__)


from ray.rllib.algorithms.ppo.ppo import PPO
from infrastructure.agent_policy import AgentPolicy


class PPOPolicyTrainer(PPO):

    def get_default_policy_class(self, config):
        return AgentPolicy

    def setup(self, config: AlgorithmConfig) -> None:
        super().setup(config)
        # Do not issue advice to student policy in evaluation
        def set_eval_policy(policy, policy_id):
            # customize evaluation policy setup
            pass
        # For safety, use remote workers for evaluation
        assert self.evaluation_workers
        self.evaluation_workers.foreach_worker(
            func = lambda worker: worker.foreach_policy(
                func=set_eval_policy
            ),
            local_worker=False # technically there shouldn't be any local workers
        )
    

    def training_step(self) -> ResultDict:
        # Collect SampleBatches from sample workers until we have a full batch.
        if self.config.count_steps_by == "agent_steps":
            train_batch = synchronous_parallel_sample(
                worker_set=self.workers, max_agent_steps=self.config.train_batch_size
            )
        else:
            train_batch = synchronous_parallel_sample(
                worker_set=self.workers, max_env_steps=self.config.train_batch_size
            )
        train_batch = train_batch.as_multi_agent()
        self._counters[NUM_AGENT_STEPS_SAMPLED] += train_batch.agent_steps()
        self._counters[NUM_ENV_STEPS_SAMPLED] += train_batch.env_steps()

        # Standardize advantages
        train_batch = standardize_fields(train_batch, ["advantages"])
        # Train
        if self.config.simple_optimizer:
            train_results = train_one_step(self, train_batch)
        else:
            train_results = multi_gpu_train_one_step(self, train_batch)

        policies_to_update = list(train_results.keys())

        global_vars = {
            "timestep": self._counters[NUM_AGENT_STEPS_SAMPLED],
            "num_grad_updates_per_policy": {
                pid: self.workers.local_worker().policy_map[pid].num_grad_updates
                for pid in policies_to_update
            },
        }

        
        def set_pol_vars(policy, policy_id):
            # customized worker's policies property update function
            pass
        
        # Update weights - after learning on the local worker - on all remote
        # workers.
        with self._timers[SYNCH_WORKER_WEIGHTS_TIMER]:
            # also update bandit arm reward to policies in all workers here
            self.workers.foreach_worker(
                lambda worker: worker.foreach_policy(
                    func=set_pol_vars
                )
            )
            # self.workers.foreach_policy(func=set_arm_vars)
            if self.workers.num_remote_workers() > 0:
                self.workers.sync_weights(
                    policies=policies_to_update,
                    global_vars=global_vars,
                )

        # For each policy: Update KL scale and warn about possible issues
        for policy_id, policy_info in train_results.items():
            # Update KL loss with dynamic scaling
            # for each (possibly multiagent) policy we are training
            kl_divergence = policy_info[LEARNER_STATS_KEY].get("kl")
            self.get_policy(policy_id).update_kl(kl_divergence)

            # Warn about excessively high value function loss
            scaled_vf_loss = (
                self.config.vf_loss_coeff * policy_info[LEARNER_STATS_KEY]["vf_loss"]
            )
            policy_loss = policy_info[LEARNER_STATS_KEY]["policy_loss"]
            if (
                log_once("ppo_warned_lr_ratio")
                and self.config.get("model", {}).get("vf_share_layers")
                and scaled_vf_loss > 100
            ):
                logger.warning(
                    "The magnitude of your value function loss for policy: {} is "
                    "extremely large ({}) compared to the policy loss ({}). This "
                    "can prevent the policy from learning. Consider scaling down "
                    "the VF loss by reducing vf_loss_coeff, or disabling "
                    "vf_share_layers.".format(policy_id, scaled_vf_loss, policy_loss)
                )
            # Warn about bad clipping configs.
            train_batch.policy_batches[policy_id].set_get_interceptor(None)
            mean_reward = train_batch.policy_batches[policy_id]["rewards"].mean()
            if (
                log_once("ppo_warned_vf_clip")
                and mean_reward > self.config.vf_clip_param
            ):
                self.warned_vf_clip = True
                logger.warning(
                    f"The mean reward returned from the environment is {mean_reward}"
                    f" but the vf_clip_param is set to {self.config['vf_clip_param']}."
                    f" Consider increasing it for policy: {policy_id} to improve"
                    " value function convergence."
                )

        # Update global vars on local worker as well.
        self.workers.local_worker().set_global_vars(global_vars)

        return train_results

    
    def _before_evaluate(self):
        # sync evaluation workers with local workers in this hook
        local_worker = self.workers.local_worker()
        def set_pol_vars(policy, policy_id):
            # customize policy vars for evaluation
            pass
        
        self.evaluation_workers.foreach_worker(
            func = lambda worker: worker.foreach_policy(
                func = set_pol_vars
            ),
            local_worker=False
        )
        print("============================= start to evaluate =======================")


    def step(self) -> ResultDict:
        # can be modified for customized evaluation frequency, etc.
        return super().step()


def get_trainer(base_class=PPO):
    """
    Currently get a PPO trainer
    """
    if base_class == PPO:
        return PPOPolicyTrainer
    else:
        raise NotImplementedError