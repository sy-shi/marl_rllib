import copy
import logging
import os
import random
from datetime import datetime

import gym
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from utils import (
    get_base_ae,
    get_required_arguments,
    iterable_equal,
    softmax,
)
from overcooked_ai_py.agents.agent import Agent, AgentPair
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import (
    EVENT_TYPES,
    OvercookedGridworld,
)

action_space = gym.spaces.Discrete(len(Action.ALL_ACTIONS))
obs_space = gym.spaces.Discrete(len(Action.ALL_ACTIONS))
timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")


class RlLibAgent(Agent):
    """
    Class for wrapping a trained RLLib Policy object into an Overcooked compatible Agent
    """

    def __init__(self, policy, agent_index, featurize_fn):
        self.policy = policy
        self.agent_index = agent_index
        self.featurize = featurize_fn

    def reset(self):
        # Get initial rnn states and add batch dimension to each
        if hasattr(self.policy.model, "get_initial_state"):
            self.rnn_state = [
                np.expand_dims(state, axis=0)
                for state in self.policy.model.get_initial_state()
            ]
        elif hasattr(self.policy, "get_initial_state"):
            self.rnn_state = [
                np.expand_dims(state, axis=0)
                for state in self.policy.get_initial_state()
            ]
        else:
            self.rnn_state = []

    def action_probabilities(self, state):
        """
        Arguments:
            - state (Overcooked_mdp.OvercookedState) object encoding the global view of the environment
        returns:
            - Normalized action probabilities determined by self.policy
        """
        # Preprocess the environment state
        obs = self.featurize(state, debug=False)
        my_obs = obs[self.agent_index]

        # Compute non-normalized log probabilities from the underlying model
        logits = self.policy.compute_actions(
            np.array([my_obs]), self.rnn_state
        )[2]["action_dist_inputs"]

        # Softmax in numpy to convert logits to normalized probabilities
        return softmax(logits)

    def action(self, state):
        """
        Arguments:
            - state (Overcooked_mdp.OvercookedState) object encoding the global view of the environment
        returns:
            - the argmax action for a single observation state
            - action_info (dict) that stores action probabilities under 'action_probs' key
        """
        # Preprocess the environment state
        obs = self.featurize(state)
        my_obs = obs[self.agent_index]

        # Use Rllib.Policy class to compute action argmax and action probabilities
        # The first value is action_idx, which we will recompute below so the results are stochastic
        _, rnn_state, info = self.policy.compute_actions(
            np.array([my_obs]), self.rnn_state
        )

        # Softmax in numpy to convert logits to normalized probabilities
        logits = info["action_dist_inputs"]
        action_probabilities = softmax(logits)

        # The original design is stochastic across different games,
        # Though if we are reloading from a checkpoint it would inherit the seed at that point, producing deterministic results
        [action_idx] = random.choices(
            list(range(Action.NUM_ACTIONS)), action_probabilities[0]
        )
        agent_action = Action.INDEX_TO_ACTION[action_idx]

        agent_action_info = {"action_probs": action_probabilities}
        self.rnn_state = rnn_state

        return agent_action, agent_action_info


class OvercookedMultiAgent(MultiAgentEnv):
    """
    Class used to wrap OvercookedEnv in an Rllib compatible multi-agent environment
    """

    # List of all agent types currently supported
    supported_agents = ["ppo", "bc"]

    # Default bc_schedule, includes no bc agent at any time
    bc_schedule = self_play_bc_schedule = [(0, 0), (float("inf"), 0)]

    # Default environment params used for creation
    DEFAULT_CONFIG = {
        # To be passed into OvercookedGridWorld constructor
        "mdp_params": {
            "layout_name": "cramped_room",
            "rew_shaping_params": {},
        },
        # To be passed into OvercookedEnv constructor
        "env_params": {"horizon": 400},
        # To be passed into OvercookedMultiAgent constructor
        "multi_agent_params": {
            "reward_shaping_factor": 0.0,
            "reward_shaping_horizon": 0,
            "bc_schedule": self_play_bc_schedule,
            "use_phi": True,
        },
    }

    def __init__(
        self,
        base_env,
        reward_shaping_factor=0.0,
        reward_shaping_horizon=0,
        bc_schedule=None,
        use_phi=True,
    ):
        """
        base_env: OvercookedEnv
        reward_shaping_factor (float): Coefficient multiplied by dense reward before adding to sparse reward to determine shaped reward
        reward_shaping_horizon (int): Timestep by which the reward_shaping_factor reaches zero through linear annealing
        bc_schedule (list[tuple]): List of (t_i, v_i) pairs where v_i represents the value of bc_factor at timestep t_i
            with linear interpolation in between the t_i
        use_phi (bool): Whether to use 'shaped_r_by_agent' or 'phi_s_prime' - 'phi_s' to determine dense reward
        """
        if bc_schedule:
            self.bc_schedule = bc_schedule
        self._validate_schedule(self.bc_schedule)
        self.base_env = base_env
        # since we are not passing featurize_fn in as an argument, we create it here and check its validity
        self.featurize_fn_map = {
            "ppo": lambda state: self.base_env.lossless_state_encoding_mdp(
                state
            ),
            "bc": lambda state: self.base_env.featurize_state_mdp(state),
        }
        self._validate_featurize_fns(self.featurize_fn_map)
        self._initial_reward_shaping_factor = reward_shaping_factor
        self.reward_shaping_factor = reward_shaping_factor
        self.reward_shaping_horizon = reward_shaping_horizon
        self.use_phi = use_phi
        self.anneal_bc_factor(0)
        self._agent_ids = set(self.reset().keys())
        # fixes deprecation warnings
        self._spaces_in_preferred_format = True

    def _validate_featurize_fns(self, mapping):
        assert "ppo" in mapping, "At least one ppo agent must be specified"
        for k, v in mapping.items():
            assert (
                k in self.supported_agents
            ), "Unsuported agent type in featurize mapping {0}".format(k)
            assert callable(v), "Featurize_fn values must be functions"
            assert (
                len(get_required_arguments(v)) == 1
            ), "Featurize_fn value must accept exactly one argument"

    def _validate_schedule(self, schedule):
        timesteps = [p[0] for p in schedule]
        values = [p[1] for p in schedule]

        assert (
            len(schedule) >= 2
        ), "Need at least 2 points to linearly interpolate schedule"
        assert schedule[0][0] == 0, "Schedule must start at timestep 0"
        assert all(
            [t >= 0 for t in timesteps]
        ), "All timesteps in schedule must be non-negative"
        assert all(
            [v >= 0 and v <= 1 for v in values]
        ), "All values in schedule must be between 0 and 1"
        assert (
            sorted(timesteps) == timesteps
        ), "Timesteps must be in increasing order in schedule"

        # To ensure we flatline after passing last timestep
        if schedule[-1][0] < float("inf"):
            schedule.append((float("inf"), schedule[-1][1]))

    def _setup_action_space(self, agents):
        action_sp = {}
        for agent in agents:
            action_sp[agent] = gym.spaces.Discrete(len(Action.ALL_ACTIONS))
        self.action_space = gym.spaces.Dict(action_sp)
        self.shared_action_space = gym.spaces.Discrete(len(Action.ALL_ACTIONS))

    def _setup_observation_space(self, agents):
        dummy_state = self.base_env.mdp.get_standard_start_state()
        # ppo observation
        featurize_fn_ppo = (
            lambda state: self.base_env.lossless_state_encoding_mdp(state)
        )
        obs_shape = featurize_fn_ppo(dummy_state)[0].shape

        high = np.ones(obs_shape) * float("inf")
        low = np.ones(obs_shape) * 0
        self.ppo_observation_space = gym.spaces.Box(
            np.float32(low), np.float32(high), dtype=np.float32
        )

        # bc observation
        featurize_fn_bc = lambda state: self.base_env.featurize_state_mdp(
            state
        )
        obs_shape = featurize_fn_bc(dummy_state)[0].shape
        high = np.ones(obs_shape) * 100
        low = np.ones(obs_shape) * -100
        self.bc_observation_space = gym.spaces.Box(
            np.float32(low), np.float32(high), dtype=np.float32
        )
        # hardcode mapping between action space and agent
        ob_space = {}
        for agent in agents:
            if agent.startswith("ppo"):
                ob_space[agent] = self.ppo_observation_space
            else:
                ob_space[agent] = self.bc_observation_space
        self.observation_space = gym.spaces.Dict(ob_space)

    def _get_featurize_fn(self, agent_id):
        if agent_id.startswith("ppo"):
            return lambda state: self.base_env.lossless_state_encoding_mdp(
                state
            )
        if agent_id.startswith("bc"):
            return lambda state: self.base_env.featurize_state_mdp(state)
        raise ValueError("Unsupported agent type {0}".format(agent_id))

    def _get_obs(self, state):
        ob_p0 = self._get_featurize_fn(self.curr_agents[0])(state)[0]
        ob_p1 = self._get_featurize_fn(self.curr_agents[1])(state)[1]
        return ob_p0.astype(np.float32), ob_p1.astype(np.float32)

    def _populate_agents(self):
        # Always include at least one ppo agent (i.e. bc_sp not supported for simplicity)
        agents = ["ppo"]

        # Coin flip to determine whether other agent should be ppo or bc
        other_agent = "bc" if np.random.uniform() < self.bc_factor else "ppo"
        agents.append(other_agent)

        # Randomize starting indices
        np.random.shuffle(agents)

        # Ensure agent names are unique
        agents[0] = agents[0] + "_0"
        agents[1] = agents[1] + "_1"

        # logically the action_space and the observation_space should be set along with the generated agents
        # the agents are also randomized in each iteration if bc agents are allowed, which requires reestablishing the action & observation space
        self._setup_action_space(agents)
        self._setup_observation_space(agents)
        return agents

    def _anneal(self, start_v, curr_t, end_t, end_v=0, start_t=0):
        if end_t == 0:
            # No annealing if horizon is zero
            return start_v
        else:
            off_t = curr_t - start_t
            # Calculate the new value based on linear annealing formula
            fraction = max(1 - float(off_t) / (end_t - start_t), 0)
            return fraction * start_v + (1 - fraction) * end_v

    def step(self, action_dict):
        """
        action:
            (agent with index self.agent_idx action, other agent action)
            is a tuple with the joint action of the primary and secondary agents in index format

        returns:
            observation: formatted to be standard input for self.agent_idx's policy
        """
        action = [
            action_dict[self.curr_agents[0]],
            action_dict[self.curr_agents[1]],
        ]

        assert all(
            self.action_space[agent].contains(action_dict[agent])
            for agent in action_dict
        ), "%r (%s) invalid" % (action, type(action))
        joint_action = [Action.INDEX_TO_ACTION[a] for a in action]
        # take a step in the current base environment

        if self.use_phi:
            next_state, sparse_reward, done, info = self.base_env.step(
                joint_action, display_phi=True
            )
            potential = info["phi_s_prime"] - info["phi_s"]
            dense_reward = (potential, potential)
        else:
            next_state, sparse_reward, done, info = self.base_env.step(
                joint_action, display_phi=False
            )
            dense_reward = info["shaped_r_by_agent"]

        ob_p0, ob_p1 = self._get_obs(next_state)

        shaped_reward_p0 = (
            sparse_reward + self.reward_shaping_factor * dense_reward[0]
        )
        shaped_reward_p1 = (
            sparse_reward + self.reward_shaping_factor * dense_reward[1]
        )

        obs = {self.curr_agents[0]: ob_p0, self.curr_agents[1]: ob_p1}
        rewards = {
            self.curr_agents[0]: shaped_reward_p0,
            self.curr_agents[1]: shaped_reward_p1,
        }
        dones = {
            self.curr_agents[0]: done,
            self.curr_agents[1]: done,
            "__all__": done,
        }
        infos = {self.curr_agents[0]: info, self.curr_agents[1]: info}
        return obs, rewards, dones, infos

    def reset(self, regen_mdp=True):
        """
        When training on individual maps, we want to randomize which agent is assigned to which
        starting location, in order to make sure that the agents are trained to be able to
        complete the task starting at either of the hardcoded positions.

        NOTE: a nicer way to do this would be to just randomize starting positions, and not
        have to deal with randomizing indices.
        """
        self.base_env.reset(regen_mdp)
        self.curr_agents = self._populate_agents()
        ob_p0, ob_p1 = self._get_obs(self.base_env.state)
        return {self.curr_agents[0]: ob_p0, self.curr_agents[1]: ob_p1}

    def anneal_reward_shaping_factor(self, timesteps):
        """
        Set the current reward shaping factor such that we anneal linearly until self.reward_shaping_horizon
        timesteps, given that we are currently at timestep "timesteps"
        """
        new_factor = self._anneal(
            self._initial_reward_shaping_factor,
            timesteps,
            self.reward_shaping_horizon,
        )
        self.set_reward_shaping_factor(new_factor)

    def anneal_bc_factor(self, timesteps):
        """
        Set the current bc factor such that we anneal linearly until self.bc_factor_horizon
        timesteps, given that we are currently at timestep "timesteps"
        """
        p_0 = self.bc_schedule[0]
        p_1 = self.bc_schedule[1]
        i = 2
        while timesteps > p_1[0] and i < len(self.bc_schedule):
            p_0 = p_1
            p_1 = self.bc_schedule[i]
            i += 1
        start_t, start_v = p_0
        end_t, end_v = p_1
        new_factor = self._anneal(start_v, timesteps, end_t, end_v, start_t)
        self.set_bc_factor(new_factor)

    def set_reward_shaping_factor(self, factor):
        self.reward_shaping_factor = factor

    def set_bc_factor(self, factor):
        self.bc_factor = factor

    def seed(self, seed):
        """
        set global random seed to make environment deterministic
        """
        # Our environment is already deterministic
        pass

    @classmethod
    def from_config(cls, env_config):
        """
        Factory method for generating environments in style with rllib guidlines

        env_config (dict):  Must contain keys 'mdp_params', 'env_params' and 'multi_agent_params', the last of which
                            gets fed into the OvercookedMultiAgent constuctor

        Returns:
            OvercookedMultiAgent instance specified by env_config params
        """
        assert (
            env_config
            and "env_params" in env_config
            and "multi_agent_params" in env_config
        )
        assert (
            "mdp_params" in env_config
            or "mdp_params_schedule_fn" in env_config
        ), "either a fixed set of mdp params or a schedule function needs to be given"
        # "layout_name" and "rew_shaping_params"
        if "mdp_params" in env_config:
            mdp_params = env_config["mdp_params"]
            outer_shape = None
            mdp_params_schedule_fn = None
        elif "mdp_params_schedule_fn" in env_config:
            mdp_params = None
            outer_shape = env_config["outer_shape"]
            mdp_params_schedule_fn = env_config["mdp_params_schedule_fn"]

        # "start_state_fn" and "horizon"
        env_params = env_config["env_params"]
        # "reward_shaping_factor"
        multi_agent_params = env_config["multi_agent_params"]
        base_ae = get_base_ae(
            mdp_params, env_params, outer_shape, mdp_params_schedule_fn
        )
        base_env = base_ae.env

        return cls(base_env, **multi_agent_params)

