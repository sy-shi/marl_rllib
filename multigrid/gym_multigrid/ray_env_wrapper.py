
from gym.spaces import Dict

from ray.rllib.env.multi_agent_env import MultiAgentEnv

from gym_multigrid.ray_env_creator import env_creator

import numpy as np


class MAUSARGame(MultiAgentEnv):
    def __init__(self, env_config) -> None:
        super().__init__()
        self.env = env_creator(env_config)
        # self.eat_agent = ray.get_actor(name="EAT_agent", namespace="1")
        self.n_agents = len(self.env.agents)
        self._agent_ids = {i for i in range(self.n_agents)}
        self.observation_space = Dict({i:self.env.observation_space for i in range(self.n_agents)})
        self.action_space = Dict({i:self.env.action_space for i in range(self.n_agents)})
        self.advice_mask = None
        self.observations = None
        self.reset()


    def reset(self):
        # self.eps_id = ray.get(self.eat_agent.assign_eps_id.remote())
        # ray.get(self.eat_agent.collect_eps_id.remote(self.eps_id))
        obs = self.env.reset()
        self.advice_mask = np.zeros((self.n_agents, self.env.action_space.n), dtype=np.float32)
        # self.advice_mask = ray.get(self.eat_agent.action_advise.remote(obs, self.eps_id, self.env.unwrapped.agents[1].pos))
        for i in range(self.n_agents):
            obs[i]["advice_mask"] = self.advice_mask[i]
        self.observations = {i:obs[i] for i in range(self.n_agents)}
        return self.observations
    

    def step(self, action_dict: dict):
        action_list = [action_dict[key] for key in action_dict]
        obs, rewards, done, _ = self.env.step(action_list)
        # ray.get(self.eat_agent.update_posterior.remote(self.observations, action_dict, self.eps_id))
        # ray.get(self.eat_agent.bandit_reward_update.remote(self.observations, action_dict, self.eps_id, done))
        self.advice_mask = np.zeros((self.n_agents, self.env.action_space.n), dtype=np.float32)
        # self.advice_mask = ray.get(self.eat_agent.action_advise.remote(obs, self.eps_id, self.env.unwrapped.agents[1].pos, done))
        # if done:
        #     ray.get(self.eat_agent.relieve_eps_id.remote(self.eps_id))
        for i in range(self.n_agents):
            obs[i]["advice_mask"] = self.advice_mask[i]
        self.observations = {i:obs[i] for i in range(self.n_agents)}
        dones = {"__all__": done}
        return self.observations,\
               {i:rewards[i] for i in range(self.n_agents)},\
               dones, {}