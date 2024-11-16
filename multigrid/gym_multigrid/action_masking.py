import gym
import numpy as np

from gym import spaces
from gym.core import ObservationWrapper

import gym_multigrid.multigrid

class ActionMasking(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        # The action mask sets a value for each action of either 0 (invalid) or 1 (valid).
        self.observation_space = spaces.Dict({
            "image": self.env.observation_space,
            "action_mask": spaces.Box(0.0, 1.0, shape=(self.action_space.n,),),
            "advice_mask": spaces.Box(0.0, 1.0, shape=(self.action_space.n,),),
        })
    
    def reset(self, agent=None, victim_pos=None, **kwargs):
        """Resets the environment, returning a modified observation using :meth:`self.observation`."""
        if kwargs.get("return_info", False):
            obs, info = self.env.reset(agent, victim_pos)
            return self.observation(obs), info
        else:
            return self.observation(self.env.reset(agent, victim_pos, **kwargs))
        
    
    def observation(self, obs):
        obs_mask = []
        for agent_idx in range(len(self.unwrapped.agents)):
            action_mask = np.ones(self.action_space.n, dtype=np.float32)

            # Look at the position directly in front of the agent
            front_pos = self.unwrapped.agents[agent_idx].front_pos
            if self.unwrapped.partial_obs:
                front_cell = self.unwrapped.grid.get(*front_pos)
                front_pos_type = gym_multigrid.multigrid.World.OBJECT_TO_IDX[front_cell.type]
            else:
                front_pos_type = obs[agent_idx][front_pos[0]][front_pos[1]][0]
            
            if front_pos_type == gym_multigrid.multigrid.World.OBJECT_TO_IDX["wall"]:
                action_mask[self.unwrapped.actions.forward] = 0.0

            if front_pos_type != gym_multigrid.multigrid.World.OBJECT_TO_IDX["key"]:
                action_mask[self.env.actions.pickup] = 0.0
            
            if front_pos_type != gym_multigrid.multigrid.World.OBJECT_TO_IDX["victim"]:
                action_mask[self.unwrapped.actions.heal] = 0.0

            if front_pos_type != gym_multigrid.multigrid.World.OBJECT_TO_IDX["ball"]:
                action_mask[self.unwrapped.actions.remove] = 0.0

            if front_pos_type != gym_multigrid.multigrid.World.OBJECT_TO_IDX["door"]:
                action_mask[self.unwrapped.actions.toggle] = 0.0
            
            # if agent_idx == 0:
            #     action_mask[self.unwrapped.actions.heal] = 0.0
            # else:
            #     action_mask[self.unwrapped.actions.pickup] = 0.0
            #     action_mask[self.unwrapped.actions.toggle] = 0.0
            # mask the healing action when there is a rubble


            # print("***********")
            # print(front_pos)
            # print(self.unwrapped._key_location)
            # if not np.all(front_pos != self.unwrapped._key_location)\
            #       or (agent_idx == 1 and len(self.unwrapped.agents) == 2):
            #     action_mask[self.unwrapped.actions.toggle] = 0.0

            # Now disable actions that we intend to never use
            # action_mask[self.unwrapped.actions.pickup] = 0.0
            # action_mask[self.unwrapped.actions.drop] = 0.0
            # action_mask[self.unwrapped.actions.toggle] = 0.0
            # action_mask[self.unwrapped.actions.done] = 0.0

            obs_mask.append({"image":obs[agent_idx], "action_mask": action_mask, "advice_mask": action_mask})

        return obs_mask
    