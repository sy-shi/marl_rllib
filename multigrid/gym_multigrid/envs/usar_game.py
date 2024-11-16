from gym_multigrid.multigrid import *
from gym.spaces import Dict, Discrete, MultiDiscrete, Tuple
import numpy as np
import os
import math

cut_down_thre = 100 

    
class USARGame(MultiRoomGrid):
    """
    Urban Search and Rescue environment
    """
    def __init__(self, config, start_rooms, goal_rooms, rubble_rooms, room_size,
        rubble_index=1,
        victim_reward = 1,
        agents_index = [],
        rubble_reward=None,
        zero_sum = False,
        global_reward = True,
        view_size=7,
        max_steps = 10000,
        n_rooms = 4
    ):
        # print('n_room', n_rooms)

        super().__init__(config,start_rooms, goal_rooms, rubble_rooms, room_size, max_steps=max_steps, rubble_reward=rubble_reward,
                         agents_index=agents_index, victim_reward=victim_reward, view_size=view_size)
        self.n_rooms = n_rooms
        self.heal_for_reward = True       
        self.zero_sum = zero_sum
        self.global_reward = global_reward
        self.steps = 0
        self.healed_flag = [0, 0]
        self.rubble_flag = [0, 0, 0]

    def seed(self, seed=None):
        if seed:
            np.random.seed(seed)
    
    def reset(self, agent=None, victim_pos=None):
        self.healed_flag = [0, 0]
        self.rubble_flag = [0, 0, 0]
        self.steps = 0
        obs= super().reset(agent, victim_pos)
        return obs
    
    
    def _reward(self, i, rewards, reward=1):
        """
        Compute the reward to be given for healing and removing rubble
        """
        # for j,a in enumerate(self.agents):
            # if self.global_reward:
        rewards[i] += reward
            # elif a.index==self.agents[i].index or a.index==0:
                # rewards[j]+=reward
    
    def _handle_pickup(self, i, rewards, fwd_pos, fwd_cell):
        # return
        if i != 2: return # only agent 2 can pickup key
        done = False
        if fwd_cell:
            if fwd_cell.can_pickup():
                self.agents[i].carrying = fwd_cell
                fwd_cell.cur_pos = np.array([-1, -1])
                self.grid.set(*fwd_pos, None)
                if i == 2:
                    rewards[i] += 0.5 - 0.5*self.step_count / self.max_steps
                else:
                    rewards[i] -= 1
                print("agent {} pick up key".format(i))
        return done

    def _handle_toggle(self, i, rewards, fwd_pos, fwd_cell):
        if i != 2: return # only agent 2 can open the door
        if fwd_cell:
            door_open = fwd_cell.toggle(self.agents[i], fwd_pos)
            if door_open:
                self.grid.set(*fwd_pos, None)
                if i == 2:
                    rewards[i] += 0.5 - 0.5*self.step_count / self.max_steps
                else:
                    rewards[i] -= 1
    
    def _handle_remove(self, i, rewards, fwd_pos, fwd_cell):
        # return
        if i != 0: return # only agent 0 can pickup rubbles
        done = False
        if fwd_cell:
            if fwd_cell.can_pickup():
                fwd_cell.cur_pos = np.array([-1, -1])
                self.grid.set(*fwd_pos, None)
                for k in range(3):
                    if self.grid.get(*self.rubble_pos[k]) is None or self.grid.get(*self.rubble_pos[k]).type != "ball":
                        self.rubble_flag[k] = 1
                if i == 0:
                    rewards[i] += 0.33 - 0.33*self.step_count / self.max_steps
                else:
                    rewards[i] -= 1
                done = False
        return done

    def _handle_heal(self, i, rewards, fwd_pos, fwd_cell):
        # return
        if i != 1: return False # only agent 1 can heal victim
        done = False
        if fwd_cell:
            if fwd_cell.can_heal():
                healed = fwd_cell.heal(self.steps, i)
                if healed:
                    self.grid.set(*fwd_pos, None)
                    for k in range(2):
                        if self.grid.get(*self.victim_pos[k]) is None or self.grid.get(*self.victim_pos[k]).type != "victim":
                            self.healed_flag[k] = 1
                    if i == 1:
                        rewards[i] += 0.5 - 0.5*self.step_count / self.max_steps
                    else:
                        rewards[i] -= 1
                    print("agent {} healed the victim".format(i))
        return False

    def judge_done(self, rewards):
        if np.sum(self.rubble_flag) == 3 and np.sum(self.healed_flag) == 2:
            print("done for job completed")
            return True
        else:
            return False

    def step(self, actions):
        obs, rewards, done, info = super().step(actions)
        self.steps += 1
        done = self.judge_done(rewards) or done

        return obs, rewards, done, info