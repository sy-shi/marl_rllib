from gym_multigrid.envs.collect_game import CollectGame4HEnv10x10N2
from gym_multigrid.envs.usar_game import USARGame
import gym
from gym_multigrid.action_masking import ActionMasking

def env_creator(config)->gym.Env:
    """
    create multigrid environments for Ray. \n
    Currently supports the collect environment.
    Input is the env_config
    """
    agents = [i+1 for i in range(config["n_agents"])]
    victim_reward = 1 # only one victim
    rubble_reward = 1
    room_config = config["config"]
    
    env = USARGame(config=room_config, 
                   start_rooms=config["start_rooms"], 
                   goal_rooms=config["goal_rooms"],
                   rubble_rooms=config["rubble_rooms"],
                   room_size=config["room_size"],
                   agents_index=agents, 
                   view_size=config["view_size"],
                   victim_reward = victim_reward,
                   rubble_reward=rubble_reward,
                   max_steps=config["max_steps"], n_rooms=len(room_config)*len(room_config[0]))
    return ActionMasking(env)
    # return env