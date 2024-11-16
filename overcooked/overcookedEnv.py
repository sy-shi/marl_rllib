import copy
import logging
import os
import random
from datetime import datetime

import gym, tqdm
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from overcooked.utils import (
    get_base_ae,
    get_required_arguments,
    iterable_equal,
    softmax,
)

from overcooked_ai_py.utils import load_dict_from_file, mean_and_std_err, append_dictionaries
from overcooked_ai_py.agents.agent import Agent, AgentPair
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.mdp.overcooked_trajectory import TIMESTEP_TRAJ_KEYS, EPISODE_TRAJ_KEYS, DEFAULT_TRAJ_KEYS
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import (
    EVENT_TYPES,
    OvercookedGridworld,
    OvercookedState,
    ObjectState,
    SoupState,
    Recipe
)
from collections import defaultdict, Counter
import itertools

timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")


def from_layout_name(layout, **params_to_overwrite):
    """
    Generates a OvercookedGridworld instance from a layout file.

    One can overwrite the default mdp configuration using partial_mdp_config.
    """
    params_to_overwrite = params_to_overwrite.copy()
    base_layout_params = load_dict_from_file(layout)

    grid = base_layout_params['grid']
    del base_layout_params['grid']
    if 'start_state' in base_layout_params:
        base_layout_params['start_state'] = OvercookedState.from_dict(base_layout_params['start_state'])

    # Clean grid
    grid = [layout_row.strip() for layout_row in grid.split("\n")]
    return OvercookedGridworld.from_grid(grid, base_layout_params, params_to_overwrite)



class OvercookedMultiAgent(MultiAgentEnv):
    """
    Class used to wrap OvercookedEnv in an Rllib compatible multi-agent environment
    """

    def __init__(self, config):
        super().__init__()
        # self.mdp = OvercookedGridworld(**config["mdp_config"])
        self.config = config
        self.env = OvercookedEnv.from_mdp(from_layout_name(config.get("layout")), horizon=config.get("max_steps"))
        
        self.n_agents = self.env.mdp.num_players
        self.agents = [i for i in range(self.n_agents)]
        self._agent_ids = self.agents
        self.n_actions = len(Action.ALL_ACTIONS)
        self.dummy_state = self.env.mdp.get_standard_start_state()
        # store all orders with a fixed sequence, used for observation encoding
        self.all_orders = copy.deepcopy(self.dummy_state._all_orders) #list[Recipe]
        self.action_space = self._setup_action_space(self.agents)
        self.observation_space = self._setup_observation_space(self.agents)
        # print(self.observation_space)
        self._initial_reward_shaping_factor = config.get("reward_shaping_factor")
        self.reward_shaping_factor = config.get("reward_shaping_factor")
        self.reward_shaping_horizon = config.get("reward_shaping_horizon")
        self.use_phi = config.get("use_phi")
        ## overwrite the methods of member objects
        self.env.start_state_fn = self.env.mdp.get_random_start_state_fn(random_start_pos=True) # randomize agent start state
        # customzied resolve interacts method for mdp, assign roles based on rewards
        self.env.mdp.resolve_interacts = resolve_interacts.__get__(self.env.mdp, type(self.env.mdp))
        self.env.mdp.get_state_transition = get_state_transition.__get__(self.env.mdp, type(self.env.mdp))
        # self.env.mdp.potential_function = potential_function.__get__(self.env.mdp, type(self.env.mdp))
        # tricks to get evaulation work
        self.start_state_fn = self.env.start_state_fn
        self.horizon = self.env.horizon
        

    def copy(self):
        return OvercookedMultiAgent(self.config)

    def _setup_action_space(self, agents):
        action_sp = {}
        for agent in agents:
            action_sp[agent] = gym.spaces.Discrete(self.n_actions)
        return gym.spaces.Dict(action_sp)
    
    def _setup_observation_space(self, agents):
        dummy_obs = self.get_lossless_state_encoding_mdp(self.dummy_state)
        obs_shape = dummy_obs[0].shape
        high = np.ones(obs_shape) * float("inf")
        low = np.ones(obs_shape) * 0
        observation_space = gym.spaces.Box(
            np.float32(low), np.float32(high), dtype=np.float32
        )
        action_mask_space = gym.spaces.Box(0.0, 1.0, shape=(self.n_actions,), dtype=np.float32)
        ob_space = {}
        for agent in agents:
            ob_space[agent] = gym.spaces.Dict({'image': observation_space, 'action_mask': action_mask_space})
        return gym.spaces.Dict(ob_space)
    
    def get_obs(self, overcooked_state):
        img = self.get_lossless_state_encoding_mdp(overcooked_state)
        action_mask = self.get_action_mask(overcooked_state)
        return {i: {"image": img[i], "action_mask": action_mask[i]} for i in self.agents}
    
    def get_action_mask(self, overcooked_state):
        action_mask = {}
        for agent_idx in self.agents:
            agent = overcooked_state.players[agent_idx]
            mask = np.ones(self.n_actions, dtype=np.float32)
            # 1. If the agent is holding nothing, it cannot serve, so mask 'serve' action.
            agent_position = agent.position
            near_terrain = False
            for direction in Direction.ALL_DIRECTIONS:
                new_pos = Action.move_in_direction(agent_position, direction)
                if new_pos in self.env.mdp.terrain_pos_dict:  # 'X' typically represents a wall or impassable terrain
                    mask[direction] = 1  # Mask this direction as it leads to a wall
                    near_terrain = True
            if not near_terrain:
                mask[Action.ACTION_TO_INDEX[Action.INTERACT]] = 1
            action_mask[agent_idx] = mask
        return action_mask
    
    def get_lossless_state_encoding_mdp(self, overcooked_state, debug=False):
        """
        Return a dictionary which contains obsevation encoding for all agents
        """
        base_map_features = ["pot_loc", "counter_loc", "onion_disp_loc", "tomato_disp_loc",
                             "dish_disp_loc", "serve_loc"] # 6 
        variable_map_features = ["onions_in_pot", "tomatoes_in_pot", "onions_in_soup", "tomatoes_in_soup",
                                 "soup_cook_time_remaining", "soup_done", "dishes", "onions", "tomatoes", "next_soup"] # 10
        # task_map_features = ["next_soup_{}".format(i) for i in range(len(self.all_orders))]
        urgency_features = ["urgency"]
        all_objects = overcooked_state.all_objects_list

        def make_layer(position, value):
            layer = np.zeros(self.env.mdp.shape)
            layer[position] = value
            return layer

        def process_for_player(ego_agent_idx, state_mask_dict, LAYERS, debug=False):
            state_mask_dict["ego"] = make_layer(overcooked_state.players[ego_agent_idx].position, 1)
            state_mask_stack = np.array([state_mask_dict[layer_id] for layer_id in LAYERS])
            assert state_mask_stack.shape[1:] == self.env.mdp.shape
            assert state_mask_stack.shape[0] == len(LAYERS)
            # NOTE: currently not including time left or order_list in featurization
            if debug:
                print(f"agent {ego_agent_idx}")
                print("terrain----")
                print(np.array(self.env.mdp.terrain_mtx))
                print("-----------")
                print(len(LAYERS))
                print(len(state_mask_dict))
                for k, v in state_mask_dict.items():
                    print(k)
                    print(np.transpose(v, (1, 0)))
            return np.array(state_mask_stack).astype(np.float32)

        def process_state():
            ordered_player_features = ["player_{}_loc".format(i) for i in range(self.n_agents)] + \
                        ["player_{}_orientation_{}".format(i, Direction.DIRECTION_TO_INDEX[d])
                        for i, d in itertools.product([i for i in range(self.n_agents)], Direction.ALL_DIRECTIONS)]

            # LAYERS = ordered_player_features + base_map_features + variable_map_features
            LAYERS = ["ego"] + ["player_loc"] + ["player_orientation_{}".format(Direction.DIRECTION_TO_INDEX[d]) for d in Direction.ALL_DIRECTIONS] + \
                     ordered_player_features + base_map_features + variable_map_features + urgency_features
            state_mask_dict = {k:np.zeros(self.env.mdp.shape) for k in LAYERS}

            # MAP LAYERS
            if self.env.horizon - overcooked_state.timestep < 40:
                state_mask_dict["urgency"] = np.ones(self.env.mdp.shape)
            
            for loc in self.env.mdp.get_counter_locations():
                state_mask_dict["counter_loc"][loc] = 1

            for loc in self.env.mdp.get_pot_locations():
                state_mask_dict["pot_loc"][loc] = 1

            for loc in self.env.mdp.get_onion_dispenser_locations():
                state_mask_dict["onion_disp_loc"][loc] = 1

            for loc in self.env.mdp.get_tomato_dispenser_locations():
                state_mask_dict["tomato_disp_loc"][loc] = 1

            for loc in self.env.mdp.get_dish_dispenser_locations():
                state_mask_dict["dish_disp_loc"][loc] = 1

            for loc in self.env.mdp.get_serving_locations():
                state_mask_dict["serve_loc"][loc] = 1
            
            if 'onion' in overcooked_state._all_orders[0].ingredients:
                state_mask_dict["next_soup"] += state_mask_dict["onion_disp_loc"]
            if 'tomato' in overcooked_state._all_orders[0].ingredients:
                state_mask_dict["next_soup"] += state_mask_dict["tomato_disp_loc"]

            # PLAYER LAYERS
            for i, player in enumerate(overcooked_state.players):
                player_orientation_idx = Direction.DIRECTION_TO_INDEX[player.orientation]
                state_mask_dict["player_{}_loc".format(i)] = make_layer(player.position, 1)
                state_mask_dict["player_{}_orientation_{}".format(i, player_orientation_idx)] = make_layer(player.position, 1)
            # COMPRESS PLAYER POSITION & ORIENTATION TOGETHER
            state_mask_dict["player_loc"] = sum([state_mask_dict["player_{}_loc".format(i)] for i, _ in enumerate(overcooked_state.players)])
            for d in Direction.ALL_DIRECTIONS:
                state_mask_dict["player_orientation_{}".format(Direction.DIRECTION_TO_INDEX[d])] = sum([state_mask_dict["player_{}_orientation_{}".format(i, Direction.DIRECTION_TO_INDEX[d])] for i, _ in enumerate(overcooked_state.players)])
            for feature in ordered_player_features:
                state_mask_dict.pop(feature)
                LAYERS.remove(feature)
            
            # OBJECT & STATE LAYERS
            for obj in all_objects:
                if obj.name == "soup":
                    # removed the next line because onion doesn't have to be in all the soups?
                    # if Recipe.ONION in obj.ingredients:
                    # get the ingredients into a {object: number} dictionary
                    ingredients_dict = Counter(obj.ingredients)
                    # assert "onion" in ingredients_dict.keys()
                    if obj.position in self.env.mdp.get_pot_locations():
                        if obj.is_idle:
                            # onions_in_pot and tomatoes_in_pot are used when the soup is idling, and ingredients could still be added
                            state_mask_dict["onions_in_pot"] += make_layer(obj.position, ingredients_dict["onion"])
                            state_mask_dict["tomatoes_in_pot"] += make_layer(obj.position, ingredients_dict["tomato"])
                        else:
                            state_mask_dict["onions_in_soup"] += make_layer(obj.position, ingredients_dict["onion"])
                            state_mask_dict["tomatoes_in_soup"] += make_layer(obj.position, ingredients_dict["tomato"])
                            state_mask_dict["soup_cook_time_remaining"] += make_layer(obj.position, obj.cook_time - obj._cooking_tick)
                            if obj.is_ready:
                                state_mask_dict["soup_done"] += make_layer(obj.position, 1)

                    else:
                        # If player soup is not in a pot, treat it like a soup that is cooked with remaining time 0
                        state_mask_dict["onions_in_soup"] += make_layer(obj.position, ingredients_dict["onion"])
                        state_mask_dict["tomatoes_in_soup"] += make_layer(obj.position, ingredients_dict["tomato"])
                        state_mask_dict["soup_done"] += make_layer(obj.position, 1)

                elif obj.name == "dish":
                    state_mask_dict["dishes"] += make_layer(obj.position, 1)
                elif obj.name == "onion":
                    state_mask_dict["onions"] += make_layer(obj.position, 1)
                elif obj.name == "tomato":
                    state_mask_dict["tomatoes"] += make_layer(obj.position, 1)
                else:
                    raise ValueError("Unrecognized object")
            # Stack of all the state masks, order decided by order of LAYERS
            return state_mask_dict, LAYERS

        # NOTE: Currently not very efficient, a decent amount of computation repeated here
        num_players = len(overcooked_state.players)
        state_mask_dict, LAYERS = process_state()
        final_obs_for_players = {i: process_for_player(i, state_mask_dict, LAYERS) for i in range(num_players)}
        return final_obs_for_players


    def step(self, action_dict):
        """
        action:
            (agent with index self.agent_idx action, other agent action)
            is a tuple with the joint action of the primary and secondary agents in index format

        returns:
            observation: formatted to be standard input for self.agent_idx's policy
        """
        action = [int(action_dict[key]) for key in action_dict] # convert from float to int
        joint_agent_action_info = {i:{} for i in self.agents}
        assert all(
            self.action_space[agent].contains(action[agent])
            for agent in action_dict
        ), "%r (%s) invalid" % (action, type(action))

        joint_action = [Action.INDEX_TO_ACTION[a] for a in action]
        # take a step in the current base environment
        assert not self.env.is_done()
        if self.use_phi:
            next_state, mdp_infos = self.env.mdp.get_state_transition(self.env.state, joint_action, display_phi=True, motion_planner=self.env.mp)

        else:
            next_state, mdp_infos = self.env.mdp.get_state_transition(self.env.state, joint_action, display_phi=False, motion_planner=self.env.mp)
        # Update game_stats 
        self.env._update_game_stats(mdp_infos)

        # Update state and done
        self.env.state = next_state
        done = self.env.is_done()
        info = self.env._prepare_info_dict(joint_agent_action_info, mdp_infos)
        
        if done: self.env._add_episode_info(info)

        timestep_sparse_reward = mdp_infos["sparse_reward_by_agent"]
        # return (next_state, timestep_sparse_reward, done, env_info)
        if self.use_phi:
            potential = info["phi_s_prime"] - info["phi_s"]
            dense_reward = [potential for _ in self.agents]
        else:
            dense_reward = info["shaped_r_by_agent"] #[0]*self.n_agents #info["shaped_r_by_agent"]

        obs = self.get_obs(next_state)
        # print(dense_reward)
        # rewards = {i : timestep_sparse_reward[i] + self.reward_shaping_factor * dense_reward[i] for i in self.agents}
        rewards = {i: timestep_sparse_reward[i] for i in self.agents}

        dones = {"__all__": done}
        infos = {i: info for i in self.agents}
        return obs, rewards, dones, infos


    def reset(self, regen_mdp=True):
        """
        When training on individual maps, we want to randomize which agent is assigned to which
        starting location, in order to make sure that the agents are trained to be able to
        complete the task starting at either of the hardcoded positions.

        NOTE: a nicer way to do this would be to just randomize starting positions, and not
        have to deal with randomizing indices.
        """
        self.env.reset(regen_mdp) # state initialized from class OvercookedState
        # self.curr_agents = self._populate_agents()
        obs = self.get_obs(self.env.state)
        events_dict = { k : [ [] for _ in range(self.env.mdp.num_players) ] for k in EVENT_TYPES }
        rewards_dict = {
            "cumulative_sparse_rewards_by_agent": np.array([0.] * self.env.mdp.num_players),
            "cumulative_shaped_rewards_by_agent": np.array([0.] * self.env.mdp.num_players)
        }
        self.env.game_stats = {**events_dict, **rewards_dict}
        return obs
    

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


    def set_reward_shaping_factor(self, factor):
        self.reward_shaping_factor = factor

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

    def get_rollouts(self, agent_pair, num_games, display=False, dir=None, final_state=False, display_phi=False,
                     display_until=np.Inf, metadata_fn=None, metadata_info_fn=None, info=True):
        """
        Simulate `num_games` number rollouts with the current agent_pair and returns processed 
        trajectories.

        Returning excessive information to be able to convert trajectories to any required format 
        (baselines, stable_baselines, etc)

        metadata_fn returns some metadata information computed at the end of each trajectory based on
        some of the trajectory data.

        NOTE: this is the standard trajectories format used throughout the codebase
        """
        trajectories = { k:[] for k in DEFAULT_TRAJ_KEYS }
        metadata_fn = (lambda x: {}) if metadata_fn is None else metadata_fn
        metadata_info_fn = (lambda x: "") if metadata_info_fn is None else metadata_info_fn
        range_iterator = tqdm.trange(num_games, desc="", leave=True) if info else range(num_games)
        for i in range_iterator:
            agent_pair.set_mdp(self.env.mdp)

            rollout_info = self.env.run_agents(agent_pair, display=False, dir=dir, include_final_state=final_state,
                                           display_phi=display_phi, display_until=display_until)
            trajectory, time_taken, tot_rews_sparse, _tot_rews_shaped = rollout_info
            obs, actions, rews, dones, infos = trajectory.T[0], trajectory.T[1], trajectory.T[2], trajectory.T[3], trajectory.T[4]
            trajectories["ep_states"].append(obs)
            trajectories["ep_actions"].append(actions)
            trajectories["ep_rewards"].append(rews)
            trajectories["ep_dones"].append(dones)
            trajectories["ep_infos"].append(infos)
            trajectories["ep_returns"].append(tot_rews_sparse)
            trajectories["ep_lengths"].append(time_taken)
            trajectories["mdp_params"].append(self.env.mdp.mdp_params)
            trajectories["env_params"].append(self.env.env_params)
            trajectories["metadatas"].append(metadata_fn(rollout_info))
            

            # we do not need to regenerate MDP if we are trying to generate a series of rollouts using the same MDP
            # Basically, the FALSE here means that we are using the same layout and starting positions
            # (if regen_mdp == True, resetting will call mdp_gen_fn to generate another layout & starting position)
            self.reset(regen_mdp=False)
            agent_pair.reset()

            if info:
                mu, se = mean_and_std_err(trajectories["ep_returns"])
                description = "Avg rew: {:.2f} (std: {:.2f}, se: {:.2f}); avg len: {:.2f}; ".format(
                    mu, np.std(trajectories["ep_returns"]), se, np.mean(trajectories["ep_lengths"]))
                description += metadata_info_fn(trajectories["metadatas"])
                range_iterator.set_description(description)
                range_iterator.refresh()

        # Converting to numpy arrays
        trajectories = {k: np.array(v) for k, v in trajectories.items()}

        # Merging all metadata dictionaries, assumes same keys throughout all
        trajectories["metadatas"] = append_dictionaries(trajectories["metadatas"])

        # TODO: should probably transfer check methods over to Env class
        from overcooked_ai_py.agents.benchmarking import AgentEvaluator
        AgentEvaluator.check_trajectories(trajectories, verbose=info)
        return trajectories


def resolve_interacts(self, new_state, joint_action, events_infos):
    """
    NOTE Assign rewards based on events of different agents, determining the role of trained policies
    Resolve any INTERACT actions, if present.

    Currently if two players both interact with a terrain, we resolve player 1's interact 
    first and then player 2's, without doing anything like collision checking.
    """
    # print(f"soups next: {new_state._all_orders}")
    pot_states = self.get_pot_states(new_state)
    # We divide reward by agent to keep track of who contributed
    sparse_reward, shaped_reward = [0] * self.num_players, [0] * self.num_players 

    for player_idx, (player, action) in enumerate(zip(new_state.players, joint_action)):
        # print(f"deal with action of {player_idx}, {action}")
        if action != Action.INTERACT:
            continue

        pos, o = player.position, player.orientation
        i_pos = Action.move_in_direction(pos, o)
        terrain_type = self.get_terrain_type_at_pos(i_pos)

        # NOTE: we always log pickup/drop before performing it, as that's
        # what the logic of determining whether the pickup/drop is useful assumes
        if terrain_type == 'X':

            if player.has_object() and not new_state.has_object(i_pos):
                obj_name = player.get_object().name
                self.log_object_drop(events_infos, new_state, obj_name, pot_states, player_idx)

                # Drop object on counter
                obj = player.remove_object()
                new_state.add_object(obj, i_pos)
                
            elif not player.has_object() and new_state.has_object(i_pos):
                obj_name = new_state.get_object(i_pos).name
                self.log_object_pickup(events_infos, new_state, obj_name, pot_states, player_idx)

                # Pick up object from counter
                obj = new_state.remove_object(i_pos)
                player.set_object(obj)
                

        elif terrain_type == 'O' and player.held_object is None:
            self.log_object_pickup(events_infos, new_state, "onion", pot_states, player_idx)

            # Onion pickup from dispenser
            obj = ObjectState('onion', pos)
            player.set_object(obj)
            if player_idx != 0: 
                # print(f"{player_idx} pickup the onion, penalty!!!!")
                shaped_reward[player_idx] -= 3
            # if 'onion' in new_state._all_orders[0]: 
                # print(f'onion in current order, {player_idx} picks it')
                    # sparse_reward[player_idx] += 2

        elif terrain_type == 'T' and player.held_object is None:
            # Tomato pickup from dispenser
            player.set_object(ObjectState('tomato', pos))
            if player_idx != 1: 
                # print(f"{player_idx} pickup the tomato, penalty!!!!")
                shaped_reward[player_idx] -= 3
            # if 'tomato' in new_state._all_orders[0]:
                # print(f'tomato in current order, {player_idx} picks it')
                    # sparse_reward[player_idx] += 2

        elif terrain_type == 'D' and player.held_object is None:
            self.log_object_pickup(events_infos, new_state, "dish", pot_states, player_idx)

            # Give shaped reward if pickup is useful
            # if self.is_dish_pickup_useful(new_state, pot_states):
                # shaped_reward[player_idx] += self.reward_shaping_params["DISH_PICKUP_REWARD"]

            # Perform dish pickup from dispenser
            obj = ObjectState('dish', pos)
            player.set_object(obj)

        elif terrain_type == 'P' and not player.has_object():
            # Cooking soup
            if self.soup_to_be_cooked_at_location(new_state, i_pos):
                soup = new_state.get_object(i_pos)
                soup.begin_cooking()
                print(f"{player_idx} starts soup on first order to cook")
                if player_idx == 2: 
                    # print(f"{player_idx} cooks, penalty!!!!")
                    shaped_reward[player_idx] -= 3
                else:
                    if soup.recipe == new_state._all_orders[0]:
                        shaped_reward[player_idx] += 0.5
        
        elif terrain_type == 'P' and player.has_object():

            if player.get_object().name == 'dish' and self.soup_ready_at_location(new_state, i_pos):
                self.log_object_pickup(events_infos, new_state, "soup", pot_states, player_idx)
                print(f"{player_idx} pick up cooked soup")
                if player_idx == 2:
                    shaped_reward[player_idx] += 1

                # Pick up soup
                player.remove_object() # Remove the dish
                obj = new_state.remove_object(i_pos) # Get soup
                player.set_object(obj)
                # shaped_reward[player_idx] += self.reward_shaping_params["SOUP_PICKUP_REWARD"]

            elif player.get_object().name in Recipe.ALL_INGREDIENTS:
                # Adding ingredient to soup

                if not new_state.has_object(i_pos):
                    # Pot was empty, add soup to it
                    new_state.add_object(SoupState(i_pos, ingredients=[]))

                # Add ingredient if possible
                soup = new_state.get_object(i_pos)
                def need_ingredient(soup, ingredient):
                    # print(f"ingredient: {ingredient.name}, soup {new_state._all_orders[0]}")
                    if ingredient.name in new_state._all_orders[0] and \
                    soup.ingredients.count(ingredient.name) < new_state._all_orders[0].ingredients.count(ingredient.name):
                        return True
                    else:
                        return False
                        
                if not soup.is_full:
                    old_soup = soup.deepcopy()
                    obj = player.remove_object()
                    if need_ingredient(soup, ingredient=obj):
                        print(f"{player_idx} add {obj.name} which is useful to current order {new_state._all_orders[0]}")
                        shaped_reward[player_idx] += 0.5
                    else:
                        shaped_reward[player_idx] -= 3
                    if player_idx == 2:
                        shaped_reward[player_idx] -= 3
                    soup.add_ingredient(obj)
                    # shaped_reward[player_idx] += self.reward_shaping_params["PLACEMENT_IN_POT_REW"]
                    # Log potting
                    self.log_object_potting(events_infos, new_state, old_soup, soup, obj.name, player_idx)
                    if obj.name == Recipe.ONION:
                        events_infos['potting_onion'][player_idx] = True

        elif terrain_type == 'S' and player.has_object():
            obj = player.get_object()
            if obj.name == 'soup':
                order_flag = 1 if (obj.recipe == new_state._all_orders[0]) else -1
                if order_flag == 1:
                    new_state._all_orders.pop(0)
                    new_state._all_orders.append(obj.recipe)
                delivery_rew = self.deliver_soup(new_state, player, obj)
                if player_idx != 2: 
                    shaped_reward[player_idx] -= 3
                # for player_idx, player in enumerate(new_state.players):
                        # sparse_reward[player_idx] += order_flag*delivery_rew
                if delivery_rew == 2:
                    for player_idx, player in enumerate(new_state.players):
                        sparse_reward[player_idx] += order_flag*delivery_rew*2

                # Log soup delivery
                events_infos['soup_delivery'][player_idx] = True
                print(f"soup delivered by {player_idx}, ",obj.ingredients, order_flag*delivery_rew)
                if order_flag == 1:
                    print(f"soups next: {new_state._all_orders}")
    return sparse_reward, shaped_reward


def get_state_transition(self, state, joint_action, display_phi=False, motion_planner=None):
    """Gets information about possible transitions for the action.

    Returns the next state, sparse reward and reward shaping.
    Assumes all actions are deterministic.

    NOTE: Sparse reward is given only when soups are delivered, 
    shaped reward is given only for completion of subgoals 
    (not soup deliveries).
    """
    events_infos = { event : [False] * self.num_players for event in EVENT_TYPES }
    assert not self.is_terminal(state), "Trying to find successor of a terminal state: {}".format(state)
    for action, action_set in zip(joint_action, self.get_actions(state)):
        if action not in action_set:
            raise ValueError("Illegal action %s in state %s" % (action, state))
    
    new_state = state.deepcopy()
    new_state._all_orders = state._all_orders # preserve the sequence of orders

    # Resolve interacts first
    sparse_reward_by_agent, shaped_reward_by_agent = self.resolve_interacts(new_state, joint_action, events_infos)

    assert new_state.player_positions == state.player_positions
    assert new_state.player_orientations == state.player_orientations
    
    # Resolve player movements
    self.resolve_movement(new_state, joint_action)

    # Finally, environment effects
    self.step_environment_effects(new_state)

    # Additional dense reward logic
    # shaped_reward += self.calculate_distance_based_shaped_reward(state, new_state)
    infos = {
        "event_infos": events_infos,
        "sparse_reward_by_agent": sparse_reward_by_agent,
        "shaped_reward_by_agent": shaped_reward_by_agent,
    }
    if display_phi:
        assert motion_planner is not None, "motion planner must be defined if display_phi is true"
        infos["phi_s"] = self.potential_function(state, motion_planner)
        infos["phi_s_prime"] = self.potential_function(new_state, motion_planner)
    return new_state, infos

