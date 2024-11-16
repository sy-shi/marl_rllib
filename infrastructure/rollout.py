import numpy as np
import pickle

import torch
import torch.nn.functional as F

# from infrastructure.config import *

from matplotlib import animation
import matplotlib.pyplot as plt
import os
from ray.rllib.utils.numpy import convert_to_numpy
import copy


def get_action(torch_model, obs):
    if type(obs['image']) != torch.Tensor:
        obs['image'] = torch.tensor(obs['image'], requires_grad = False).to(next(torch_model.parameters()).device)
    if type(obs['action_mask']) != torch.Tensor:
        obs['action_mask'] = torch.tensor(obs['action_mask'], requires_grad = False).to(next(torch_model.parameters()).device)
    obs['image'] = torch.reshape(obs['image'], (1, 13, 13, 4))
    action_logit = torch_model({"obs": obs})[0][0]
    action = torch.distributions.Categorical(logits=action_logit).sample().cpu().detach().numpy()
    # log_action_prob = np.log(action_prob)
    # action = np.argmax(action_prob) # is argmax having issues?
    # action_prob = action_prob[0]
    # action = np.random.choice(np.arange(0, len(action_prob)), p=action_prob/sum(action_prob))
    # importance = np.max(log_action_prob) - np.min(log_action_prob)
    return action



def rollout_episode(loaded_model, env, max_steps = 500, flatten_obs = True, render = True, save_render = False):
    obs = env.reset()
    # print ('obs', obs)
    done = False

    observations, actions, rewards = [obs], [], []
    frames = []

    step_idx = 0
    total_reward = 0
    n_agents = 2
    while not done:     
        action_dict = {i:0 for i in range(n_agents)}
        for i in range(n_agents):
            action = get_action(loaded_model[i], copy.deepcopy(obs[i]))
            action_dict[i] = action
        obs, reward_dict, dones, info = env.step(action_dict)
        reward = sum(reward_dict[i] for i in range(n_agents))
        
        observations.append(obs)
        actions.append(action_dict)
        rewards.append(reward_dict)

        done = dones
        step_idx += 1
        total_reward += reward

        if render:
            img = env.render(highlight=False)
            if save_render == True:
                frames.append(img)
        if step_idx > max_steps:
            break

    env.reset()

    states = [observations, actions, rewards]
    return states, step_idx, total_reward, frames


def rollout_episodes(loaded_model, env, num_episodes = 1, save_rollouts = False, render = False, save_render = False, render_name = None, max_steps = 500):
    all_episode_states = []
    num_steps = []
    rewards = []
    Frames = []

    for _ in range(num_episodes):
        states, steps, reward, frames = rollout_episode(loaded_model, env, max_steps = max_steps, render=render, save_render = save_render)
        all_episode_states.append(states)
        num_steps.append(steps)
        rewards.append(reward)
        Frames.append(frames)

    if save_rollouts:
        with open('model_trajectories_treasure.pkl', 'wb') as f:
            pickle.dump(all_episode_states, f)
    
    if save_render:
        dir = 'videos/' + render_name + '/'
        if not os.path.exists(dir):
            os.makedirs(dir, exist_ok=True)
        assert render_name is not None
        print("saving rollout video ...")
        for i in range(len(Frames)):
            file_name = "eval_episode_" + str(i) + ".gif"
            save_frames_as_gif(Frames[i], dir, file_name)
        
    return np.mean(rewards), np.mean(num_steps)


def rollout_steps(loaded_model, env, num_steps = 100, max_steps = 500, flatten_obs = True):
    steps_collected = 0

    all_episode_states = []

    while steps_collected < num_steps:
        states, steps, _, _ = rollout_episode(loaded_model, env, max_steps = max_steps, flatten_obs = flatten_obs)
        all_episode_states.extend(states)
        steps_collected += steps

    all_episode_states = all_episode_states[:num_steps]

    return all_episode_states


def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=10)
    plt.close()