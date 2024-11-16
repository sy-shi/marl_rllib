import torch
from model.actor_critic import ActorCritic
# from model.option_critic_model import OptionActorCritic
from ray import cloudpickle
from pathlib import Path
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from typing import Dict, Tuple
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import AgentID, PolicyID
from ray.rllib.policy import Policy
import numpy as np


def load_torch_model(import_path, action_space, observation_space, config):
    # Assume the teacher's model is the same as the students'. All customzied CC model
    if not config.get("custom_model", None):
        print ('-----------------------------------------------------------')
        raise("teacher model is not specified")
    # "custom_model" is "cc_model"
    elif config["custom_model"] == "cc_model":
        # loaded_model = TorchCentralizedCriticModel(obs_space = observation_space, 
        loaded_model = ActorCritic(obs_space = observation_space, 
                                   action_space = action_space,
                                   num_outputs = action_space.n,
                                   model_config = config,  # model_config of the agent
                                   name = "cc_model",  # is the name appropriate?
                                   **config["custom_model_config"])# central_vf_dim=config["custom_model_config"]["central_vf_dim"])# not called by rllib, can't neglect  
    

        checkpoint_path = Path(import_path).expanduser()
        with checkpoint_path.open('rb') as f:
            checkpoint = cloudpickle.load(f)
            filtered_state_dict = {k: v for k, v in checkpoint['weights'].items() if not k.startswith('q_input_layers')}
            # filtered_state_dict = {k: v for k, v in checkpoint['weights'].items() if k.startswith('input_layers' or 'critic_input_layers' or 'actor_layers' or 'critic_layers')}
            loaded_model.load_state_dict(convert_to_torch_tensor(filtered_state_dict,device=next(loaded_model.parameters()).device),
                                         strict=False)
        # if import_path is not None:
        # loaded_model.load_state_dict(torch.load(import_path, map_location=next(loaded_model.parameters()).device))
        loaded_model.eval()
    elif config["custom_model"] == "option_model":
        pass
        # loaded_model = OptionActorCritic(obs_space = observation_space,
        #                                  action_space = action_space,
        #                                  num_outputs = action_space.n,
        #                                  model_config = config,
        #                                  name = "option_model",
        #                                  **config["custom_model_config"])
        # checkpoint_path = Path(import_path).expanduser()
        # with checkpoint_path.open('rb') as f:
        #     checkpoint = cloudpickle.load(f)
        #     filtered_state_dict = {k: v for k, v in checkpoint['weights'].items() if not k.startswith('q_input_layers')}
        #     # filtered_state_dict = {k: v for k, v in checkpoint['weights'].items() if k.startswith('input_layers' or 'critic_input_layers' or 'actor_layers' or 'critic_layers')}
        #     loaded_model.load_state_dict(convert_to_torch_tensor(filtered_state_dict,device=next(loaded_model.parameters()).device),
        #                                  strict=False)
        # loaded_model.eval()
    
    return loaded_model


class ModelWrapper():
    def __init__(self, model=None):
        self.model = model

    def load(self, import_path, action_space, observation_space, config):
        self.model = load_torch_model(import_path, action_space, observation_space, config)     



def compute_is_ratio(batch: Dict[AgentID, Tuple[Policy, SampleBatch]]):
    sample_num = batch[0][1]["infos"].shape[0]
    teacher_action_logp = np.array([batch[0][1]["infos"][i]["teacher_action_logp"] for i in range(sample_num)])
    student_action_logp = batch[1][1][SampleBatch.ACTION_LOGP]
    teacher_is_ratio = np.exp(teacher_action_logp - student_action_logp)
    teacher_action_idx_find = np.array([1 if batch[0][1]["infos"][i]["issue advice"] is True else 0 for i in range(sample_num)])
    teacher_is_ratio = np.where(teacher_action_idx_find == 1, 1, teacher_is_ratio)
    return teacher_is_ratio
    