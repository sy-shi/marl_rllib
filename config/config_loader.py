import collections.abc
import os
import pathlib
import yaml

from infrastructure.callbacks import LoggingCallbacks
from ray import tune
import torch
import gym.spaces
import numpy as np


class ConfigLoader():

    def load_config(name, use_hpo = False):
        names = name.split("/")

        path = os.path.join("config", names[0] + ".yaml")
        configs = yaml.safe_load(open(path))
        
        configs = ConfigLoader._process_config(configs)
        print(configs)
        configs = ConfigLoader._initialize_configs(configs)

        config = {"BASE_CONFIG":configs["BASE_CONFIG"], "EAT_AGENT_CONFIG":configs["EAT_AGENT_CONFIG"]}

        if len(names) > 1:
            config.update(configs[names[1]])

        if "HPO_CONFIG" in configs and use_hpo:
            # config.update(configs["HPO_CONFIG"])
            config = ConfigLoader._update_config(config, configs["HPO_CONFIG"])
        # print(config)
        return config
    load_config = staticmethod(load_config)


    def _process_config(config):
        for key, value in config.items():
            if isinstance(value, dict):
                config[key] = ConfigLoader._process_config(value)

            elif isinstance(value, str):
                if len(value) >= 5 and value[:5] == "tune.":
                    config[key] = eval(value)

                elif len(value) >= 5 and value[:5] == "/$SRC":
                    # Note: could use string replace here instead of only checking prefix but am not sure it would ever be necessary
                    config[key] = os.path.join(pathlib.Path(__file__).parent.parent.resolve(), value[5:])
            elif isinstance(value, list):
                for i, content in enumerate(value):
                    if isinstance(content, str):
                        if len(content) >= 5 and content[:5] == "$SRC/":
                            config[key][i] = os.path.join(pathlib.Path(__file__).parent.parent.resolve(), content[5:])
        return config
    _process_config = staticmethod(_process_config)


    def _initialize_configs(config):
        # config["ENV_CONFIG"]["model"] = config["MODEL_CONFIG"]

        config["BASE_CONFIG"]["env_config"] = config["ENV_CONFIG"]
        config["BASE_CONFIG"]["model"] = config["MODEL_CONFIG"]
        config["BASE_CONFIG"]["callbacks"] = LoggingCallbacks
        config["BASE_CONFIG"]["num_gpus"] = 1 if torch.cuda.is_available() else 0
        config["EAT_AGENT_CONFIG"]["teacher_model_config"] = config["MODEL_CONFIG"]
        config["BASE_CONFIG"]["env_config"]["observation_space"] = gym.spaces.Dict({
            "image": gym.spaces.Box(0, float('inf'), shape=(23,6,5), dtype=np.float32),
            "action_mask": gym.spaces.Box(0.0, 1.0, shape=(6,),dtype=np.float32),
        })
        config["BASE_CONFIG"]["env_config"]["action_space"] = gym.spaces.Discrete(6)
        # config["EAT_AGENT_CONFIG"]["observation_space"] = config["BASE_CONFIG"]["env_config"]["observation_space"]
        # config["EAT_AGENT_CONFIG"]["action_space"] = config["BASE_CONFIG"]["env_config"]["action_space"]

        return config
    _initialize_configs = staticmethod(_initialize_configs)


    def _update_config(d, u):
        for k, v in u.items():
            if isinstance(d, collections.abc.Mapping):
                if isinstance(v, collections.abc.Mapping):
                    r = ConfigLoader._update_config(d.get(k, {}), v)
                    d[k] = r
                else:
                    d[k] = u[k]
            else:
                d = {k: u[k]}
        return d
    _update_config = staticmethod(_update_config)
