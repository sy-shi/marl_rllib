from multigrid.gym_multigrid.ray_env_wrapper import MAUSARGame
from overcooked.overcookedEnv import OvercookedMultiAgent
from ray.tune import register_env

def register_envs():
    register_env("usar", lambda env_config: MAUSARGame(env_config))
    register_env("overcooked", lambda env_config: OvercookedMultiAgent(env_config))