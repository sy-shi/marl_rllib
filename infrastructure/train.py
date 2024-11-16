import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.tune.logger import UnifiedLogger
from infrastructure.rollout import rollout_episodes
from infrastructure.callbacks import CustomCheckpointCallback
from infrastructure.env_loader import register_envs
from gym_multigrid.ray_env_creator import env_creator
from overcooked.rllib_utils import evaluate
from infrastructure.agent_trainer import get_trainer
from model.actor_critic import ActorCritic
from config.config_loader import ConfigLoader

from datetime import datetime
import tempfile
import os


def logger_creater(args):
    "create logger to a customized path"
    timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    logdir_prefix = "{}_{}".format(args.name, timestr)
    dir = './data'
    if args.mode == "train":
        dir = dir + "/train"
    if args.mode == "eval":
        dir = dir + "/eval"
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)
    logdir = tempfile.mkdtemp(prefix=logdir_prefix, dir=dir)
    def default_logger_creator(config):
        """Creates a Unified logger with the default prefix."""
        return UnifiedLogger(config, logdir, loggers=None)
    return default_logger_creator


def train(args):
    Config = ConfigLoader.load_config(args.config)
    print(Config)
    Config["BASE_CONFIG"]["multiagent"] = {
        "policies": {
            "pol" + str(i+1):(
                None, # policy spec
                None, # observation spec
                None, # action spec
                {
                    "name": "pol" + str(i+1),
                    "framework": "torch",
                    "model": Config["BASE_CONFIG"]["model"],
                },
            ) for i in range(Config["BASE_CONFIG"]["env_config"]["n_agents"])
        },
        "policy_mapping_fn": (lambda aid, **kwargs: "pol" + str(aid+1)),
        "policies_to_train": [f'pol{i+1}'for i in range(Config["BASE_CONFIG"]["env_config"]["n_agents"])]
    }
    
    os.environ["CUDA_VISIBLE_DEVICES"] ="0" #"5,6,7"
    assert args.model_path is not None # need to specify a path to store trained models
    ray.init()
    register_envs()
    ModelCatalog.register_custom_model(
        "cc_model",
        ActorCritic
    )

    if args.mode == "train":
        trainer = get_trainer()(config=Config["BASE_CONFIG"], logger_creator=logger_creater(args))
        if args.load_model == True:
            assert os.path.exists(args.restore_path)
            print("load model to keep training...")
            trainer.restore(args.restore_path)
            print("start training...")
        for i in range(args.stop_iters):
            print("training ...")
            result = trainer.train()
            print('*******************************************************')
            print('training iteration: ', i)
            print('average training reward: ', result['episode_reward_mean'])
            print('total timesteps: ', result['timesteps_total'])
            # print("Evaluated {} episodes. Average reward: {}. Average num steps: {}".format(config.algo_config["evaluation_num_episodes"], result['evaluation']['episode_reward_mean'], result['evaluation']['episode_len_mean']))
            if (i % args.ckpt_freq == 0 or i == args.stop_iters-1):
                print("save model parameters to... ")
                ckpt_path = trainer.save(args.model_path)
                print(ckpt_path)

    if args.mode == "tune":
        restore = args.restore_path if args.load_model else None
        assert os.path.exists(args.restore_path) if restore else True
        
        tune.run(
            get_trainer(),
            name = args.name,
            stop = {"timesteps_total":args.timesteps_total},#"training_iteration": args.stop_iters, },
            config = Config["BASE_CONFIG"],
            local_dir=f"./data/{args.name}/",
            verbose=3,                    # set to enable different extent of logging.
            checkpoint_freq=args.ckpt_freq,
            keep_checkpoints_num=5,
            checkpoint_at_end=True,
            restore = restore,
            callbacks = [CustomCheckpointCallback(args.model_path)]
        )
        ray.shutdown()

    if args.mode == "eval":
        trainer = get_trainer()(config=Config["BASE_CONFIG"], logger_creator=logger_creater(args))
        trainer.restore(args.model_path)
        models = []
        policies = []
        for i,policy_id in enumerate(Config["BASE_CONFIG"]["multiagent"]["policies"].keys()):
            models.append(trainer.get_policy(policy_id).model)
            policies.append(trainer.get_policy(policy_id))
            models[i].eval()
        if args.config == 'overcooked_':
            evaluate(Config["BASE_CONFIG"]["env_config"], policies, num_episodes= args.eval_episodes, display=args.render, ifsave=args.save_render, save='./data/')
        elif args.config == 'usar_':
            env = env_creator(Config["BASE_CONFIG"]["env_config"])
            reward, steps = rollout_episodes(models, env, num_episodes= args.eval_episodes, max_steps=Config["BASE_CONFIG"]["env_config"]["max_steps"], \
                                            render=args.render, save_rollouts = False, save_render=args.save_render, render_name=args.name)
            print("Evaluated {} episodes. Average reward: {}. Average num steps: {}".format(args.eval_episodes, reward, steps))