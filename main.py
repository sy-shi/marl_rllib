import argparse

from infrastructure.train import train

# environment args
parser = argparse.ArgumentParser()

# model args

# training setting args
parser.add_argument("--config", type=str, default="usar")
parser.add_argument("--name", type=str, default=None, help="Desired name description")
parser.add_argument("--mode", choices=["train","tune","eval"], default="train")
parser.add_argument("--model_path", type=str, default=None, help="folder path to save or load model")
parser.add_argument("--stop_iters", type=int, default=100, help="Number of iterations to train.")
parser.add_argument("--timesteps_total", type=int, default=100000)
parser.add_argument("--ckpt_freq", type=int, default=20, help="model parameter checkpoint save frequency")
parser.add_argument("--load_model", type=bool, default=False, help="whether load model to keep training")
parser.add_argument("--restore_path", type=str, default=None, help="the path to restore checkpoint")

# evaluation setting args
parser.add_argument("--eval_episodes", default=10, type=int)
parser.add_argument('--render', default=False, type=bool, help="render during evaluation")
parser.add_argument('--save_render', default=False, type=bool, help='save renders during evaluation.')


if __name__ == '__main__':
    args = parser.parse_args()
    train(args=args)