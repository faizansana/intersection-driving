import argparse
import datetime
import os
import pathlib
import shutil
import sys

import gymnasium as gym
import yaml
from sb3_contrib import RecurrentPPO
from stable_baselines3 import DDPG, DQN, PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.monitor import Monitor

from train_config import SaveOnBestTrainingRewardCallback, CustomCombinedExtractor, SaveLatestModelCallback


def parse_arguments():
    argparser = argparse.ArgumentParser(description="Train Agent")
    argparser.add_argument(
        "-m", "--model",
        help="Model to train",
        default="DDPG",
        metavar="NAME",
        choices=["DDPG", "PPO", "RecurrentPPO", "SAC", "DQN"])
    argparser.add_argument(
        "-t", "--timesteps",
        help="Number of timesteps to train for",
        default=100000,
        metavar="N",
        type=int)
    argparser.add_argument(
        "-v", "--verbose",
        help="Verbosity level",
        default=0,
        metavar="N",
        type=int)
    argparser.add_argument(
        "-c", "--carla-host",
        help="IP Address of CARLA host",
        default="carla_server",
        metavar="IP",
        type=str)
    argparser.add_argument(
        "-p", "--carla-port",
        help="Port of CARLA host",
        default="2000",
        metavar="PORT",
        type=int)
    argparser.add_argument(
        "--config-file",
        help="Path to config file",
        default="./custom_carla_gym/src/config_discrete.yaml",
        metavar="PATH",
        type=str)
    argparser.add_argument(
        "--model-path",
        help="Path to load model for RETRAINING",
        default=None,
        metavar="PATH",
        type=str)

    return argparser


def train(model: BaseAlgorithm, timesteps: int, model_dir: os.path, log_dir: os.path, check_freq: int = 2000, verbose: int = 0) -> None:
    """Train an agent on a given environment for a given number of timesteps"""
    # Create callback
    best_model_callback = SaveOnBestTrainingRewardCallback(check_freq=check_freq, log_dir=log_dir, save_path=model_dir, verbose=verbose)
    latest_model_callback = SaveLatestModelCallback(check_freq=10000, save_path=model_dir, verbose=verbose)
    # Check if retraining
    if model.num_timesteps > 0:
        timesteps = timesteps - model.num_timesteps
        # If timesteps is negative, do not retrain
        if timesteps < 0:
            return
    # Train
    model.learn(total_timesteps=timesteps, callback=[best_model_callback, latest_model_callback], progress_bar=True, reset_num_timesteps=False)
    # Save final model
    model.save(os.path.join(model_dir, "final_model"))


def setup_env(log_dir: str, carla_host: str, carla_port: int, config_file: str = "./custom_carla_gym/config.yaml") -> gym.Env:
    """Setup environment"""
    sys.path.append("./custom_carla_gym/src")
    from custom_carla_gym.src.carla_env_custom import CarlaEnv
    cfg = yaml.safe_load(open(config_file))
    env = CarlaEnv(cfg=cfg, host=carla_host, port=carla_port)

    if log_dir:
        # There seems to be a bug in Monitor, so this workaround is used.
        # If override_existing is set to False and monitor.csv does not exist, it creates the file but doesn't add header.
        # This is needed by the callback to function properly.
        if os.path.exists(os.path.join(log_dir, "monitor.csv")):
            env = Monitor(env, log_dir + "/", override_existing=False)
        else:
            env = Monitor(env, log_dir + "/")
    return env


def load_new_model(args: argparse.Namespace, log_dir: os.path, env: gym.Env):
    LEARNING_RATE = 1e-4
    BUFFER_SIZE = 100000
    LEARNING_STARTS = 1000
    GAMMA = 0.98
    TRAIN_FREQ = (1, "episode")
    GRADIENT_STEPS = -1
    VERBOSE = args.verbose
    policy_kwargs = {
        # "features_extractor_class": CustomCombinedExtractor,
        "net_arch": [400, 300]
    }
    policy = "MlpPolicy"

    # Setup Model
    if args.model == "DDPG":
        model = DDPG(policy,
                     env,
                     learning_rate=LEARNING_RATE,
                     policy_kwargs=policy_kwargs,
                     buffer_size=BUFFER_SIZE,
                     learning_starts=LEARNING_STARTS,
                     gamma=GAMMA,
                     train_freq=TRAIN_FREQ,
                     gradient_steps=GRADIENT_STEPS,
                     verbose=VERBOSE,
                     tensorboard_log=log_dir)
    elif args.model == "PPO":
        model = PPO(policy,
                    env,
                    learning_rate=LEARNING_RATE,
                    gamma=GAMMA,
                    policy_kwargs=policy_kwargs,
                    verbose=VERBOSE,
                    tensorboard_log=log_dir)
    elif args.model == "SAC":
        model = SAC(policy,
                    env,
                    learning_rate=LEARNING_RATE,
                    buffer_size=BUFFER_SIZE,
                    learning_starts=LEARNING_STARTS,
                    gamma=GAMMA,
                    policy_kwargs=policy_kwargs,
                    train_freq=TRAIN_FREQ,
                    gradient_steps=GRADIENT_STEPS,
                    verbose=VERBOSE,
                    tensorboard_log=log_dir)
    elif args.model == "RecurrentPPO":
        model = RecurrentPPO("MultiInputLstmPolicy",
                             env,
                             learning_rate=LEARNING_RATE,
                             gamma=GAMMA,
                             tensorboard_log=log_dir,
                             verbose=VERBOSE,
                             policy_kwargs=policy_kwargs
                             )
    elif args.model == "DQN":
        model = DQN(policy,
                    env,
                    learning_rate=LEARNING_RATE,
                    buffer_size=BUFFER_SIZE,
                    learning_starts=LEARNING_STARTS,
                    gamma=GAMMA,
                    train_freq=TRAIN_FREQ,
                    gradient_steps=GRADIENT_STEPS,
                    verbose=VERBOSE,
                    tensorboard_log=log_dir)

    return model


def load_current_model(model_dir: str, env: gym.Env):
    # Load model
    if "RecurrentPPO" in model_dir:
        model = RecurrentPPO.load(model_dir, env=env)
    elif "PPO" in model_dir:
        model = PPO.load(model_dir, env=env)
    elif "DDPG" in model_dir:
        model = DDPG.load(model_dir, env=env)
    elif "SAC" in model_dir:
        model = SAC.load(model_dir, env=env)
    elif "DQN" in model_dir:
        model = DQN.load(model_dir, env=env)
    else:
        raise ValueError("Model not supported")
    return model


def main():
    # Parse arguments
    args = parse_arguments().parse_args()

    if args.model_path:
        assert os.path.exists(args.model_path), "The model path does not exist"
        model_dir = os.path.dirname(args.model_path)
        print("Loading model from", args.model_path)
    else:
        # Create model and log directory
        model_dir = os.path.join("Training", "Models", args.model, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        print("TRAINING MODEL:", args.model)

    log_dir = os.path.join(model_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    print(f"------ {args.timesteps:,} ------")

    try:
        # Create environment
        env = setup_env(log_dir, args.carla_host, args.carla_port, args.config_file)

        # Load model
        if args.model_path:
            model = load_current_model(args.model_path, env)
        else:
            model = load_new_model(args, log_dir, env)

        # Train model
        train(model, args.timesteps, model_dir, log_dir, verbose=args.verbose)
    except Exception as e:
        print(e)
        # Delete model_dir if it was created and no .zip file is within that folder
        model_path = pathlib.Path(model_dir)
        zip_files = model_path.glob("*.zip")
        if not args.model_path and not any(zip_files):
            print("Deleting model directory:", model_path)
            shutil.rmtree(model_path)


if __name__ == "__main__":

    main()
