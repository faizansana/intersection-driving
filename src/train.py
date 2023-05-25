import argparse
import datetime
import os
import pathlib
import shutil
import sys

import gym
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
        "-e", "--env",
        help="Environment to train on. Path of env is appended",
        default="custom_carla_gym",
        metavar="NAME",
        choices=["custom_carla_gym", "gym_carla"])
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
        "--tm-port",
        help="Port of Traffic Manager server",
        default=8000,
        metavar="PORT",
        type=int)
    argparser.add_argument(
        "--config-file",
        help="Path to config file",
        default="./custom_carla_gym/config.yaml",
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
    latest_model_callback = SaveLatestModelCallback(check_freq=10, save_path=model_dir, verbose=verbose)
    # Train
    model.learn(total_timesteps=timesteps, callback=[best_model_callback, latest_model_callback], progress_bar=True, reset_num_timesteps=False)
    # Save final model
    model.save(os.path.join(model_dir, "final_model"))


def setup_env(env_name: str, log_dir: str, carla_host: str, tm_port: int = 8000, config_file: str = "./custom_carla_gym/config.yaml") -> gym.Env:
    """Setup environment"""
    if env_name == "custom_carla_gym":
        sys.path.append("./custom_carla_gym")
        from custom_carla_gym.carla_env_custom import CarlaEnv
        cfg = yaml.safe_load(open(config_file))
        env = CarlaEnv(cfg=cfg, host=carla_host, tm_port=tm_port)

    elif env_name == "gym_carla":
        sys.path.append("./gym_carla")
        from gym_carla.gym_carla.envs.carla_env import CarlaEnv
        params = {
            "number_of_vehicles": 40,
            "number_of_walkers": 10,
            "display_size": 256,  # screen size of bird-eye render
            "max_past_step": 1,  # the number of past steps to draw
            "dt": 0.1,  # time interval between two frames
            "discrete": False,  # whether to use discrete control space
            "discrete_acc": [-3.0, 0.0, 3.0],  # discrete value of accelerations
            "discrete_steer": [-0.2, 0.0, 0.2],  # discrete value of steering angles
            "continuous_accel_range": [-3.0, 3.0],  # continuous acceleration range
            "continuous_steer_range": [-0.3, 0.3],  # continuous steering angle range
            "ego_vehicle_filter": "vehicle.lincoln*",  # filter for defining ego vehicle
            "host": carla_host,  # which host to use
            "port": 2000,  # connection port
            "town": "Town03",  # which town to simulate
            "task_mode": "intersection",  # mode of the task, [random, roundabout (only for Town03)]
            "max_time_episode": 500,  # maximum timesteps per episode
            "max_waypt": 12,  # maximum number of waypoints
            "obs_range": 32,  # observation range (meter)
            "lidar_bin": 0.125,  # bin size of lidar sensor (meter)
            "d_behind": 12,  # distance behind the ego vehicle (meter)
            "out_lane_thres": 2.0,  # threshold for out of lane
            "desired_speed": 8,  # desired speed (m/s)
            "max_ego_spawn_times": 200,  # maximum times to spawn ego vehicle
            "display_route": True,  # whether to render the desired route
            "pixor_size": 64,  # size of the pixor labels
            "pixor": False  # whether to output PIXOR observation
        }
        env = gym.make('carla-v0', params=params, apply_api_compatibility=True)
    else:
        raise ValueError(f"Unknown environment: {env_name}")

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
        "features_extractor_class": CustomCombinedExtractor,
        "net_arch": [400, 300]
    }
    policy = "MultiInputPolicy"

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

    print("------", args.env, "------")
    print("------", args.timesteps, "------")

    try:
        # Create environment
        env = setup_env(args.env, log_dir, args.carla_host, args.tm_port, args.config_file)

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
