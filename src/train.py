import argparse
import datetime
import os
import sys

import gym
import yaml
from stable_baselines3 import DDPG, PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.monitor import Monitor

from train_config import SaveOnBestTrainingRewardCallback


def parse_arguments():
    argparser = argparse.ArgumentParser(description="Train Agent")
    argparser.add_argument(
        "-m", "--model",
        help="Model to train",
        default="DDPG",
        metavar="NAME",
        choices=["DDPG", "PPO"])
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

    return argparser


def train(model: BaseAlgorithm, env: gym.Env, timesteps: int, model_dir: os.path, log_dir: os.path, save_path: os.path, check_freq: int = 20, verbose: int = 0) -> None:
    """Train an agent on a given environment for a given number of timesteps"""
    # Create callback
    callback = SaveOnBestTrainingRewardCallback(check_freq=check_freq, log_dir=log_dir, save_path=save_path, verbose=verbose)
    # Train
    model.learn(total_timesteps=timesteps, callback=callback, progress_bar=True)
    # Save final model
    model.save(os.path.join(model_dir, "final_model"))


def setup_env(env_name: str, log_dir: str, carla_host: str, tm_port: int = 8000) -> gym.Env:
    """Setup environment"""
    if env_name == "custom_carla_gym":
        sys.path.append("./custom_carla_gym")
        from custom_carla_gym.carla_env_custom import CarlaEnv
        cfg = yaml.safe_load(open("./custom_carla_gym/config.yaml"))
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
        env = Monitor(env, log_dir + "/")
    return env


def main():
    # Parse arguments
    args = parse_arguments().parse_args()

    print("------", args.model, "------")
    print("------", args.env, "------")
    print("------", args.timesteps, "------")

    # Create model and log directory
    model_dir = os.path.join("Training", "Models", args.model, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    save_path = os.path.join(model_dir, f"{args.env}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")

    log_dir = os.path.join(model_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Create environment
    env = setup_env(args.env, log_dir, args.carla_host, args.tm_port)

    LEARNING_RATE = 1e-4
    BUFFER_SIZE = 100000
    LEARNING_STARTS = 1000
    GAMMA = 0.98
    TRAIN_FREQ = (1, "episode")
    GRADIENT_STEPS = -1
    VERBOSE = args.verbose

    # Setup Model
    if args.model == "DDPG":
        model = DDPG("MlpPolicy",
                     env,
                     learning_rate=LEARNING_RATE,
                     policy_kwargs={"net_arch": [400, 300]},
                     buffer_size=BUFFER_SIZE,
                     learning_starts=LEARNING_STARTS,
                     gamma=GAMMA,
                     train_freq=TRAIN_FREQ,
                     gradient_steps=GRADIENT_STEPS,
                     verbose=VERBOSE,
                     tensorboard_log=log_dir)
    elif args.model == "PPO":
        model = PPO("MlpPolicy",
                    env,
                    learning_rate=LEARNING_RATE,
                    gamma=GAMMA,
                    policy_kwargs={"net_arch": [400, 300]},
                    tensorboard_log=log_dir)

    # Train model
    train(model, env, args.timesteps, model_dir, log_dir, save_path, verbose=VERBOSE)


if __name__ == "__main__":

    main()
