import argparse

import numpy as np
import pygame
from sb3_contrib import RecurrentPPO
from stable_baselines3 import DDPG, DQN, PPO, SAC
from tqdm import tqdm

from train import setup_env


def parse_arguments():
    argparser = argparse.ArgumentParser(description="Test Agent")
    argparser.add_argument(
        "-m", "--model-path", help="Path to model to test", metavar="PATH", type=str
    )
    argparser.add_argument(
        "-v", "--verbose", help="Verbosity level", default=0, metavar="N", type=int
    )
    argparser.add_argument(
        "-c",
        "--carla-host",
        help="IP Address of CARLA host",
        default="carla_server",
        metavar="IP",
        type=str,
    )
    argparser.add_argument(
        "-p",
        "--carla-port",
        help="Port of CARLA host",
        default="2000",
        metavar="PORT",
        type=int,
    )
    argparser.add_argument(
        "--episodes",
        help="Number of episodes to test for",
        default=100,
        metavar="N",
        type=int,
    )
    argparser.add_argument(
        "-d",
        "--display",
        help="Whether to display the environment",
        action="store_true",
    )
    argparser.add_argument(
        "--config-file",
        help="Path to config file",
        default="./intersection_carla_gym/src/config_continuous.yaml",
        metavar="PATH",
        type=str,
    )
    argparser.add_argument(
        "-r",
        "--random-model",
        help="Whether to test a random model",
        action="store_true",
    )

    return argparser


def main():
    # Parse arguments
    args = parse_arguments().parse_args()

    # Setup environment
    env = setup_env(
        log_dir="",
        carla_host=args.carla_host,
        carla_port=args.carla_port,
        config_file=args.config_file,
    )

    # Only setup model if random model is not selected
    if not args.random_model:
        # Load model
        if "RecurrentPPO" in args.model_path:
            model = RecurrentPPO.load(args.model_path)
        elif "PPO" in args.model_path:
            model = PPO.load(args.model_path)
        elif "DDPG" in args.model_path:
            model = DDPG.load(args.model_path)
        elif "SAC" in args.model_path:
            model = SAC.load(args.model_path)
        elif "DQN" in args.model_path:
            model = DQN.load(args.model_path)
        else:
            raise ValueError("Model not supported")

    # Setup pygame display
    if args.display:
        pygame.init()
        display = pygame.display.set_mode(
            (1024, 1024), pygame.HWSURFACE | pygame.DOUBLEBUF
        )

    # Setup metrics
    pedestrian_collision = 0
    crashed = 0
    episode_length = 0
    success = 0
    reward_sum = 0

    # Test model
    try:
        for episode in tqdm(range(args.episodes)):
            obs, info = env.reset()
            done = False

            # cell and hidden state of the LSTM
            lstm_states = None
            # Episode start signals are used to reset the lstm states
            episode_starts = np.zeros((1,), dtype=bool)

            while not done:
                if args.random_model:
                    action = env.action_space.sample()
                else:
                    action, lstm_states = model.predict(
                        obs.copy(),
                        state=lstm_states,
                        episode_start=episode_starts,
                        deterministic=True,
                    )
                obs, reward, done, _, info = env.step(action)
                episode_starts = done

                episode_length += 1
                reward_sum += reward

                if args.verbose > 0:
                    print("Action:", action)
                    print("Reward:", reward)
                if args.display:
                    env.display(display=display)
                    pygame.display.flip()

            if info["collision"]:
                crashed += 1
                if info["pedestrian_collision"]:
                    pedestrian_collision += 1
            if info["success"]:
                success += 1

    except KeyboardInterrupt:
        print("Exiting...")
        if args.display:
            pygame.display.quit()

    episode += 1
    print("Crashes:", crashed)
    print("Pedestrian Collisions:", pedestrian_collision)
    print("Episodes:", episode)
    print(f"Percentage of crashes: {round(crashed / episode, 4) * 100}%")
    if crashed != 0:
        print(
            f"Percentage of pedestrian collisions in total collisions: {round(pedestrian_collision / crashed, 2) * 100}%"
        )
    else:
        print("Percentage of pedestrian collisions in total collisions: 0%")
    print("Episode Mean Length:", round(episode_length / episode, 2))
    print(f"Success Rate: {round(success / episode, 4) * 100}%")
    print("Average Reward:", round(reward_sum / episode, 4))


if __name__ == "__main__":
    main()
