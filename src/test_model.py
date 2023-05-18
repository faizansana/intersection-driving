import argparse

import pygame
from sb3_contrib import RecurrentPPO
from stable_baselines3 import DDPG, PPO, SAC
from tqdm import tqdm

from train import setup_env


def parse_arguments():
    argparser = argparse.ArgumentParser(description="Test Agent")
    argparser.add_argument(
        "-m", "--model-path",
        help="Path to model to test",
        default="",
        metavar="PATH",
        type=str)
    argparser.add_argument(
        "-e", "--env",
        help="Environment to test on. Path of env is appended",
        default="custom_carla_gym",
        metavar="NAME",
        choices=["custom_carla_gym", "gym_carla"],
        type=str)
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
        "--episodes",
        help="Number of episodes to test for",
        default=100,
        metavar="N",
        type=int)
    argparser.add_argument(
        "-d", "--display",
        help="Whether to display the environment",
        action="store_true"
    )

    return argparser


def main():
    # Parse arguments
    args = parse_arguments().parse_args()

    # Setup environment
    env = setup_env(env_name=args.env, log_dir="", carla_host=args.carla_host)

    # Load model
    if args.model_path == "":
        raise ValueError("No model path provided")

    # Check if model path string contains PPO

    if "PPO" in args.model_path:
        model = PPO.load(args.model_path)
    elif "DDPG" in args.model_path:
        model = DDPG.load(args.model_path)
    elif "SAC" in args.model_path:
        model = SAC.load(args.model_path)
    elif "RecurrentPPO" in args.model_path:
        model = RecurrentPPO.load(args.model_path)
    else:
        raise ValueError("Model not supported")

    # Setup pygame display
    if args.display:
        pygame.init()
        display = pygame.display.set_mode(
            (1024, 1024),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

    # Setup metrics
    crashed = 0
    episode_length = 0
    success = 0
    reward_sum = 0

    # Test model
    try:

        for episode in tqdm(range(args.episodes)):
            obs = env.reset()
            done = False

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)

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
            if info["success"]:
                success += 1

    except KeyboardInterrupt:
        print("Exiting...")
        if args.display:
            pygame.display.quit()

    print("Crashes:", crashed)
    print("Episodes:", episode)
    print(f"Percentage of crashes: {round(crashed / episode, 2)}%")
    print("Episode Mean Length:", episode_length / episode)
    print(f"Success Rate: {round(success / episode, 2)}%")
    print("Average Reward:", reward_sum/episode)


if __name__ == "__main__":
    main()
