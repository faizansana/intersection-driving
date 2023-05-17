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
    if args.verbose > 0:
        pygame.init()
        display = pygame.display.set_mode(
            (1024, 1024),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

    # Test model
    try:

        for episode in tqdm(range(args.episodes)):
            obs, info = env.reset()
            done = False

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, done, info = env.step(action)

                if args.verbose > 0:
                    print("Action:", action)
                    print("Reward:", reward)
                    env.display(display=display)
                    pygame.display.flip()

    except KeyboardInterrupt:
        print("Exiting...")
        if args.verbose > 0:
            pygame.display.quit()


if __name__ == "__main__":
    main()
