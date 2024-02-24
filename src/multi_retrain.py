import argparse
import logging
from multiprocessing import Process

import yaml

import multi_train

logging.basicConfig(filename='multi_retrain.log', level=logging.INFO, format="%(asctime)s %(message)s")


def get_model_name(model_path: str) -> str:
    return model_path.split("/")[-3]


def parse_arguments():
    argparser = argparse.ArgumentParser(description="Retrain Multiple Agents")
    argparser.add_argument(
        "-f", "--file",
        help="Path to YAML file containing paths to models to retrain",
        metavar="FILEPATH",
        default="Training/retrain_models.yaml",
        type=str)
    argparser.add_argument(
        "-c", "--base-carla-host",
        help="DNS name of base CARLA host",
        default="intersection-driving-carla_server",
        metavar="SERVER_NAME",
        type=str)
    argparser.add_argument(
        "-t", "--timesteps",
        help="Number of timesteps to train for",
        default=1500000,
        metavar="N",
        type=int)

    return argparser


def main():
    args = parse_arguments().parse_args()
    with open(args.file, 'r') as f:
        model_paths = yaml.safe_load(f)
    total_runs = len(model_paths)

    processes = []

    for i in range(total_runs):
        model_path = model_paths[i]
        model_name = get_model_name(model_path)
        log_file = f"train_{model_name}.log"

        if "DQN" in model_path or "PPO" in model_path:
            config_file = "./custom_carla_gym/src/config_discrete.yaml"
        else:
            config_file = "./custom_carla_gym/src/config_continuous.yaml"

        process = Process(target=multi_train.run_retraining, args=(model_path, f"{args.base_carla_host}-{i+1}", config_file, log_file, args.timesteps))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()


if __name__ == "__main__":
    main()
    logging.info("All Retraining finished")
