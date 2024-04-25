import argparse
import logging
import subprocess
from multiprocessing import Process
import signal

logging.basicConfig(filename='multitrain.log', level=logging.INFO, format="%(asctime)s %(message)s")


def parse_arguments():
    argparser = argparse.ArgumentParser(description="Train Multiple Agents")
    # Create list of models to train
    argparser.add_argument(
        "-m", "--models",
        help="Models to train",
        metavar="NAME",
        nargs='+',
        default=["DDPG", "DQN", "SAC", "PPO", "RecurrentPPO"])
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


def get_saved_model_location(log_file: str) -> str:
    with open(log_file, 'r') as f:
        for line in f:
            if "Saving latest model to " in line:
                return line.split("Saving latest model to ")[1].strip() + ".zip"
    return ""


def run_retraining(model_path: str, server_name: str, config_file: str, log_file: str, timesteps: int = 1500000):

    command = ["python", "train.py", "--model-path", model_path, "-t", str(timesteps), "-c", server_name, "-v", "1", "--config-file", config_file]

    try:
        with open(log_file, 'a') as output:
            subprocess.run(command, check=True, stdout=output, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        logging.info(f"Run {model_path} failed with error: {e}.")
        if abs(e.returncode) == signal.Signals.SIGSEGV or abs(e.returncode) == signal.Signals.SIGABRT:
            logging.info(f"Attempting to resume training for {model_path}...")
            run_retraining(model_path, server_name, config_file, log_file, timesteps)
        logging.info(f"Failed to resume training for {model_path} with error: {e}")


def run_training(model: str, server_name: str, timesteps: int, config_file: str = "./intersection_carla_gym/config.yaml"):
    log_file = f"train_{model}.log"
    command = ["python", "train.py", "-m", model, "-t", str(timesteps), "-c", server_name, "-v", "1", "--config-file", config_file]

    try:
        with open(log_file, 'w') as output:
            subprocess.run(command, check=True, stdout=output, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        logging.info(f"Run {model} failed with error: {e}.")
        if abs(e.returncode) == signal.Signals.SIGSEGV or abs(e.returncode) == signal.Signals.SIGABRT:
            logging.info(f"Attempting to resume training for {model}...")
            model_path = get_saved_model_location(log_file)
            if model_path == "":
                run_training(model, server_name, timesteps, config_file)
            else:
                run_retraining(model_path, server_name, config_file, log_file, timesteps)
        logging.info(f"Failed to resume training for {model} in RUN_TRAINING with error: {e}")


def main():
    args = parse_arguments().parse_args()
    models = args.models
    total_runs = len(models)

    processes = []

    for i in range(total_runs):
        model = models[i]
        if model == "DQN" or model == "PPO" or model == "RecurrentPPO":
            config_file = "./intersection_carla_gym/src/config_discrete.yaml"
        else:
            config_file = "./intersection_carla_gym/src/config_continuous.yaml"

        process = Process(target=run_training, args=(model, f"{args.base_carla_host}-{i+1}", args.timesteps, config_file))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()


if __name__ == "__main__":
    main()
    logging.info("All training finished")
