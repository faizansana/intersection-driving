import logging
import subprocess
from multiprocessing import Process
import signal

logging.basicConfig(filename='multitrain.log', level=logging.INFO, format="%(asctime)s %(message)s")

models = ["DDPG", "PPO", "RecurrentPPO", "SAC", "DQN"]
total_runs = len(models)
base_server_name = "intersection-driving-carla_server"
base_tm_port = 8000


def get_saved_model_location(log_file: str) -> str:
    with open(log_file, 'r') as f:
        for line in f:
            if "Saving new best model to " in line:
                return line.split("Saving new best model to ")[1].strip() + ".zip"
    return ""


def run_retraining(model_path: str, tm_port: int, server_name: str, config_file: str, log_file: str):

    command = ["python", "train.py", "--model-path", model_path, "-t", "1500000", "-c", server_name, "--tm-port", str(tm_port), "-v", "1", "--config-file", config_file]

    try:
        with open(log_file, 'a') as output:
            subprocess.run(command, check=True, stdout=output, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        logging.info(f"Run {model} failed with error: {e}.")
        if abs(e.returncode) == signal.Signals.SIGSEGV or abs(e.returncode) == signal.Signals.SIGABRT:
            logging.info(f"Attempting to resume training for {model}...")
            run_retraining(model_path, tm_port, server_name, config_file, log_file)
        logging.info(f"Failed to resume training for {model_path} with error: {e}")


def run_training(model: str, tm_port: int, server_name: str, config_file: str = "./custom_carla_gym/config.yaml"):
    log_file = f"train_{model}.log"
    command = ["python", "train.py", "-m", model, "-t", "1500000", "-c", server_name, "--tm-port", str(tm_port), "-v", "1", "--config-file", config_file]

    try:
        with open(log_file, 'w') as output:
            subprocess.run(command, check=True, stdout=output, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        logging.info(f"Run {model} failed with error: {e}.")
        if abs(e.returncode) == signal.Signals.SIGSEGV or abs(e.returncode) == signal.Signals.SIGABRT:
            logging.info(f"Attempting to resume training for {model}...")
            model_path = get_saved_model_location(log_file)
            if model_path == "":
                run_training(model, tm_port, server_name, config_file)
            else:
                run_retraining(model_path, tm_port, server_name, config_file, log_file)
        logging.info(f"Failed to resume training for {model} in RUN_TRAINING with error: {e}")


processes = []

for i in range(total_runs):
    model = models[i]
    tm_port = base_tm_port + i
    if model == "DQN" or model == "PPO" or model == "RecurrentPPO":
        config_file = "./custom_carla_gym/config_discrete.yaml"
    else:
        config_file = "./custom_carla_gym/config_continuous.yaml"

    process = Process(target=run_training, args=(model, tm_port, f"{base_server_name}-{i+1}", config_file))
    processes.append(process)
    process.start()

for process in processes:
    process.join()
