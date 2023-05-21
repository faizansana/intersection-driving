import logging
import subprocess
from multiprocessing import Process

logging.basicConfig(filename='multitrain.log', level=logging.INFO, format="%(asctime)s %(message)s")

models = ["DDPG", "PPO", "RecurrentPPO", "SAC", "DQN"]
total_runs = len(models)
base_server_name = "intersection-driving-carla_server"
base_tm_port = 8000


def run_training(model: str, tm_port: int, server_name: str, config_file: str = "./custom_carla_gym/config.yaml"):
    log_file = f"train_{model}.log"
    command = ["python", "train.py", "-m", model, "-t", "1500000", "-c", server_name, "--tm-port", str(tm_port), "-v", "1", "--config-file", config_file]

    try:
        with open(log_file, 'w') as output:
            subprocess.run(command, check=True, stdout=output, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        logging.error(f"Run {model} failed with error: {e}")


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
