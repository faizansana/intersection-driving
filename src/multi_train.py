import logging
import subprocess
from multiprocessing import Process

logging.basicConfig(filename='multitrain.log', level=logging.INFO)

models = ["DDPG", "PPO", "RecurrentPPO", "SAC"]
total_runs = len(models)
base_server_name = "intersection-driving-carla_server_low"
base_tm_port = 8000


def run_training(model: str, tm_port: int, run_index: int, server_name: str):
    log_file = f"train_{model}_{run_index + 1}.log"
    command = ["python", "train.py", "-m", model, "-t", "500000", "-c", server_name, "--tm-port", str(tm_port), "-v", "1"]

    try:
        with open(log_file, 'w') as output:
            subprocess.run(command, check=True, stdout=output, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        logging.error(f"Run {run_index + 1} failed with error: {e}")


processes = []

for i in range(total_runs):
    model = models[i]
    tm_port = base_tm_port + i
    process = Process(target=run_training, args=(model, tm_port, i, f"{base_server_name}-{i+1}"))
    processes.append(process)
    process.start()

for process in processes:
    process.join()
