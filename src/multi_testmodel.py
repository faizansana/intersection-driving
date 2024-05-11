import logging
import os
import subprocess
from multiprocessing import Process

logging.basicConfig(
    filename="multitest.log", level=logging.INFO, format="%(asctime)s %(message)s"
)

model_paths = [
    "/home/docker/src/src/Training/Models/PPO/2024-04-09_21-53-35/best_model.zip"
]

assert len(model_paths) <= 5, "Not enough servers for all models"
total_runs = len(model_paths)
base_server_name = "intersection-driving-carla_server"
base_tm_port = 10000


def run_test(
    model_path: str,
    server_name: str,
    config_file: str = "./intersection_carla_gym/config.yaml",
):
    directory = os.path.dirname(model_path)
    log_file = os.path.join(directory, "test_model.log")

    command = [
        "python",
        "test_model.py",
        "-m",
        model_path,
        "-c",
        server_name,
        "-v",
        "1",
        "--config-file",
        config_file,
        "--episodes",
        str(1000),
    ]

    try:
        with open(log_file, "w") as output:
            subprocess.run(command, check=True, stdout=output, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        logging.error(f"Run {model_path} failed with error: {e}")


processes = []

try:
    for i in range(total_runs):
        model_path = model_paths[i]
        if "DQN" in model_path or "PPO" in model_path:
            config_file = "./intersection_carla_gym/src/config_discrete.yaml"
        else:
            config_file = "./intersection_carla_gym/src/config_continuous.yaml"

        process = Process(
            target=run_test, args=(model_path, f"{base_server_name}-{i+1}", config_file)
        )
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

except Exception as ex:
    logging.error(f"An error occured: {ex}")

finally:
    logging.info("All tests finished")
    for process in processes:
        process.terminate()
        process.join()
