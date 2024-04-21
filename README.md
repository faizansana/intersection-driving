# Training Architecture for CARLA-based Reinforcement Learning Environments

[![Push to Docker Hub](https://github.com/faizansana/intersection-driving/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/faizansana/intersection-driving/actions/workflows/docker-publish.yml)

Containerized DRL training architecture for [gymnasium](https://gymnasium.farama.org/index.html) based [CARLA Simulator](https://carla.org/) environments. Particularly designed for the [intersection carla gym](https://github.com/faizansana/intersection-carla-gym) repository.

# Getting Started

## DRL Algorithms Supported

- Deep Deterministic Policy Gradient (DDPG)
- Deep Q-Learning (DQN)
- Proximal Policy Optimization (PPO)
- Recurrent PPO
- Soft Actor Critic (SAC)

## System Requirements

The following are the requirements for running this repository using the provided Docker files:

- Operating System: Linux (tested on Ubuntu 20.04/22.04)
- NVIDIA GPU with CUDA support (tested on NVIDIA GeForce RTX 3060/3080/3090/4080/4090)

## Setup

1. Clone the repository

    ```
    git clone https://github.com/faizansana/intersection-driving.git
    ```

2. Run the `dev_config.sh` file to set the environment variables for docker.

    ```
    bash dev_config.sh
    ```

3. From within the working directory, open the `.env` file to change any specific requirements such as CARLA version, CUDA version etc. The following are the default configurations:
    
    | Variable                  | Description              | Default Value                   |
    |---------------------------|--------------------------|-------------------------|
    | FIXUID                    | UID of current user      | (UID of your current user) |
    | FIXGID                    | GID of current user      | (GID of current user) |
    | CARLA_VERSION             | Version of CARLA         | 0.9.10.1                |
    | CARLA_QUALITY             | Quality setting for CARLA| Low                     |
    | GPU_ID_CARLA_MAIN         | GPU ID for CARLA main    | 0                       |
    | GPU_ID_CARLA_DEBUG        | GPU ID for CARLA debug   | 0                       |
    | GPU_ID_MAIN_CONTAINER     | GPU ID for main container| 0                       |
    | CARLA_SERVER_REPLICAS     | Number of CARLA server replicas | 5              |
    | CARLA_DEBUG_SERVER_REPLICAS | Number of CARLA debug server replicas | 0         |
    | CUDA_VERSION              | Version of CUDA          | 12.0.0                  |

    *Note:* The GPU IDs are automatically set by checking the least used GPUs on the system. 


4. Pull the already built containers from Docker Hub if available.

    ```
    docker compose pull
    ```

5. After the containers have been pulled, start them using the following command.

    ```
    docker compose up -d
    ```

7. Open the `main_container`, and attach it to VS Code using the `Remote Explorer` extension.

# Scripts Usage (from within main container)

The following are the scripts developed for use (found within `src` folder):

1. `multi_retrain.py`: Retrain multiple DRL models using a `yaml` file with their locations. 

    Example usage:
    ```python
    python multi_retrain.py -f file_with_model_paths.yaml -t number_of_timesteps_to_train
    ```

2. `multi_testmodel.py`: Test multiple models based on the performance metrics defined in *test_model.py*.

    Example usage:
    ```python
    python multi_testmodel.py
    ```

    *Note:* Modify the *model_paths* list in the script to select the model paths

3. `multi_train.py`: Train multiple DRL algorithms in parallel in different CARLA instances

    Example usage:
    ```python
    python multi_train.py -t number_of_timesteps_to_train
    ```

4. `test_model.py`: Test a single DRL model.

    Example usage:
    ```python
    python test_model.py -m path_to_model -v verbosity_level -c carla_host --episodes numberof_episodes -d display_or_not --config-file path_to_environment_config 
    ```

5. `train.py`: Train a single DRL model or retrain a model.

    Example usage:
    ```python
    python train.py -m name_of_model -v verbosity_level -c carla_host --episodes numberof_episodes -d display_or_not --config-file path_to_environment_config -p carla_port
    ```
