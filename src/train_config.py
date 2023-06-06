import os

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == "ego_state":
                # Conv and relu
                extractors[key] = torch.nn.Linear(subspace.shape[0], subspace.shape[0])
                total_concat_size += subspace.shape[0]
            elif key == "adv_vehicles":
                # Conv and relu
                extractors[key] = torch.nn.Linear(subspace.shape[0], subspace.shape[0])
                total_concat_size += subspace.shape[0]
            elif key == "pedestrians":
                # Conv and relu
                extractors[key] = torch.nn.Linear(subspace.shape[0], subspace.shape[0])
                total_concat_size += subspace.shape[0]

        self.extractors = torch.nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return torch.cat(encoded_tensor_list, dim=1)


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level.
    """

    def __init__(self, check_freq: int, log_dir: str, save_path: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = save_path
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_training_start(self) -> None:
        x, y = ts2xy(load_results(self.log_dir), 'timesteps')
        if len(x) > 0:
            # Mean training reward over the last 100 episodes
            self.best_mean_reward = np.mean(y[-100:])

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}")

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    save_path = os.path.join(self.save_path, "best_model")
                    # Example for saving best model
                    if self.verbose > 0:
                        print(f"Saving new best model to {save_path}")
                    self.model.save(save_path)
        return True


class SaveLatestModelCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level.
    """

    def __init__(self, check_freq: int, save_path: str, verbose: int = 1):
        super(SaveLatestModelCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.previous_episode_num = 0

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_training_start(self) -> None:
        self.previous_episode_num = self.model._episode_num

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # if self.model._episode_num > self.previous_episode_num:
            self.previous_episode_num = self.model._episode_num
            save_path = os.path.join(self.save_path, "latest_model")
            # Example for saving best model
            if self.verbose > 0:
                print(f"Saving latest model to {save_path}")
            self.model.save(save_path)
        return True
