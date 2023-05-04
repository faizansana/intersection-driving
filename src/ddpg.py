import gymnasium as gym
from stable_baselines3 import DDPG

import sys
sys.path.append("./gym_carla")

import gym_carla


if __name__ == "__main__":

    # parameters for the network
    NEURONS = (400, 300)
    LEARNING_RATE = 1e-3
    BUFFER_SIZE = 100
    LEARNING_STARTS = 10
    GAMMA = 0.98
    TRAIN_FREQ = (1, "episode")
    GRADIENT_STEPS = -1
    VERBOSE = 1


    params = {
        "number_of_vehicles": 40,
        "number_of_walkers": 10,
        "display_size": 256,  # screen size of bird-eye render
        "max_past_step": 1,  # the number of past steps to draw
        "dt": 0.1,  # time interval between two frames
        "discrete": False,  # whether to use discrete control space
        "discrete_acc": [-3.0, 0.0, 3.0],  # discrete value of accelerations
        "discrete_steer": [-0.2, 0.0, 0.2],  # discrete value of steering angles
        "continuous_accel_range": [-3.0, 3.0],  # continuous acceleration range
        "continuous_steer_range": [-0.3, 0.3],  # continuous steering angle range
        "ego_vehicle_filter": "vehicle.lincoln*",  # filter for defining ego vehicle
        "host": "172.18.0.4",  # which host to use
        "port": 2000,  # connection port
        "town": "Town03",  # which town to simulate
        "task_mode": "random",  # mode of the task, [random, roundabout (only for Town03)]
        "max_time_episode": 1000,  # maximum timesteps per episode
        "max_waypt": 12,  # maximum number of waypoints
        "obs_range": 32,  # observation range (meter)
        "lidar_bin": 0.125,  # bin size of lidar sensor (meter)
        "d_behind": 12,  # distance behind the ego vehicle (meter)
        "out_lane_thres": 2.0,  # threshold for out of lane
        "desired_speed": 8,  # desired speed (m/s)
        "max_ego_spawn_times": 200,  # maximum times to spawn ego vehicle
        "display_route": True,  # whether to render the desired route
        "pixor_size": 64,  # size of the pixor labels
        "pixor": False,  # whether to output PIXOR observation
    }

    env = gym.make('carla-v0', params=params, apply_api_compatibility=True)
    obs = env.reset()

    model = DDPG("MultiInputPolicy",
                 env,
                 policy_kwargs={"net_arch": [400, 300]},
                 learning_rate=LEARNING_RATE,
                 buffer_size=BUFFER_SIZE,
                 learning_starts=LEARNING_STARTS,
                 gamma=GAMMA,
                 train_freq=TRAIN_FREQ,
                 gradient_steps=GRADIENT_STEPS,
                 verbose=VERBOSE,
                 tensorboard_log="./logs/DDPG/")
    model.learn(total_timesteps=2e3, log_interval=10, progress_bar=True)
    model.save("DDPG_carla")