import gym
import numpy as np
import random, torch
import pybullet_envs
from SAC import SAC



import torch


class NormalizedActions(gym.ActionWrapper):
    def action(self, action):
        low  = self.action_space.low
        high = self.action_space.high
        
        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)
        
        return action

    def _reverse_action(self, action):
        low  = self.action_space.low
        high = self.action_space.high
        
        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)
        
        return actions

env = NormalizedActions(gym.make('HalfCheetah-v2'))
seed=2
env.seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

model = SAC(env=env, learning_rate_Q_functions=3e-4, learning_rate_policy=3e-4, gamma=0.99, 
        memory_capacity=500_000,  train_freq=32,gradient_steps=32, polyak_factor=0.005, batch_size=256)

model.learn(total_timesteps=50000)

