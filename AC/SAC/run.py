import gym
import numpy as np
import random, torch
import pybullet_envs
from sacA import SAC



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

env = NormalizedActions(gym.make('InvertedPendulumBulletEnv-v0'))
seed=2
env.seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

model = SAC(env=env, learning_rate_Q_functions=3e-4, learning_rate_policy=3e-4, gamma=0.99, 
        memory_capacity=500_000, temperature=0.1, train_freq=32,gradient_steps=32, polyak_factor=0.005, min_replay_size=256, batch_size=256)

model.learn(total_timesteps=50000)

"""
model = SAC("MlpPolicy", env, learning_rate=3e-4, batch_size=512, ent_coef=0.1, train_freq=32, gradient_steps=32, 
	gamma=0.9999, tau=0.01, learning_starts=0, use_sde=True, policy_kwargs=dict(log_std_init=-3.67,net_arch=[64,64]),verbose=1)
model.learn(total_timesteps=50000, log_interval=4)

model = SAC.load("sac_pendulum")

obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()
"""