import gym
import random
import numpy as np
import torch
from PolicyGradient import Agent

if __name__ == "__main__":
  
  seed = 2



  



  env = gym.make("CartPole-v0")
  env.seed(seed)
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  agent = Agent()
  

  score = []
  batch = []
  for episode in range(1000):
      #initialized sequence
      obs = env.reset()
      done = False
      total_reward = 0
      
      trajectory = []
      while not(done):
        action = agent.choose_action(obs)
        next_obs, reward, done, _ =  env.step(action)  
        trajectory.append((obs, action, reward, next_obs))  
        obs = next_obs
        total_reward += reward

      batch.append(trajectory)
      if len(batch) > 1: 
        agent.train(batch)
        batch = []


      score.append(total_reward)
      print("step:", episode,"reward:", total_reward, "loss", agent.loss) 
