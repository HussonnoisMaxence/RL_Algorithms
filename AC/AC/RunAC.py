import gym
import random
import numpy as np
import torch
from ActorCritic import Agent
import matplotlib.pyplot as plt
if __name__ == "__main__":
  
  seed = 2



  



  env = gym.make("CartPole-v0")
  env.seed(seed)
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  agent = Agent()
  
  steps = []
  step = 0
  score = []

  for episode in range(5000):
      #initialized sequence
      obs = env.reset()
      done = False
      total_reward = 0
      
      trajectory = []
      while not(done):
        action = agent.choose_action(obs)
        next_obs, reward, done, _ =  env.step(action)  
        trajectory.append((obs, action, reward, next_obs, done))  
        obs = next_obs
        total_reward += reward

      agent.train(trajectory)
      print("step:", episode,"reward:", total_reward, "loss", agent.loss) 
      score.append(total_reward)

ep = [i for i in range(episode+1)]

plt.plot(ep,score)
plt.show()