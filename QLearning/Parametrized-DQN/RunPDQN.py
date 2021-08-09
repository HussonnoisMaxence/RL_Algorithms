import gym
import random
import numpy as np
import torch
from utils import *
from Environment import *
from torch.utils.tensorboard import SummaryWriter
from PDQN import *
import argparse
import matplotlib.pyplot as plt

if __name__ == "__main__":
  
  parser = argparse.ArgumentParser()
  parser.add_argument('-env', type=str, default="CP", help='CP/LL')
  args = parser.parse_args()

  if args.env == "CP":
    config = read_config("./configCP.yaml")
    env = Environment2()
  elif args.env == "LL":
    config = read_config("./configLL.yaml")
    env = Environment()

  print(config)

  #writer =  SummaryWriter()


  env.seed(config["seed"])
  random.seed(config["seed"])
  np.random.seed(config["seed"])
  torch.manual_seed(config["seed"])
  agent = Agent(config)
  steps = []
  step = 0
  q_values = []
  q_loss = []
  p_loss = []
  for episode in range(config["episode_batch"]):

      #initialized sequence
      obs = env.reset()
      done = False
      total_reward = 0  
      while not(done):
        action, params = agent.choose_action(obs)
        #perform action
        next_obs, reward, done, _ =  env.step([action, params[action]])    
        #print("reward:", reward)     
        #store transition in replay
        agent.memory.push(torch.tensor(obs, device=agent.device, dtype=torch.float), 
                          torch.tensor(action, device=agent.device, dtype=torch.long), 
                          torch.tensor(params, device=agent.device, dtype=torch.float),
                          torch.tensor(reward, device=agent.device, dtype=torch.float), 
                          torch.tensor(next_obs, device=agent.device, dtype=torch.float), 
                          torch.tensor(not(done), device=agent.device, dtype=torch.bool))
        #train neural network
        agent.train_nn()


        obs = next_obs
        total_reward += reward

        steps.append(step)
        q_values.append(agent.Qvalues)
        q_loss.append(agent.lossQ)
        p_loss.append(agent.lossP)
        step += 1 
      """
      writer.add_scalar("Score", total_reward, episode)
      writer.add_scalar("LossQ", agent.lossQ, episode)
      writer.add_scalar("LossP", agent.lossP, episode)
      """
      print("step:", episode,"reward:", total_reward,"eps:", agent.epsilon, "memsize:", len(agent.memory)) 

  plt.plot(steps, q_values, color='green')
  plt.plot(steps, q_loss, color='blue')
  plt.plot(steps, p_loss, color='red')
  plt.show()
  #writer.close()