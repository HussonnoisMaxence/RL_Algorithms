
import torch
import random
import numpy as np
import torch.nn as nn
from collections import deque

from NeuralNetwork import Network

class Agent:
  def __init__(self, obs_size, n_actions, gamma=0.99, epsilon=0.9, learning_rate=0.001, batch_size=20,
               target_update = 5, min_replay_size = 20, epsilon_decay=0.999, epsilon_min =0.01, 
               memory_size = 1_000_000 ):
    
    self.n_actions = n_actions
    self.replay = deque(maxlen = memory_size)

    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    self.Q = Network(obs_size, self.n_actions, learning_rate).to(self.device)
    self.target_Q = Network(obs_size, self.n_actions, learning_rate).to(self.device)

    self.target_Q.load_state_dict(self.Q.state_dict())

    self.gamma = gamma
    self.epsilon = epsilon

    self.batch_size = batch_size
    self.memory_size = memory_size
    self.min_replay_size = min_replay_size

    self.target_update_counter = 0
    self.target_update = target_update

    self.epsilon_decay = epsilon_decay
    self.epsilon_min = epsilon_min


  def choose_action(self, obs):
        #Choose a action according to greedy epsilon
        if np.random.uniform() < self.epsilon:
          action = np.random.choice(self.n_actions)
        else:
          action = torch.argmax(self.Q(torch.tensor(obs, device=self.device, dtype=torch.float))).item()
        return action


  def train_nn(self):
    if len(self.replay) < self.min_replay_size:
      return

    idx = np.random.choice(len(self.replay), self.batch_size, replace=True) 
    mini_batch = np.array(self.replay)[idx]

    current_states = torch.tensor([transition[0] for transition in mini_batch], device=self.device, dtype=torch.float)
    actions = torch.tensor([transition[1] for transition in mini_batch], device=self.device, dtype=torch.long)
    rewards = torch.tensor([transition[2] for transition in mini_batch], device=self.device, dtype=torch.float)
    new_current_states = torch.tensor([transition[3] for transition in mini_batch], device=self.device, dtype=torch.float)
    dones = torch.tensor([not(transition[4]) for transition in mini_batch], device=self.device, dtype=torch.bool)

    actions_eval = torch.argmax(self.Q(new_current_states), dim=1)
    next_state_values = self.target_Q(new_current_states).gather(dim=1, index=actions_eval.unsqueeze(-1)).squeeze(-1)
    values = rewards + self.gamma*next_state_values*dones

    target_values = self.Q(current_states).gather(dim=1,index=actions.unsqueeze(-1)).squeeze(-1)

   
    #fit/backpropagation
    self.Q.optimizer.zero_grad()
    loss_function = nn.MSELoss()
    loss_t = loss_function(values, target_values)
    loss_t.backward()
    self.Q.optimizer.step()

    #Udate target every C step
    self.update_target()
    #Update epsilon
    self.update_epsilon()
   
  def update_epsilon(self):
    self.epsilon *= self.epsilon_decay
    self.epsilon = max(self.epsilon, self.epsilon_min)

  def update_target(self):
    self.target_update_counter +=1
    if self.target_update_counter > self.target_update:
       self.target_Q.load_state_dict(self.Q.state_dict())
       self.target_update_counter = 0

       
  def reward_shipping(self, reward):
    if reward > 1:
      return 1
    if reward < -1:
      return -1
    return reward


  