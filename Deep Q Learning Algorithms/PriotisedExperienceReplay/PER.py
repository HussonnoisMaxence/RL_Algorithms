import torch
import random, math
import numpy as np
import torch.nn as nn

from PrioritisedBuffer import PrioritisedExperienceReplay, Transition
from NeuralNetwork import DuelingNetwork

class Agent:
  def __init__(self, config):
    self.config = config

    self.memory = PrioritisedExperienceReplay(config["buffer"])

    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #Create Q network and its target
    self.Q = DuelingNetwork(config).to(self.device)
    self.target_Q = DuelingNetwork(config).to(self.device)
    self.Q_optimizer = torch.optim.Adam(params=self.Q.parameters(), lr=config["learning_rate"])

    #self.target_Q.load_state_dict(self.Q.state_dict())
    
    self.epsilon = self.config["EPS_START"]

    self.Q_loss = torch.nn.SmoothL1Loss() #torch.nn.MSELoss()
    self.lossQ = 0
    
    self.target_update_counter = 0
    self.steps_done = 0

    self.Qvalues = 0

  def print_params(self, network):
    model_parameters = filter(lambda p: p.requires_grad, network.parameters())
    paramsQ = sum([np.prod(p.size()) for p in model_parameters])
    print(paramsQ)

  def choose_action(self, obs):
    #Choose a action according to greedy epsilon
    obs = torch.tensor(obs, dtype=torch.float, device=self.device).unsqueeze(0)
    if random.random() < self.epsilon:
        action = np.random.choice(self.config["n_actions"])

    else:
      with torch.no_grad():
        action = self.Q(obs)
        self.Qvalues = action.max(0)[0]
        action = torch.argmax(action).detach().item()

    return action


  def train_nn(self):
    if len(self.memory) < self.config["min_replay_size"]:
      return

    #---------TRAINING------------------------
    transitions, weights = self.memory.sample(self.config["batch_size"]) #, weight 
    batch = Transition(*zip(*transitions))
    
    weights = torch.tensor(weights, device=self.device, dtype=torch.float)
    #-- Get each element of the transition in batches
    state_batch = torch.stack(batch.state)
    next_state_batch = torch.stack(batch.next_state)
    action_batch = torch.tensor(batch.action, device=self.device)
    reward_batch = torch.tensor(batch.reward, device=self.device)
    done_batch = torch.tensor(batch.done, device=self.device)

    
    #------------ optimize Q_network--------------

    #compute Target value with double Q-learning
    actions_eval = torch.argmax(self.Q(next_state_batch), dim=1)
    next_state_values = self.target_Q(next_state_batch).gather(dim=1, index=actions_eval.unsqueeze(-1)).squeeze(-1).detach()

    target_values = reward_batch + self.config["gamma"]*next_state_values*done_batch
    

    #Compute  value
    values_p = self.Q(state_batch)
    values = values_p.gather(dim=1,index=action_batch.unsqueeze(-1)).squeeze(-1)



    #compute td-error
    td_error = target_values - values  
   
    self.memory.update_error(td_error)

    ## Compute loss
    #loss_function(values*weight, target_values*weight)
    loss_t = self.Q_loss(values*weights, target_values*weights) #MSE loss for Q network

    self.Q_optimizer.zero_grad()

    loss_t.backward()
    torch.nn.utils.clip_grad_norm_(self.Q.parameters(), 10)

    self.Q_optimizer.step()
    

    #------------ update target--------------
    #Udate target every C step
    #self.update_target()
    self.polyak_update(self.Q, self.target_Q, self.config["polyak_factor_Q"] )
    #------------other--------------
    self.lossQ = loss_t
    self.update_epsilon() #update epsilon

  def update_epsilon(self):
    self.epsilon = self.config["EPS_END"] + (self.config["EPS_START"] - self.config["EPS_END"]) * \
      math.exp(-1. * self.steps_done / self.config["EPS_DECAY"])

    self.steps_done +=1 

  def polyak_update(self, network, network_target, factor):
    polyak_factor = factor
    for target_param, param in zip(network_target.parameters(), network.parameters()):
      target_param.data.copy_(polyak_factor*param.data + target_param.data*(1.0 - polyak_factor))


  def update_target(self):
    self.target_update_counter +=1
    if self.target_update_counter > self.config["target_update"]:
      self.target_Q.load_state_dict(self.Q.state_dict())
      self.target_update_counter = 0



  
