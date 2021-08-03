import torch
import random, math
import numpy as np
import torch.nn as nn

from Buffer import ReplayMemory, Transition
from NeuralNetwork import Network

class AgentDQN:
  def __init__(self, config, model):
    self.config = config

    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.model = Network(config).to(self.device)
    self.model.load_state_dict(model.state_dict())
    #Create Q network and its target
    self.target_Q = Network(config).to(self.device)

    self.target_Q.load_state_dict(model.state_dict())
    
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
        action= self.model(obs)
        self.Qvalues = action.max(0)[0]
        action = torch.argmax(action).detach().item()
    
    return action



  def train_nn(self, obs, action, reward, next_obs, done, model):
    self.Q_optimizer = torch.optim.Adam(params=model.parameters(), lr=self.config["learning_rate"])
    self.model.load_state_dict(model.state_dict())
    state_batch = obs
    action_batch = action
    reward_batch = reward
    next_state_batch = next_obs
    done_batch = done

    #------------ optimize Q_network--------------

    #compute True value
    next_state_values = self.target_Q(next_state_batch)
    next_state_values = next_state_values.max(0)[0].detach()
    target_values = reward_batch + self.config["gamma"]*next_state_values*done_batch

    #Compute target value
    
    values_p = self.model(state_batch)
    values = values_p.gather(dim=0,index=action_batch.unsqueeze(-1)).squeeze(-1)
    loss_t = self.Q_loss(values, target_values) #MSE loss for Q network

    loss_t.backward()

    if not((self.steps_done + 1)%10):
        ensure_shared_grads(self.model, model)
        self.Q_optimizer.step()
        self.Q_optimizer.zero_grad()
        

    #------------ update target--------------
    #Udate target every C step
    #self.update_target(self.model)
    self.polyak_update(model, self.target_Q, self.config["polyak_factor_Q"] )
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
    #self.target_Q.load_state_dict(target_param)


  def update_target(self, model):
    self.target_update_counter +=1
    if self.target_update_counter > self.config["target_update"]:
      self.target_Q.load_state_dict(model.state_dict())
      self.target_update_counter = 0



def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad
  