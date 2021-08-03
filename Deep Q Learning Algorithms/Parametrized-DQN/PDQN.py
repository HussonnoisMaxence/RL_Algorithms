import torch
import random, math
import numpy as np
import torch.nn as nn
from collections import deque
from NeuralNetwork import *
from Buffer import *


class Agent:
  def __init__(self, config):
    self.config = config

    
    self.memory = ReplayMemory(config["size_buffer"])

    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.device = torch.device("cpu")

    self.Q = QNetwork(config).to(self.device)
    model_parameters = filter(lambda p: p.requires_grad, self.Q.parameters())
    paramsQ = sum([np.prod(p.size()) for p in model_parameters])
    print(paramsQ)

    self.parameters_net = PNetwork(config).to(self.device)
    self.parameters_net_target = PNetwork(config).to(self.device)
    self.target_Q = QNetwork(config).to(self.device)

    #self.target_Q.load_state_dict(self.Q.state_dict())

    
    self.epsilon = self.config["EPS_START"]

    self.Q_loss = torch.nn.SmoothL1Loss() #torch.nn.MSELoss()
    self.lossQ = 0
    self.lossP = 0
    
    self.target_update_counter = 0
    self.steps_done = 0

    self.Qvalues = 0

  def choose_action(self, obs):
        #Choose a action according to greedy epsilon
        
        if random.random() < self.epsilon:
            action = np.random.choice(self.config["n_actions"])
        
            params = np.array([random.uniform(self.config["params_range"][0],self.config["params_range"][1]) 
                for n in range(self.config["n_actions"])])
        else:
          with torch.no_grad():
            params = self.parameters_net(torch.tensor(obs, dtype=torch.float, device=self.device))
            action= self.Q(torch.tensor(obs, dtype=torch.float, device=self.device),
                            params)
            self.Qvalues = action.max(0)[0]
            action = torch.argmax(action).detach().item()
            #print(self.Qvalues)
            params = params.detach().cpu().numpy()

        self.update_epsilon() #update epsilon
        return action, params


  def train_nn(self):
    if len(self.memory) < self.config["min_replay_size"]:
      return
    transitions = self.memory.sample(self.config["batch_size"])
    batch = Transition(*zip(*transitions))

    # Get each element of the transition in batches
    state_batch = torch.stack(batch.state)
    action_batch = torch.tensor(batch.action, device=self.device)
    params_batch = torch.stack(batch.params)
    reward_batch = torch.tensor(batch.reward, device=self.device)
    next_state_batch = torch.stack(batch.next_state)
    done_batch = torch.tensor(batch.done, device=self.device)

    
    #------------ optimize Q_network--------------
    #Double Q learning
    next_params = self.parameters_net_target(next_state_batch)
    actions_eval = torch.argmax(self.Q(next_state_batch, next_params), dim=1)
    next_state_values = self.target_Q(next_state_batch, next_params).gather(dim=1, index=actions_eval.unsqueeze(-1)).squeeze(-1).detach()

    #------- without D2QN------------
    #compute True value
    #compute params
    #next_state_values = self.target_Q(next_state_batch, next_params)
    #next_state_values = next_state_values.max(1)[0].detach()
    #------- without D2QN------------
    

    values = reward_batch + self.config["gamma"]*next_state_values*done_batch

    #Compute target value
    target_values_p = self.Q(state_batch, params_batch)
    target_values = target_values_p.gather(dim=1,index=action_batch.unsqueeze(-1)).squeeze(-1)
    loss_t = self.Q_loss(target_values, values) #MSE loss for Q network

    self.Q.optimizer.zero_grad()
    loss_t.backward()
    torch.nn.utils.clip_grad_norm_(self.Q.parameters(), 10)
    self.Q.optimizer.step()



    #------------ optimize Parametrized_network--------------


    target_values_params = self.Q(state_batch, self.parameters_net(state_batch))
    t = -torch.sum(target_values_params, 0) #Sum for Parameter network

    loss_p = torch.mean(t)
    #fit/backpropagation
    self.parameters_net.optimizer.zero_grad()
    
    loss_p.backward() 
    
    torch.nn.utils.clip_grad_norm_(self.parameters_net.parameters(), 10)
    self.parameters_net.optimizer.step()





    #------------ update target--------------
    #Udate target every C step
    #self.update_target()
    self.polyak_update(self.Q, self.target_Q, self.config["polyak_factor_Q"] )
    self.polyak_update(self.parameters_net, self.parameters_net_target, self.config["polyak_factor_param"] )
    #------------other--------------
    self.lossQ = loss_t
    self.lossP = loss_p


  def update_epsilon(self):
    self.epsilon = self.config["EPS_END"] + (self.config["EPS_START"] - self.config["EPS_END"]) * \
        math.exp(-1. * self.steps_done / self.config["EPS_DECAY"])

    self.steps_done +=1 

  def polyak_update(self, network, network_target, factor):
    polyak_factor = factor
    for target_param, param in zip(network_target.parameters(), network.parameters()):
        target_param.data.copy_(polyak_factor*param.data + target_param.data*(1.0 - polyak_factor))
    #self.target_Q.load_state_dict(target_param)


  def update_target(self):
    self.target_update_counter +=1
    if self.target_update_counter > self.config["target_update"]:
       self.target_Q.load_state_dict(self.Q.state_dict())
       #self.polyak_update()
       self.target_update_counter = 0

       
  def reward_shipping(self, reward):
    if reward > 1:
      return 1
    if reward < -1:
      return -1
    return reward