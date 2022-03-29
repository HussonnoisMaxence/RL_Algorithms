import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import torch as T
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, input_shape, arch, init=0.0003):
        super(QNetwork, self).__init__()


        

        self.network = []
        self.activation_fuction = F.relu
        self.network.append(nn.Linear(input_shape, arch[0]))
        for index in range(len(arch)-1):
            self.network.append(nn.Linear(arch[index], arch[index+1]))

        self.network.append(nn.Linear(arch[-1], 1))

        
        self.network = nn.Sequential(*self.network)
        self.network[-1].weight.data.uniform_(-init, +init)


        self.network[-1].bias.data.uniform_(-init, +init)
        
    def forward(self, x, actions):
        x = torch.cat((x, actions), 1-len(actions.size()))

        for i in range(len(self.network)-1):
            x = self.network[i](x)
            x = self.activation_fuction(x)

        x = self.network[-1](x)

        return x

class VNetwork(nn.Module):
    def __init__(self, input_shape, arch, init=0.0003):
        super(VNetwork, self).__init__()
        self.network = []
        self.activation_fuction = F.relu
        self.network.append(nn.Linear(input_shape, arch[0]))
        for index in range(len(arch)-1):
            self.network.append(nn.Linear(arch[index], arch[index+1]))

        self.network.append(nn.Linear(arch[-1], 1))

        
        self.network = nn.Sequential(*self.network)
        self.network[-1].weight.data.uniform_(-init, +init)


        self.network[-1].bias.data.uniform_(-init, +init)
        
    def forward(self, x):

        for i in range(len(self.network)-1):
            x = self.network[i](x)
            x = self.activation_fuction(x)

        x = self.network[-1](x)

        return x

class PolicyNetwork(nn.Module):
    def __init__(self, input_shape, arch, out_shape, max_action,reparam_noise = 1e-6, init=0.0003):
        super(PolicyNetwork, self).__init__()

        self.max_action = max_action
        self.reparam_noise = reparam_noise

        self.network = []
        self.activation_fuction = F.relu
        self.network.append(nn.Linear(input_shape, arch[0]))
        for index in range(len(arch)-1):
            self.network.append(nn.Linear(arch[index], arch[index+1]))
        
        self.network = nn.Sequential(*self.network)
        self.mean = nn.Sequential(nn.Linear(arch[-1], out_shape))
        
        
        self.std = nn.Sequential(nn.Linear(arch[-1], out_shape))
        
        
        
        self.mean[0].weight.data.uniform_(-init, +init)
        self.std[0].weight.data.uniform_(-init, +init)
        self.mean[0].bias.data.uniform_(-init, +init)
        self.std[0].bias.data.uniform_(-init, +init)

       

    def forward(self, x, reparameterize):

        for i in range(len(self.network)):
            x = self.network[i](x)
            x = self.activation_fuction(x)

        mean = self.mean(x)
        std = torch.clamp(self.std(x), min=self.reparam_noise, max=1)

        probabilities = Normal(mean, std)

        if reparameterize:
            z = Normal(0,1).sample()
            actions = mean +std*z
            action = T.tanh(actions)*T.tensor(self.max_action)


        else:
            actions = probabilities.sample()
            action = T.tanh(actions)*T.tensor(self.max_action)
        
        log_probs = probabilities.log_prob(actions)
        log_probs -= T.log(1-action.pow(2)+self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)
        return action, log_probs
