import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class QNetwork(nn.Module):
    def __init__(self, config):
        super(QNetwork, self).__init__()

        input_shape = config["input_shape"]
        hidden_size = config["hidden_size"]
        n_actions = config["n_actions"]
        learning_rate = config["learning_rate"]
        


        self.net = nn.Sequential(
            nn.Linear(input_shape + n_actions, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        
        )
        self.optimizer = optim.Adam(params=self.parameters(),  #optim.RMSprop(self.parameters())
                                    lr=learning_rate)

    def forward(self, x, x_params):

        x_cat = torch.cat((x,x_params), 1-len(x_params.size()))

        out = self.net(x_cat)

        return out




class PNetwork(nn.Module):
    def __init__(self, config):
        super(PNetwork, self).__init__()

        input_shape = config["input_shape"]
        hidden_size = config["hidden_size"]
        n_actions = config["n_actions"]
        learning_rate = config["learning_rate_param"]
        
        self.ev = config["ev"]

        self.net = nn.Sequential(
            nn.Linear(input_shape, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
            
        )
        self.optimizer = optim.Adam(params=self.parameters(),  #optim.RMSprop(self.parameters())
                                    lr=learning_rate)
        self.sfm = nn.Softmax(dim=0)
    def forward(self, x):
        x_params = self.net(x)
        if self.ev == "CP":
            return self.sfm(x_params)
        elif self.ev == "LL":
            return torch.tanh(x_params)