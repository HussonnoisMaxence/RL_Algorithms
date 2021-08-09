import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class Network(nn.Module):
    def __init__(self, config):
        super(Network, self).__init__()

        input_shape = config["input_shape"]
        hidden_size = config["hidden_size"]
        n_actions = config["n_actions"]
        learning_rate = config["learning_rate"]
        


        self.net = nn.Sequential(
            nn.Linear(input_shape , hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        
        )

    def forward(self, x):
        out = self.net(x)
        return out

