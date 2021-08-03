import torch.nn as nn
import torch.optim as optim

class DuelingNetwork(nn.Module):
    def __init__(self, config):
        super(DuelingNetwork, self).__init__()

        input_shape = config["input_shape"]
        hidden_size = config["hidden_size"]
        n_actions = config["n_actions"]
        learning_rate = config["learning_rate"]
        


        self.net_value = nn.Sequential(
            nn.Linear(input_shape , hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        
        )
        self.net_advantage = nn.Sequential(
            nn.Linear(input_shape , hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        
        )

    def forward(self, x):
        values = self.net_value(x)
        advantages = self.net_advantage(x)
        qvals = values + (advantages - advantages.mean())
        
        return qvals