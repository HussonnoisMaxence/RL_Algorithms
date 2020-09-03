import torch.nn as nn
import torch.optim as optim



class Network(nn.Module):
    def __init__(self, obs_size, n_actions, learning_rate, hidden_size=128):
        super(Network, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )
        self.optimizer = optim.Adam(params=self.parameters(), 
                                    lr=learning_rate)
        
    def forward(self, x):
        return self.net(x)