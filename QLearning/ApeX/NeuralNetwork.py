import torch.nn as nn
import torch.optim as optim
import torch


class DuelingNetwork(nn.Module):
	def __init__(self, obs_size, n_actions, learning_rate, hidden_size=128):
		super(DuelingNetwork, self).__init__()

		self.net_value = nn.Sequential(
			nn.Linear(obs_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, 1)
		)
		self.net_advantage = nn.Sequential(
			nn.Linear(obs_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, n_actions)
		)
		self.optimizer = optim.Adam(params=self.parameters(), 
									lr=learning_rate)
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

		self.to(self.device)
		
	def forward(self, x):
		values = self.net_value(x)
		advantages = self.net_advantage(x)
		qvals = values + (advantages - advantages.mean())
		
		return qvals