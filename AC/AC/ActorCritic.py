import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        input_shape = 4
        hidden_size = 128
        n_actions = 2
        learning_rate = 0.01
        


        self.net = nn.Sequential(
            nn.Linear(input_shape, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
            nn.Softmax(dim=1)
        
        )

    def forward(self, x):
        out = self.net(x)
        return out


class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()

        input_shape = 4
        hidden_size = 128
        n_actions = 1
        learning_rate = 0.01
        


        self.net = nn.Sequential(
            nn.Linear(input_shape, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        
        )

    def forward(self, x):
        out = self.net(x)
        return out


class Agent(object):
	"""docstring for Agent"""
	def __init__(self):
		super(Agent, self).__init__()
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		self.policy = Network().to(self.device)
		self.Q = QNetwork().to(self.device)
		self.optimizer = optim.Adam(params=self.policy.parameters(),
									lr=0.0003)

		self.Q_optimizer = optim.Adam(params=self.Q.parameters(),
									lr=0.0003)
		self.gamma = 0.99
		self.loss = None

	def choose_action(self, obs):
		probs = self.policy(torch.tensor(obs, dtype=torch.float, device=self.device).unsqueeze(0)).squeeze(0)
		m = torch.distributions.categorical.Categorical(probs)	
		action = m.sample().detach().item()
		return action



	def train(self, T):
		states = torch.tensor([s for (s,a,r,ns,d) in T], dtype=torch.float, device=self.device) 
		actions = torch.tensor([a for (s,a,r,ns,d) in T], dtype=torch.long, device=self.device) 
		rewards = torch.tensor([r for (s,a,r,ns,d) in T], dtype=torch.long, device=self.device) 
		next_states = torch.tensor([ns for (s,a,r,ns,d) in T], dtype=torch.float, device=self.device) 
		dones = torch.tensor([not(d) for (s,a,r,ns,d) in T], dtype=torch.bool, device=self.device)
		log_probs = torch.log(self.policy(states).gather(dim=1, index=actions.unsqueeze(-1))).squeeze(-1)

		Qvalues = self.Q(states).squeeze(1)

		



		next_states_values = self.Q(next_states).squeeze(-1)
		target_values = rewards +self.gamma*next_states_values

		loss = -torch.sum(log_probs*(target_values - Qvalues))
		Qloss = nn.MSELoss()(Qvalues, target_values)

		self.loss = loss
		self.optimizer.zero_grad()
		loss.backward(retain_graph=True)
		self.optimizer.step()

		self.Q_optimizer.zero_grad()
		Qloss.backward()
		self.Q_optimizer.step()