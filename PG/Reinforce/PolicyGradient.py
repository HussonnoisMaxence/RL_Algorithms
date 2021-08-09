import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        input_shape = 4
        hidden_size = 256
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





class Agent(object):
	"""docstring for Agent"""
	def __init__(self):
		super(Agent, self).__init__()
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		self.policy = Network().to(self.device)
		self.optimizer = optim.SGD(params=self.policy.parameters(),
									lr=0.00005)
		
		self.gamma = 1
		self.loss = None

	def choose_action(self, obs):
		probs = self.policy(torch.tensor(obs, dtype=torch.float, device=self.device).unsqueeze(0))
		m = torch.distributions.categorical.Categorical(probs)
		
		action = m.sample().detach().item()
		return action


	def gamma_mat(self, size):
		mat = []
		for i in range(size):
			t = [gamma]

	def compute_loss_T(self, T):
		states = torch.tensor([s for (s,a,r,ns) in T], dtype=torch.float, device=self.device) 
		actions = torch.tensor([a for (s,a,r,ns) in T], dtype=torch.long, device=self.device) 
		rewards = torch.tensor([r for (s,a,r,ns) in T], dtype=torch.long, device=self.device) 
		log_probs = torch.log(self.policy(states).gather(dim=1, index=actions.unsqueeze(-1))).squeeze(-1)


		G = torch.tensor([
			torch.sum(
					torch.tensor([r*self.gamma**(t) for t, r in enumerate(rewards[i:])])
				)
			for i in range(len(T))
			], dtype=torch.float, device=self.device)

		loss = -torch.sum(log_probs*G)
		return loss

	def train(self, batch_T):
		batch_loss = torch.mean(torch.stack([self.compute_loss_T(T) for T in batch_T]))
		self.loss = batch_loss
		self.optimizer.zero_grad()
		batch_loss.backward()
		self.optimizer.step()
		"""
		for i, (s,a,r,ns) in enumerate(T):
			probs = self.policy(torch.tensor(s, dtype=torch.float, device=self.device).unsqueeze(0)).squeeze(0)
			log = torch.log(probs)[a]
			G = sum([rs*self.gamma**(t) for t, rs in enumerate(rewards[i:])])
			U.append(log*G)
		
		loss = - torch.sum(torch.stack(U))
		self.loss = loss
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		
		"""