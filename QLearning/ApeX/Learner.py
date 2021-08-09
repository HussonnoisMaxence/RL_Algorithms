import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time

from NeuralNetwork import DuelingNetwork

class Learner:
	def __init__(self, obs_size, n_actions, gamma=0.9, learning_rate=0.01, batch_size=20,
				min_replay_size=20, target_update=5, learning_step=10000):
		#Define networks

		self.Q = DuelingNetwork(obs_size, n_actions, learning_rate)
		self.target_Q = DuelingNetwork(obs_size, n_actions, learning_rate)
		self.target_Q.load_state_dict(self.Q.state_dict())
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

		self.gamma = gamma
		self.batch_size = batch_size
		self.min_replay_size = min_replay_size

		self.target_update_counter = 0
		self.target_update = target_update

		self.learning_step = learning_step
	
	def compute_td(self, mini_batch):
		current_states= torch.tensor([transition[0] for transition in mini_batch], device=self.device, dtype=torch.float)

		actions = torch.tensor([transition[1] for transition in mini_batch], device=self.device, dtype=torch.long)
		rewards = torch.tensor([transition[2] for transition in mini_batch], device=self.device, dtype=torch.float)

		new_current_states = torch.tensor([transition[3] for transition in mini_batch], device=self.device, dtype=torch.float)
	  
		dones = torch.tensor([not(transition[4]) for transition in mini_batch], device=self.device, dtype=torch.bool)

		actions_eval = torch.argmax(self.Q(new_current_states), dim=1)
		next_state_values = self.target_Q(new_current_states).gather(dim=1, index=actions_eval.unsqueeze(-1)).squeeze(-1)
		values = rewards + self.gamma*next_state_values*dones

		target_values = self.Q(current_states).gather(dim=1, index=actions.unsqueeze(-1)).squeeze(-1)

		td_error = target_values - values
		return td_error.detach().cpu().numpy()


	def train_nn(self, replay, server):
		updated_step = 0
		while replay.get_size() < self.min_replay_size:
			continue
		print("Start learning")
		while updated_step < self.learning_step:
				
			#Sample random minibatch of transitions from replay
			mini_batch, weight = replay.get_batch(self.batch_size) # appel remote object method
			
			weight = torch.tensor(weight, device=self.device, dtype=torch.float)
			#Split data transitions into multiples tensors
			current_states = torch.tensor([transition[0] for transition in mini_batch], device=self.device, dtype=torch.float)

			actions = torch.tensor([transition[1] for transition in mini_batch], device=self.device, dtype=torch.long)
			rewards = torch.tensor([transition[2] for transition in mini_batch], device=self.device, dtype=torch.float)

			new_current_states = torch.tensor([transition[3] for transition in mini_batch], device=self.device, dtype=torch.float)
				
			dones = torch.tensor([not(transition[4]) for transition in mini_batch], device=self.device, dtype=torch.bool)
			


			actions_eval = torch.argmax(self.Q(new_current_states), dim=1)
			next_state_values = self.target_Q(new_current_states).gather(dim=1, index=actions_eval.unsqueeze(-1)).squeeze(-1)
			values = rewards + self.gamma*next_state_values*dones

			target_values = self.Q(current_states).gather(dim=1,index=actions.unsqueeze(-1)).squeeze(-1)

			td_error = target_values - values

			replay.update_error(td_error.detach().cpu().numpy()) # appel remote object method

			#fit/backpropagation
			self.Q.optimizer.zero_grad()
			loss_function = nn.MSELoss()
			loss_t = loss_function(values*weight, target_values*weight) 
			loss_t.backward()

			self.Q.optimizer.step()

			self.update_target()
			server.update_params(self.return_params()) # appel remote object method
			updated_step += 1

		print("finish")

	def update_target(self):
		self.target_update_counter +=1
		if self.target_update_counter > self.target_update:
			self.target_Q.load_state_dict(self.Q.state_dict())
			self.target_update_counter = 0

	def return_params(self):
		params = []
		for q_param in (self.Q.parameters()):
			params.append(q_param.detach().cpu())
		return params
	