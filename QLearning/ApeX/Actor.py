import torch
import numpy as np
import time
import random, sys
from NeuralNetwork import DuelingNetwork

class Actor:
	def __init__ (self, obs_size, n_actions, epsilon_min, epsilon=0.9, 
				epsilon_decay=0.995, update_network=2, update_replay=50):
		#Define env
		self.epsilon = epsilon
		self.epsilon_min = epsilon_min
		self.epsilon_decay = epsilon_decay
		self.local_replay = []
		
		self.Q = DuelingNetwork(obs_size, n_actions,learning_rate=0.01)
		self.n_actions = n_actions
		self.update_network = update_network
		self.update_replay = update_replay
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	def choose_action(self, obs):
		#Choose a action according to greedy epsilon
		if np.random.uniform() < self.epsilon:
			action = np.random.choice(self.n_actions)
		else:
			action = torch.argmax(self.Q(torch.tensor(obs, device=self.device, dtype=torch.float))).item()
		
		return action

	def update_epsilon(self):
		self.epsilon *= self.epsilon_decay
		self.epsilon = max(self.epsilon, self.epsilon_min)

	def add_transition(self, obs, action, reward, next_obs, done):
		self.local_replay.append([obs, action, self.reward_clipping(reward), next_obs, done]) 

	def reward_clipping(self, reward):
		if reward > 1:
			reward = 1
		elif reward <-1:
			reward = -1
		return reward



def run(replay, learner, server, env, episode_batch, epsilon_min):

	actor = Actor(obs_size=env.observation_space.shape[0], n_actions=env.action_space.n, 
				epsilon_min=epsilon_min)

	update_network_count = 0
	for episode in range(episode_batch):
		obs = env.reset()
		done = False
		total_reward = 0
		update_network_count +=1 

		while not(done):
			action = actor.choose_action(obs)
			#perform action
			next_obs, reward, done, _ = env.step(action)           
			#store transition in replay
			actor.add_transition(obs, action, reward, next_obs, done)  
			obs = next_obs
			total_reward += reward  
			actor.update_epsilon()

		if len(actor.local_replay) > actor.update_replay:
			td_errors = learner.compute_td(actor.local_replay)
			replay.add_all(actor.local_replay, td_errors)# appel remote object method
			actor.local_replay = []

		if update_network_count > actor.update_network:
			params = server.get_params()            
			if params:
				for q_param, param in zip(actor.Q.parameters(),
										   params):
					new_param = torch.Tensor(param)
					q_param.data.copy_(new_param)
				print("get")
			update_network_count = 0
			

		print("step:", episode,"reward:", total_reward,"eps:", actor.epsilon)
		time.sleep(1)