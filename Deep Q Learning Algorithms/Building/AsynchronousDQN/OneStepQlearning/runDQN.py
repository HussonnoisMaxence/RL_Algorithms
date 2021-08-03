import gym, torch
import torch.multiprocessing as mp
from DQN import AgentDQN as Agent
from torch import optim
import torch
from NeuralNetwork import Network
import numpy as np
from utils import read_config
import random
import math

def ensure_shared_grads(model, shared_model):
		for param, shared_param in zip(model.parameters(),
																	 shared_model.parameters()):
				if shared_param.grad is not None:
						return
				shared_param._grad = param.grad

def train(shared_model, config):
	model = Network(config)
	model.load_state_dict(shared_model.state_dict())
	target_network = Network(config)
	target_network.load_state_dict(shared_model.state_dict())
	env = gym.make("CartPole-v0")

	optimizer = torch.optim.Adam(params=shared_model.parameters(), lr=config["learning_rate"])

	epsilon = config["EPS_START"]
	steps = 0
	for episode in range(config["episode_batch"]):
			# Sync with the shared model
			

			#initialized sequence
			obs = env.reset()
			done = False
			total_reward = 0

			while not(done):

				#choose an action
				#--------------------------
				if random.random() < epsilon:
					action = np.random.choice(config["n_actions"])

				else:
					qvals = model(torch.tensor(obs, dtype=torch.float).unsqueeze(0))
					action = torch.argmax(qvals).detach().item()
					#print(qvals)
				#--------------------------

				#perform action in environment
				next_obs, reward, done, _ =  env.step(action)


				#Compute gradient
				#------------------------------------------------------------------------------
				## transform transition into tensor
				obs_t       = torch.tensor(obs, dtype=torch.float)
				action_t    = torch.tensor(action, dtype=torch.long)
				reward_t    = torch.tensor(reward, dtype=torch.float)
				next_obs_t  = torch.tensor(next_obs, dtype=torch.float)
				done_t      = torch.tensor(not(done), dtype=torch.bool)
				## compute target value
				next_state_values = target_network(next_obs_t)
				next_state_values = next_state_values.max(0)[0].detach()
				target_values = reward_t + config["gamma"]*next_state_values*done_t

				## compute estimated value
				values_p = model(obs_t)
				values = values_p.gather(dim=0,index=action_t.unsqueeze(-1)).squeeze(-1)

				## compute loss
				loss = torch.nn.SmoothL1Loss()(values, target_values) #MSE loss for Q network
				loss.backward()

				## update target model
				polyak_factor = config["polyak_factor_Q"]
				for target_param, param in zip(target_network.parameters(), model.parameters()):
					target_param.data.copy_(polyak_factor*param.data + target_param.data*(1.0 - polyak_factor))
				"""
				if not((steps+1)%config["target_update"]):
					target_network.load_state_dict(shared_model.state_dict())
				"""
				## update shared model
				if not((steps + 1)%config["shared_update"]):
					print(steps)
					ensure_shared_grads(model, shared_model)
					optimizer.step()
					optimizer.zero_grad()
					model.load_state_dict(shared_model.state_dict())
				#------------------------------------------------------------------------------
				##update epsilon
				epsilon = config["EPS_END"] + (config["EPS_START"] - config["EPS_END"]) * \
								math.exp(-1. * steps / config["EPS_DECAY"])
				#next ...
				obs = next_obs
				total_reward += reward
				steps += 1

			
			print("step:", episode,"reward:", total_reward,"eps:", epsilon)


if __name__ == "__main__":
	config = read_config("./config.yaml")
	torch.multiprocessing.set_start_method('spawn', force=True)
	num_processes = 3

	model = Network(config)
	model.share_memory()
	processes = []

	t = model.net[0].weight[1]
	print(t)
	for rank in range(num_processes):
				p = mp.Process(target=train, args=(model,config, ))
				p.start()
				processes.append(p)

	for p in processes:
				p.join()


	#print(t)
	#print("-")
	print(model.net[0].weight[1])
