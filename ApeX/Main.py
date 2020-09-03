
import multiprocessing
from multiprocessing import Process, Manager
from multiprocessing.managers import BaseManager
import numpy as np

from Learner import Learner
from Actor import run
from PrioritizedReplayBuffer import Memory
from Server import Server

import gym
from AutoProxy import AutoProxy

multiprocessing.managers.AutoProxy = AutoProxy



if __name__ == "__main__":

	epsilons = [0.001, 0.001]
	N_ACTORS = 2
	episode_batch = 100
	envs = []

	for actor in range(N_ACTORS):
		envs.append(gym.make("CartPole-v0"))

	BaseManager.register('ReplayBM', Memory)
	BaseManager.register('LearnerBM', Learner)
	BaseManager.register('ServerBM', Server)
	manager = BaseManager()
	manager.start()
	server = manager.ServerBM()
	replay = manager.ReplayBM()


	learner = manager.LearnerBM(obs_size=envs[0].observation_space.shape[0], n_actions=envs[0].action_space.n)
	
	processes = [Process(target=run, args=(replay, learner, server, envs[p], episode_batch, epsilons[p])) for p in range(N_ACTORS) ]

	p_learner = Process(target=learner.train_nn,  args=(replay, server,))
	
	p_learner.start()
	for p in processes:
	  p.start()
  
	
	p_learner.join()
	for p in processes:
	  p.join()