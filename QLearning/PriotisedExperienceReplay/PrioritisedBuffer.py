
from collections import deque
import numpy as np
import random
import sys
import math

from collections import namedtuple
Transition = namedtuple('Transition',
                        ('state', 'action','reward', 'next_state', 'done'))


class PrioritisedExperienceReplay(object):

    def __init__(self, config):
        self.capacity = config["capacity"]
        self.memory = []
        self.priorities = []
        self.position = 0

        self.index_sample = None

        ###
        self.epsilon = config["epsilon"]
        self.alpha = config["alpha"]
        self.beta = config["beta"]
        self.beta_inc = config["beta_inc"]
        ###

    def init_priority(self):
        if self.position:
            priority = max(self.get_priorities())         
        else:
            priority = 1
            
        return priority



    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
            self.priorities.append(None)

        self.memory[self.position] = Transition(*args)
        self.priorities[self.position] = self.init_priority()
        self.position = (self.position + 1) % self.capacity



    def update_error(self, error):
        p = (abs(error.detach().cpu().numpy() ) + self.epsilon)**(self.alpha)
        np.array(self.priorities)[self.index_sample] = p


    def get_priorities(self):
        return self.priorities[:-1]

    def get_probs(self):
        priorities = self.get_priorities()
        return np.array(priorities)/sum(priorities)

    def sample(self, batch_size):

        probability = self.get_probs()
        N = self.position -1
        index_sample = np.random.choice(a=N, size=batch_size, p=probability, replace=True)    

        mini_batch = np.array(self.memory)[index_sample]

        pr = np.array(probability)[index_sample]
        weight = (N*pr)**(-self.beta)
        isw =  weight/max(weight)
        self.beta += self.beta_inc*self.beta
        self.index_sample = index_sample
        return mini_batch, isw
        



    def __len__(self):
        return len(self.memory)



class ReplayBuffer:
    def __init__(self, size_buffer, alpha=0.6, beta=0.4, beta_inc=0.0001, epsilon=0.00001):

        self.transitions = deque(maxlen=size_buffer)
        self.priority = deque(maxlen=size_buffer)

        self.alpha = alpha
        self.beta = beta
        self.beta_inc = beta_inc
        self.epsilon = epsilon

        self.index_sample = None

    
    def add_experience(self, experience):
        priority = max(self.priority) if self.priority  else 1
    
        self.priority.append(priority)
        self.transitions.append(experience)
  
    def update_error(self, error):
        
        p = (abs(error.detach().cpu() ) + self.epsilon)**(self.alpha)

        np.array(self.priority)[self.index_sample] = p

    def get_batch(self, size):
        self.probability = np.array(self.priority)/sum(self.priority)
        N = len(self.transitions)

        index_sample = np.random.choice(N, size, p=self.probability, replace=True)    
        mini_batch = np.array(self.transitions)[index_sample]

        pr = np.array(self.probability)[index_sample]
        weight = (N*pr)**(-self.beta)
        isw =  weight/max(weight)
        self.index_sample = index_sample
        self.beta += self.beta_inc*self.beta

        return mini_batch, isw