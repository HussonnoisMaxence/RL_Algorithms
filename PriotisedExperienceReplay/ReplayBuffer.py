
from collections import deque
import numpy as np
import random
import sys
import math



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