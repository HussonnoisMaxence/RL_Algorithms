
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import sys
import random


class Memory:
    def __init__(self, size_buffer=1_000_000, alpha=0.6, beta=0.4,
                beta_inc=0.0001, epsilon=0.00001, learning_step=100):
        self.transitions = deque(maxlen = size_buffer)
        self.priority = deque(maxlen = size_buffer)


        self.probability = deque(maxlen=size_buffer)
        self.td_error = deque(maxlen=size_buffer)

        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon

        self.priority_beta_increment = ((1 - self.beta)/ learning_step)

        self.index_sample = None


    def add_all(self, experiences, errors):
        for exp, err in zip(experiences, errors):
            self.priority.append((abs(err )+ self.epsilon)**(self.alpha))
            self.transitions.append(exp)

    def update_error(self, error):
        t = (abs(error )+ self.epsilon)
        np.array(self.priority)[self.index_sample] = t**(self.alpha)


    def get_batch(self, size):
        N = int(self.get_size())
        self.probability = np.array(self.priority)[:N]/sum(np.array(self.priority)[:N])
        index_sample = np.random.choice(N, size, p=self.probability, replace=True)

        t = np.array([x for x in self.transitions])
    
        mini_batch = t[index_sample]
        weight = N*np.array(self.probability)[index_sample]**(-self.beta)
        isw =  weight/max(weight)
        self.index_sample = index_sample
        self.beta += self.priority_beta_increment

        return mini_batch, isw

    def get_size(self):
        return len(self.transitions)