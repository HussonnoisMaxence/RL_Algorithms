import random
import math
import numpy as np
from sklearn import linear_model
class mini_buffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = args
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.memory) == 0:
            return [1]
        else:
            return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Window(object):
    """docstring for Online"""
    def __init__(self, task_list, learning_rate, K):
        super(Window, self).__init__()
        self.task_list = task_list
        self.buffersD = [mini_buffer(K) for n in self.task_list]
        self.buffersE = [mini_buffer(K) for n in self.task_list]
        
        self.learning_rate = learning_rate

        self.Q = [0 for task in task_list]



        self.steps_done = 0

        self.tau = 0.0001


        self.last_reward = 0
        self.reward_list = []
        self.K = K
        self.k = 0
        self.task = None



    def get_boltzmann(self):
        b = sum([np.exp(self.Q[i]/self.tau) for i in range(len(self.task_list))])
        a = [np.exp(self.Q[i]/self.tau) for i in range(len(self.task_list))]/b
       
        return a

    def get_results(self, task, new_reward, timestep):
        action = self.task_list.index(task)
        self.buffersD[action].push(new_reward)
        self.buffersE[action].push(timestep)

        timestep_l = self.buffersE[action].memory
        re_l = self.buffersD[action].memory
        regr = linear_model.LinearRegression()
        regr.fit(np.array(timestep_l,dtype=object).reshape(-1, 1), 
                np.array(re_l, dtype=object).reshape(-1, 1))
        reward = regr.coef_[0][0]
        self.Q[action] = self.learning_rate*reward + (1-self.learning_rate)*self.Q[action]


    def choose_task(self):
        task = np.random.choice(self.task_list,p=self.get_boltzmann())
        return task
        
def main():
    model = Window([1,2,3], 0.001, 5)
    for i in range(1000):
        task = model.choose_task()
        reward = np.random.randint(-1,1)
        model.get_results(task, reward, i)
        
        print(task, reward)
    print(model.get_boltzmann())
if __name__ == '__main__':
    main()