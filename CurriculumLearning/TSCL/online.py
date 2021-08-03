import random
import math
import numpy as np

class Online(object):
    """docstring for Online"""
    def __init__(self, task_list, learning_rate, epsilon_decay, epsilon_max, epsilon_min):
        super(Online, self).__init__()
        self.task_list = task_list
        
        self.learning_rate = learning_rate

        self.Q = [0 for task in task_list]

        self.epsilon = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0
        self.tau = 0.0001
        self.last_reward = 0

    def update_epsilon(self):
        self.epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * \
            math.exp(-1. * self.steps_done / self.epsilon_decay)

        self.steps_done +=1 

    def get_boltzmann(self):
        b = sum([np.exp(self.Q[i]/self.tau) for i in range(len(self.task_list))])
        a = [np.exp(self.Q[i]/self.tau) for i in range(len(self.task_list))]/b
        return a

    def get_results(self, task, new_reward):
        reward = new_reward - self.last_reward
        self.last_reward = new_reward
        action = self.task_list.index(task)
        self.Q[action] = self.learning_rate*reward + (1-self.learning_rate)*self.Q[action]

    def choose_task(self):

        task = np.random.choice(self.task_list,p=self.get_boltzmann())
        #self.update_epsilon()
        return task

def main():
    model = Online([1,2,3], 0.001, 100, 0, 0.01)
    for i in range(100000):
        task = model.choose_task()
        reward = np.random.randint(-1,1)
        model.get_results(task, reward)
        
        print(task, reward, model.epsilon)
    print(model.get_boltzmann())
if __name__ == '__main__':
    main()