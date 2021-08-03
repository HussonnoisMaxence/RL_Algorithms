import random
import math
import numpy as np
from sklearn import linear_model

class Naive(object):
    """docstring for Online"""
    def __init__(self, task_list, learning_rate, epsilon_decay, epsilon_max, epsilon_min, K):
        super(Naive, self).__init__()
        self.task_list = task_list
        
        self.learning_rate = learning_rate

        self.Q = [0 for task in task_list]

        self.epsilon = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0
        self.tau = 0.001
        self.last_reward = 0
        self.reward_list = []
        self.K = K
        self.k = 0
        self.task = None

    def update_epsilon(self):
        self.epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * \
            math.exp(-1. * self.steps_done / self.epsilon_decay)

        self.steps_done +=1 

    def get_boltzmann(self):
        b = sum([np.exp(self.Q[i]/self.tau) for i in range(len(self.task_list))])
        a = [np.exp(self.Q[i]/self.tau) for i in range(len(self.task_list))]/b
       
        return a

    def get_results(self, task, new_reward):
        if self.k == self.K:
            regr = linear_model.LinearRegression()
            regr.fit(np.array([k for k in range(self.K-1)],dtype=object).reshape(-1, 1), 
                    np.array(self.reward_list, dtype=object).reshape(-1, 1))
            reward = regr.coef_[0][0]
            action = self.task_list.index(task)
            self.Q[action] = self.learning_rate*reward + (1-self.learning_rate)*self.Q[action]
        else:
            self.reward_list.append(new_reward)

    def choose_task(self):
        if self.k == 0:
            if random.random() < self.epsilon:
                action = np.random.choice(len(self.task_list))
                task = self.task_list[action]
            else:
                task = np.random.choice(self.task_list,p=self.get_boltzmann())
            #self.update_epsilon()
            self.last_task = task
        else:
            task = self.last_task

        self.k += 1
        return task
        
def main():
    model = Naive([1,2,3], 0.001, 100, 0, 0.01, 5)
    for i in range(1000):
        task = model.choose_task()
        reward = np.random.randint(-1,1)
        model.get_results(task, reward)
        
        print(task, reward)
    print(model.get_boltzmann())
if __name__ == '__main__':
    main()