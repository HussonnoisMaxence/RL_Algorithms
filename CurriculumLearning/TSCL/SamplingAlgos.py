import random
from collections import namedtuple
import numpy as np
Transition = namedtuple('Transition',
                        ('reward'))


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

class SamplingBuffer(object):
    def __init__(self, N_Task, capacity):
        self.N_Task =N_Task
        self.buffers = [mini_buffer(capacity) for n in range(N_Task)]

    def push(self, action, new_reward):
        if len(self.buffers[action]) == 0:
            last_reward = 0
        else:
            last_reward = self.buffers[action].memory[-1]

        reward = new_reward - last_reward
        self.buffers[action].push(reward)


    def choose_action(self):
        t = self.buffers[0].sample(1)
        rewards = [buf.sample(1)[0] for buf in self.buffers]
        return np.argmax(np.array(rewards))


def main():
    teacher = SamplingBuffer(3, 4)
    for i in range(300):
        a = teacher.choose_action()
        print(a)
        teacher.push(a, np.random.randint(-1,1))

if __name__ == '__main__':
    main()
