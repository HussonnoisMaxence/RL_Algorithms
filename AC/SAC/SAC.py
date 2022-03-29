
import torch
from networks import PolicyNetwork, QNetwork
from Buffers import ReplayMemory, Transition

from sacA import SAC as SAC_A
from sacQ import SAC as SAC_Q
from sacV import SAC as SAC_V

def SAC (env, sac_version="Automatic_Entropy", learning_rate_Q_functions=0.01, learning_rate_policy=0.01, gamma=0.99, 
        memory_capacity=50_000, polyak_factor=0.5, batch_size=2, train_freq=2, gradient_steps=32):

    if sac_version == "Value_based":
        return SAC_V(env,learning_rate_Q_functions, learning_rate_policy, gamma, 
        memory_capacity, polyak_factor, batch_size, train_freq, gradient_steps)


    if sac_version == "Q_based":
        return SAC_Q(env, learning_rate_Q_functions, learning_rate_policy, gamma, 
        memory_capacity, polyak_factor, batch_size, train_freq, gradient_steps)


    if sac_version == "Automatic_Entropy":
        return SAC_A(env, learning_rate_Q_functions, learning_rate_policy, gamma, 
        memory_capacity, polyak_factor, batch_size, train_freq, gradient_steps)

