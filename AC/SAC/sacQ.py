
import torch
from networks import PolicyNetwork, QNetwork, VNetwork
from Buffers import ReplayMemory, Transition
class SAC:
    """docstring for SAC"""
    def __init__(self, env, learning_rate_Q_functions=0.01, learning_rate_policy=0.01, gamma=0.99, 
        memory_capacity=50_000, temperature=0.5, polyak_factor=0.5, min_replay_size=2, batch_size=2, train_freq=2, gradient_steps=32):
        super (SAC, self).__init__()
        self.env = env
        self.observation_shape = self.env.observation_space.shape[0] #need to do it properly for image observation or other stuff
        self.action_space = self.env.action_space.shape[0]
        print("oh", self.action_space)
        #Initiate networks
        self.policy = PolicyNetwork(input_shape=self.observation_shape, hidden_size=256, out_shape=self.action_space,max_action=env.action_space.high)

        self.Q1 = QNetwork(input_shape=self.observation_shape+self.action_space, hidden_size=256)
        self.Q2 = QNetwork(input_shape=self.observation_shape+self.action_space, hidden_size=256)

        self.target_Q1 = QNetwork(input_shape=self.observation_shape+self.action_space, hidden_size=256)
        self.target_Q2 = QNetwork(input_shape=self.observation_shape+self.action_space, hidden_size=256)
        
        self.polyak_update(self.Q1, self.target_Q1, 1)
        self.polyak_update(self.Q2, self.target_Q2, 1)    
        #Initiate optimizer
        self.Q1_optimizer = torch.optim.Adam(params=self.Q1.parameters(), lr=learning_rate_Q_functions)
        self.Q2_optimizer = torch.optim.Adam(params=self.Q2.parameters(), lr=learning_rate_Q_functions)
        self.policy_optimizer = torch.optim.Adam(params=self.policy.parameters(), lr=learning_rate_policy)

        #Initiate replay memory
        self.memory = ReplayMemory(capacity=memory_capacity)
        self.min_replay_size = min_replay_size
        self.batch_size = batch_size

        #Training hyperparameters
        self.gamma = gamma
        self.temperature = temperature
        self.scale = 1/temperature
        self.polyak_factor_Q = polyak_factor
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps

        self.losses=None

    def learn(self, total_timesteps):
        self.timesteps = 0
        ep = 0
        while self.timesteps <= total_timesteps:
            #Initiate the episode
            reward_total = 0
            obs = self.env.reset()
            done = False
            
            while not(done):
                #Interaction with the environments
                action = self.predict(obs)
                next_obs, reward, done, info = self.env.step(action)

                #Off Policy learning
                self.add_sample(obs, action, reward, next_obs, done)
                self.train()

                #Update values
                obs = next_obs
                reward_total += reward
                self.timesteps += 1
            ep += 1
            print("ep:",ep, "timesteps:", self.timesteps, "reward:", reward_total, "Losses:", self.losses)


    def predict(self, obs ):
        obs = torch.tensor(obs, dtype=torch.float).unsqueeze(0)
        action, _ = self.policy(obs, False)

        return action[0].detach().numpy()
        
    
    def add_sample(self, obs, action, reward, next_obs, done):
        self.memory.push(obs, action, reward, next_obs, not(done))

    def polyak_update(self, network, network_target, factor):
        polyak_factor = factor
        for target_param, param in zip(network_target.parameters(), network.parameters()):
          target_param.data.copy_(polyak_factor*param.data + target_param.data*(1.0 - polyak_factor))

    def train(self):
        """
        if not((self.timesteps+1)%self.train_freq):
            return
        """
        if len(self.memory) < self.min_replay_size:
            return
        for _ in range (1) :#self.gradient_steps):
        #---------TRAINING------------------------
            transitions = self.memory.sample(self.batch_size)
            batch = Transition(*zip(*transitions))
            

            #-- Get each element of the transition in batches
            state_batch = torch.tensor(batch.state, dtype=torch.float)
            next_state_batch = torch.tensor(batch.next_state, dtype=torch.float)
            action_batch = torch.tensor(batch.action, dtype=torch.float)
            reward_batch = torch.tensor(batch.reward, dtype=torch.float)
            done_batch = torch.tensor(batch.done, dtype=torch.float)

            



            
            #--- Training Value function

            self.Q1_optimizer.zero_grad()
            self.Q2_optimizer.zero_grad()

            actions, log_prob = self.policy(next_state_batch, False)

            target_Q1_values = self.target_Q1(next_state_batch, actions).squeeze(1)
            target_Q2_values = self.target_Q2(next_state_batch, actions).squeeze(1)
            target_Qvalue = torch.min(target_Q1_values, target_Q2_values)
            target_Qvalue = self.scale*reward_batch + self.gamma*done_batch*(target_Qvalue-log_prob)

            Q1_values = self.Q1(state_batch, action_batch).squeeze(1)
            Q2_values = self.Q2(state_batch, action_batch).squeeze(1)           

            
            loss = torch.nn.MSELoss()
            Q1_loss = loss(Q1_values, target_Qvalue)
            Q2_loss = loss(Q2_values, target_Qvalue)   

            Q_losses = Q1_loss + Q2_loss
           

            Q_losses.backward(retain_graph=True)
            self.Q1_optimizer.step()
            self.Q2_optimizer.step()
            
            #-----Training policy
            self.policy_optimizer.zero_grad()

            actions, log_prob = self.policy(state_batch, True)
            Q1_values = self.Q1(state_batch, actions)
            Q2_values = self.Q2(state_batch, actions)
            Q_values = torch.min(Q1_values, Q2_values)
            policy_loss = (log_prob - Q_values).mean()

            
            policy_loss.backward()
            self.policy_optimizer.step()


            self.losses = (policy_loss.detach().numpy(),  Q1_loss.detach().numpy(), Q2_loss.detach().numpy())

            
            
            self.polyak_update(self.Q1, self.target_Q1, self.polyak_factor_Q )
            self.polyak_update(self.Q2, self.target_Q2, self.polyak_factor_Q )