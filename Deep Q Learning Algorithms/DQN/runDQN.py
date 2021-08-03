import gym, torch
from DQN import AgentDQN as Agent

from utils import read_config
if __name__ == "__main__":
  config = read_config("./config.yaml")

  episodes_batch = 50

  env = gym.make("CartPole-v0")

  agent = Agent(config)
  
  for episode in range(config["episode_batch"]):

      #initialized sequence
      obs = env.reset()
      done = False
      total_reward = 0

      while not(done):
        action = agent.choose_action(obs)
        #perform action
        next_obs, reward, done, _ =  env.step(action)           
        #store transition in replay
        agent.memory.push(torch.tensor(obs, device=agent.device, dtype=torch.float), 
                                torch.tensor(action, device=agent.device, dtype=torch.long),
                                torch.tensor(reward, device=agent.device, dtype=torch.float), 
                                torch.tensor(next_obs, device=agent.device, dtype=torch.float), 
                                torch.tensor(not(done), device=agent.device, dtype=torch.bool)
                                )
        #train neural network
        agent.train_nn()


        obs = next_obs
        total_reward += reward

     
      print("step:", episode,"reward:", total_reward,"eps:", agent.epsilon)
  