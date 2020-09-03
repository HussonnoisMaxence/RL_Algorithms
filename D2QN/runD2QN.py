import gym

from D2QN import Agent


if __name__ == "__main__":
  episodes_batch = 50

  env = gym.make("CartPole-v0")

  agent = Agent(obs_size=env.observation_space.shape[0], n_actions=env.action_space.n)

  for episode in range(episodes_batch):

      #initialized sequence
      obs = env.reset()
      done = False
      total_reward = 0

      while not(done):
        action = agent.choose_action(obs)
        #perform action
        next_obs, reward, done, _ =  env.step(action)           
        #store transition in replay
        agent.replay.append((obs, action, reward, next_obs, done))
        #train neural network
        agent.train_nn()


        obs = next_obs
        total_reward += reward

     
      print("step:", episode,"reward:", total_reward,"eps:", agent.epsilon)
      