import numpy as np
from ddpg import DDPG
from tqdm import tqdm
from environment import LearningEnvironment


PATH = 'models'
EPISODES = 10000



if __name__=='__main__':
    # Setup
    env = LearningEnvironment()
    agent = DDPG(env)
    agent.restore_model(PATH)

    # Train
    for episode in range(EPISODES):
        done = False
        state = env.reset()
        rewardEpisode = []
        shouldRender = (episode%10==0)

        for i in tqdm(range(env.max_steps)):
            action = agent.noise_action(state)
            next_state, reward, done, info = env.step(action)
            if shouldRender:
                env.render()
            else:
                agent.perceive(state,action,reward,next_state,done)
            # Setup for next cycle
            state = next_state
            rewardEpisode.append(reward)
            averageReward = np.mean(rewardEpisode)

        # Save model
        if episode%100==1:
            agent.save_model(PATH,episode)

        print("Episode {0}, Reward {1}".format(episode,averageReward))
