from dqn.schedule import LinearExploration, LinearSchedule
from dqn.nature import NatureQN
from dqn.configs.nature import config
from environment import LearningEnvironment


if __name__=='__main__':
    # Setup
    env = LearningEnvironment(num_particles=1,disable_render=False)
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