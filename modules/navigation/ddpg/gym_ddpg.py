import filter_env
from ddpg import *
import gc
gc.enable()
import matplotlib.pyplot as plt

ENV_NAME = 'MountainCarContinuous-v0'
PATH = 'experiments/' + ENV_NAME + '-E4/'
EPISODES = 100000
TEST = 5

def main():
    env = filter_env.makeFilteredEnv(gym.make(ENV_NAME))
    agent = DDPG(env)
    #env.monitor.start('experiments/' + ENV_NAME,force=True)
    env = gym.wrappers.Monitor(env, PATH, force=True)

    returns = []
    rewords = []

    for episode in xrange(EPISODES):
        state = env.reset()
    reward_episode = []
        print "episode:",episode
        # Train
        for step in xrange(env.spec.timestep_limit):
            env.render()
            action = agent.noise_action(state)
            next_state,reward,done,_ = env.step(action)
            #print('state={}, action={}, reward={}, next_state={}, done={}'.format(state, action, reward, next_state, done))
            agent.perceive(state,action,reward,next_state,done)
            state = next_state
            reward_episode.append(reward)
        if done:
                break

    plt.figure(3)
    plt.plot(reward_episode)
    plt.show()

        # Testing:
        #if episode % 1 == 0:
        if episode % 10 == 0 and episode > 50:
            agent.save_model(PATH, episode)

            total_return = 0
            ave_reward = 0
            for i in xrange(TEST):
                state = env.reset()
                reward_per_step = 0
                for j in xrange(env.spec.timestep_limit):
                    #env.render()
                    action = agent.action(state) # direct action for test
                    state,reward,done,_ = env.step(action)
                    total_reward += reward
                    if done:
                        break
                    reward_per_step += (reward - reward_per_step)/(j+1)
                ave_reward += reward_per_step

            ave_return = total_return/TEST
            ave_reward = ave_reward/TEST
            returns.append(ave_return)
            rewards.append(ave_reward)

            plt.figure(1)
            plt.plot(returns)
            plt.figure(2)
            plt.plot(rewards)

            plt.show()

            print 'episode: ',episode,'Evaluation Average Return:',ave_return, '  Evaluation Average Reward: ', ave_reward
    env.monitor.close()

if __name__ == '__main__':
    main()
