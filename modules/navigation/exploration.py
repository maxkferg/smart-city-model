import random
from dqn.schedule import LinearExploration


class GuidedExploration(LinearExploration):
    def __init__(self, env, eps_begin, eps_end, nsteps):
        """
        Args:
            env: gym environment
            eps_begin: initial exploration
            eps_end: end exploration
            nsteps: number of steps between the two values of eps
        """
        super().__init__(env, eps_begin, eps_end, nsteps)


    def get_action(self, best_action):
        """
        Returns a random action with prob epsilon, otherwise return the best_action

        Args:
            best_action: (int) best action according some policy
        Returns:
            an action
        """
        # If epsilon low we choose the best_action
        if random.random()>self.epsilon:
            return best_action

        if random.random() > 0.4:
            return self.env.get_best_action()

        return self.env.action_space.sample()

        ##############################################################
        ######################## END YOUR CODE ############## ########

