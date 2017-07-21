import random
import numpy as np
from .utils.test_env import EnvTest


class LinearSchedule(object):
    def __init__(self, eps_begin, eps_end, nsteps):
        """
        Args:
            eps_begin: initial exploration
            eps_end: end exploration
            nsteps: number of steps between the two values of eps
        """
        self.epsilon        = eps_begin
        self.eps_begin      = eps_begin
        self.eps_end        = eps_end
        self.nsteps         = nsteps


    def update(self, t):
        """
        Updates epsilon

        Args:
            t: (int) nth frames
        """
        ##############################################################
        """
        TODO: modify self.epsilon such that
               for t = 0, self.epsilon = self.eps_begin
               for t = self.nsteps, self.epsilon = self.eps_end
               linear decay between the two

              self.epsilon should never go under self.eps_end
        """
        ##############################################################
        ################ YOUR CODE HERE - 3-4 lines ##################
        self.epsilon = self.eps_begin - t * (self.eps_begin - self.eps_end) / float(self.nsteps)
        self.epsilon = max(self.epsilon, self.eps_end)
        self.epsilon = min(self.epsilon, self.eps_begin)
        ##############################################################
        ######################## END YOUR CODE ############## ########


class LinearExploration(LinearSchedule):
    def __init__(self, env, eps_begin, eps_end, nsteps):
        """
        Args:
            env: gym environment
            eps_begin: initial exploration
            eps_end: end exploration
            nsteps: number of steps between the two values of eps
        """
        self.env = env
        super(LinearExploration, self).__init__(eps_begin, eps_end, nsteps)


    def get_action(self, best_action):
        """
        Returns a random action with prob epsilon, otherwise return the best_action

        Args:
            best_action: (int) best action according some policy
        Returns:
            an action
        """
        ##############################################################
        """
        TODO: with probability self.epsilon, return a random action
               else, return best_action

               you can access the environment stored in self.env
               and epsilon with self.epsilon
        """
        ##############################################################
        ################ YOUR CODE HERE - 4-5 lines ##################
        if random.random()<self.epsilon:
            return self.env.action_space.sample()
        return best_action
        ##############################################################
        ######################## END YOUR CODE ############## ########



def test1():
    env = EnvTest((5, 5, 1))
    exp_strat = LinearExploration(env, 1, 0, 10)

    found_diff = False
    for i in range(10):
        rnd_act = exp_strat.get_action(0)
        if rnd_act != 0 and rnd_act is not None:
            found_diff = True

    assert found_diff, "Test 1 failed."
    print("Test1: ok")


def test2():
    env = EnvTest((5, 5, 1))
    exp_strat = LinearExploration(env, 1, 0, 10)
    exp_strat.update(5)
    assert exp_strat.epsilon == 0.5, "Test 2 failed"
    print("Test2: ok")


def test3():
    env = EnvTest((5, 5, 1))
    exp_strat = LinearExploration(env, 1, 0.5, 10)
    exp_strat.update(20)
    assert exp_strat.epsilon == 0.5, "Test 3 failed"
    print("Test3: ok")


def test_update():
    """
    Test the update method
    """
    eps_begin = 0.8
    eps_end = 0.4
    nsteps = 100
    schedule = LinearSchedule(eps_begin, eps_end, nsteps)
    # Without update
    assert schedule.epsilon == eps_begin
    # Midway point
    schedule.update(t=nsteps/2)
    assert schedule.epsilon == (eps_begin + eps_end)/2
    # End point
    schedule.update(t=nsteps)
    assert schedule.epsilon == eps_end
    # Past end point
    schedule.update(t=2*nsteps)
    assert schedule.epsilon == eps_end
    # Initial point
    schedule.update(t=0)
    assert schedule.epsilon == eps_begin
    # Before initial point
    schedule.update(t=-1)
    assert schedule.epsilon == eps_begin
    print("Test update: ok")


def test_get_action():
    n = 1000
    epsilon = 0.3 # Return a random action 30 % of the time
    env = EnvTest((5, 5, 1))
    best_action = 6
    exp_strat = LinearExploration(env, 1, 0, 100)
    exp_strat.epsilon = epsilon
    random.seed(0)
    actions = [exp_strat.get_action(best_action) for i in range(n)]
    num_random_actions = sum([a!=best_action for a in actions])
    assert num_random_actions > n*epsilon-10
    assert num_random_actions < n*epsilon+10
    print("Test get_action: ok")


if __name__ == "__main__":
    test1()
    test2()
    test3()
    test_update()
    test_get_action()