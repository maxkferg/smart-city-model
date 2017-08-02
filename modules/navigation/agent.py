from dqn.schedule import LinearSchedule
from dqn.nature import NatureQN
from dqn.configs.nature import config
from exploration import GuidedExploration
from environment import LearningEnvironment


"""
Use deep Q network for the Atari game. Please report the final result.
Feel free to change the configurations (in the configs/ folder).
If so, please report your hyperparameters.

You'll find the results, log and video recordings of your agent every 250k under
the corresponding file in the results folder. A good way to monitor the progress
of the training is to use Tensorboard. The starter code writes summaries of different
variables.

To launch tensorboard, open a Terminal window and run
tensorboard --logdir=results/
Then, connect remotely to
address-ip-of-the-server:6006
6006 is the default port used by tensorboard.
"""
if __name__ == '__main__':
    # make env
    env = LearningEnvironment(num_particles=1)

    # exploration strategy
    exp_schedule = GuidedExploration(env, config.eps_begin, config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end, config.lr_nsteps)

    # train model
    model = NatureQN(env, config)
    model.restore(config.model_output)
    model.run(exp_schedule, lr_schedule)
