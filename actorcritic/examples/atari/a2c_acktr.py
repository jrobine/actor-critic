"""An example of how to use A2C and ACKTR to learn to play an Atari game."""

import functools
import os

import gym
import kfac
import tensorflow as tf

import actorcritic.envs.atari.wrappers as wrappers
from actorcritic.agents import MultiEnvAgent
from actorcritic.envs.atari.model import AtariModel
from actorcritic.kfac_utils import ColdStartPeriodicInvUpdateKfacOpt
from actorcritic.multi_env import MultiEnv, create_subprocess_envs
from actorcritic.nn import ClipGlobalNormOptimizer
from actorcritic.objectives import A2CObjective


def train_a2c_acktr(acktr, env_id, num_envs, num_steps, save_path, model_name):
    """Trains an Atari model using A2C or ACKTR. Automatically saves and loads the trained model.

    Args:
        acktr (:obj:`bool`):
            Whether the ACKTR or the A2C algorithm should be used. ACKTR uses the K-FAC optimizer and uses 32 filters in
            the third convolutional layer of the neural network instead of 64.

        env_id (:obj:`string`):
            An id passed to :meth:`gym.make` to create the environments.

        num_envs (:obj:`int`):
            The number of environments that will be used (so `num_envs` subprocesses will be created).
            A2C normally uses 16. ACKTR normally uses 32.

        num_steps (:obj:`int`):
            The number of steps to take in each iteration. A2C normally uses 5. ACKTR normally uses 20.

        save_path (:obj:`string`):
            A directory to load and save the model.

        model_name (:obj:`string`):
            A name of the model. The files in the `save_path` directory will have this name.
    """

    # creates functions to create environments (binds values to make_atari_env)
    # render first environment to visualize the learning progress
    env_fns = [functools.partial(make_atari_env, env_id, render=i == 0) for i in range(num_envs)]
    envs = create_subprocess_envs(env_fns)

    # stacking frames inside the subprocesses would cause the frames to be passed between processes multiple times
    envs = [wrappers.FrameStackWrapper(env, 4) for env in envs]
    multi_env = MultiEnv(envs)

    # acktr uses only 32 filters in the last layer
    model = AtariModel(multi_env.observation_space, multi_env.action_space, 32 if acktr else 64)

    agent = MultiEnvAgent(multi_env, model, num_steps)

    objective = A2CObjective(model, discount_factor=0.99, entropy_regularization_strength=0.01)

    if acktr:
        # required for the K-FAC optimizer
        layer_collection = kfac.LayerCollection()
        model.register_layers(layer_collection)
        model.register_predictive_distributions(layer_collection)

        # use SGD optimizer for the first few iterations, to prevent NaN values  # TODO
        cold_optimizer = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9)
        cold_optimizer = ClipGlobalNormOptimizer(cold_optimizer, clip_norm=0.25)

        optimizer = ColdStartPeriodicInvUpdateKfacOpt(
            num_cold_updates=30, cold_optimizer=cold_optimizer,
            invert_every=10, learning_rate=0.25, cov_ema_decay=0.99, damping=0.01,
            layer_collection=layer_collection, momentum=0.9, norm_constraint=0.0001,  # trust region radius
            cov_devices=['/gpu:0'], inv_devices=['/gpu:0'])

    else:
        optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0007)
        optimizer = ClipGlobalNormOptimizer(optimizer, clip_norm=0.5)  # clip the gradients

    global_step = tf.train.get_or_create_global_step()

    # create optimizer operation for shared parameters
    optimize_op = objective.optimize_shared(optimizer, baseline_loss_weight=0.5, global_step=global_step)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        try:
            latest_checkpoint_path = tf.train.latest_checkpoint(save_path)
            if latest_checkpoint_path is None:
                raise FileNotFoundError()

            saver.restore(session, latest_checkpoint_path)
            print('Loaded model')

        except (tf.errors.NotFoundError, FileNotFoundError):
            print('No model loaded')

        step = None
        try:
            while True:
                # sample batch of trajectories
                observations, actions, rewards, terminals, next_observations, infos = agent.interact(session)

                # update policy and baseline
                step, _ = session.run([global_step, optimize_op], feed_dict={
                    model.observations_placeholder: observations,
                    model.bootstrap_observations_placeholder: next_observations,
                    model.actions_placeholder: actions,
                    model.rewards_placeholder: rewards,
                    model.terminals_placeholder: terminals
                })

                if step % 100 == 0 and step > 0:
                    # save every 100th step
                    saver.save(session, save_path + '/' + model_name, step)
                    print('Saved model (step {})'.format(step))

        except KeyboardInterrupt:
            multi_env.close()

            # save when interrupted
            if step is not None:
                saver.save(session, save_path + '/' + model_name, step)
                print('Saved model (step {})'.format(step))


def make_atari_env(env_id, render):
    """Creates a :obj:`gym.Env` and wraps it with all Atari wrappers in :mod:`actorcritic.envs.atari.wrappers`.

    Args:
        env_id (:obj:`string`):
            An id passed to :meth:`gym.make`.

        render (:obj:`bool`):
            Whether this environment should be rendered.

    Returns:
        :obj:`gym.Env`:
            The environment.
    """
    env = gym.make(env_id)

    # execute the 'NOOP' action a random number of times between 1 and 30 after a reset
    env = wrappers.AtariNoopResetWrapper(env, noop_max=30)

    # use only 4th frame while repeating the action on the remaining 3 frames
    env = wrappers.AtariFrameskipWrapper(env, frameskip=4)

    # preprocess (convert to grayscale and scale down) the observations in the subprocesses to decrease computation time
    # the preprocessing should not be done on the gpu, since the amount of data that will be passed to the gpu will be
    # drastically decreased, which is much less time-consuming
    env = wrappers.AtariPreprocessFrameWrapper(env)
    env = wrappers.EpisodeInfoWrapper(env)  # stores episode info in 'info' at the end of episode
    env = wrappers.AtariEpisodicLifeWrapper(env)  # terminate episodes after a life has been lost inside the game

    # execute the 'FIRE' action after a reset (at start and after a life has been lost)
    # this is required for most games to start
    env = wrappers.AtariFireResetWrapper(env)

    env = wrappers.AtariClipRewardWrapper(env)  # clips the rewards between -1 and 1

    if render:
        env = wrappers.RenderWrapper(env)

    env = wrappers.AtariInfoClearWrapper(env)  # removes redundant info to reduce inter-process data

    return env


if __name__ == '__main__':
    acktr = True  # whether to use ACKTR or A2C
    env_id = 'SeaquestNoFrameskip-v4'  # id of the gym environment
    num_envs = 32  # number of multiple environments
    num_steps = 20  # number of steps per update

    # save in project root directory
    save_path = os.path.abspath('./model')
    os.makedirs(save_path, exist_ok=True)
    model_name = 'atari'

    train_a2c_acktr(acktr, env_id, num_envs, num_steps, save_path, model_name)

    # If you encounter an InvalidArgumentError 'Received a label value of x which is outside the valid range of [0, x)',
    # just restart the program until it works. This should only happen at the beginning of the learning process. This is
    # not intended and hopefully will be fixed in the future.
