"""An example of how to use `A2C` and `ACKTR` to learn to play an Atari game."""

import functools
import os

import gym
import kfac
import numpy as np
import tensorflow as tf

import actorcritic.envs.atari.wrappers as wrappers
from actorcritic.agents import MultiEnvAgent
from actorcritic.envs.atari.model import AtariModel
from actorcritic.kfac_utils import ColdStartPeriodicInvUpdateKfacOpt
from actorcritic.multi_env import MultiEnv, create_subprocess_envs
from actorcritic.nn import ClipGlobalNormOptimizer, linear_decay
from actorcritic.objectives import A2CObjective


def train_a2c_acktr(acktr, env_id, num_envs, num_steps, checkpoint_path, model_name, summary_path=None):
    """Trains an Atari model using `A2C` or `ACKTR`. Automatically saves and loads the trained model.

    Args:
        acktr (:obj:`bool`):
            Whether the `ACKTR` or the `A2C` algorithm should be used. `A2C` uses the RMSProp optimizer and 64 filters
            in the third convolutional layer of the neural network. `ACKTR` uses the K-FAC optimizer and 32 filters.

        env_id (:obj:`string`):
            An id passed to :meth:`gym.make` to create the environments.

        num_envs (:obj:`int`):
            The number of environments that will be used (so `num_envs` subprocesses will be created).
            `A2C` normally uses 16. `ACKTR` normally uses 32.

        num_steps (:obj:`int`):
            The number of steps to take in each iteration. `A2C` normally uses 5. `ACKTR` normally uses 20.

        checkpoint_path (:obj:`string`):
            A directory where the model's checkpoints will be loaded and saved.

        model_name (:obj:`string`):
            A name of the model. The files in the `checkpoint_path` directory will have this name.

        summary_path (:obj:`string`, optional):
            A directory where the TensorBoard summaries will be saved. If not specified, no summaries will be saved.
    """

    envs = create_environments(env_id, num_envs)
    multi_env = MultiEnv(envs)

    # acktr uses 32 filters in the third convolutional layer
    conv3_num_filters = 32 if acktr else 64
    model = AtariModel(multi_env.observation_space, multi_env.action_space, conv3_num_filters)

    agent = MultiEnvAgent(multi_env, model, num_steps)

    objective = A2CObjective(model, discount_factor=0.99, entropy_regularization_strength=0.01)

    global_step = tf.train.get_or_create_global_step()

    # train for 10,000,000 'time steps', which equals 40,000,000 frames since we stack the last 4 frames
    # since we are using the global_step we have to convert from 'time steps' to 'global steps' by dividing by the batch
    # size for each update, i.e. the number of environments times the number of steps
    max_step = 10000000 / (num_envs * num_steps)

    if acktr:
        # use a linear decaying learning rate from 0.25 to 0.025
        learning_rate = linear_decay(0.25, 0.025, global_step, max_step, name='learning_rate')
    else:
        # use a linear decaying learning rate from 0.0007 to 0.00007
        learning_rate = linear_decay(0.0007, 0.00007, global_step, max_step, name='learning_rate')

    optimizer = create_optimizer(acktr, model, learning_rate)

    # create optimizer operation for shared parameters
    optimize_op = objective.optimize_shared(optimizer, baseline_loss_weight=0.5, global_step=global_step)

    with tf.Session() as session:
        # placeholder for summary only
        episode_reward_placeholder = tf.placeholder(tf.float32, [])

        # setup summaries if requested by the user
        if summary_path is not None:
            with tf.name_scope('model'):
                tf.summary.scalar('policy_loss', objective.policy_loss)
                tf.summary.scalar('baseline_loss', objective.baseline_loss)
                tf.summary.scalar('policy_entropy', objective.mean_entropy)

            with tf.name_scope('environment'):
                tf.summary.scalar('episode_reward', episode_reward_placeholder)

            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(summary_path, session.graph)
        else:
            summary_op = tf.no_op()
            summary_writer = None

        session.run(tf.global_variables_initializer())

        # load the saved model to continue training
        saver = tf.train.Saver()
        load_model(saver, checkpoint_path, session)

        step = session.run(global_step)

        try:
            while step < max_step:
                # sample batch of trajectories
                observations, actions, rewards, terminals, next_observations, infos = agent.interact(session)

                # get the episode rewards from infos
                episode_rewards = wrappers.EpisodeInfoWrapper.get_episode_rewards_from_info_batch(infos)
                # compute the mean episode reward
                mean_episode_reward = np.nan if np.all(np.isnan(episode_rewards)) else np.nanmean(episode_rewards)

                # update policy and baseline
                summary, step, _ = session.run([summary_op, global_step, optimize_op], feed_dict={
                    model.observations_placeholder: observations,
                    model.bootstrap_observations_placeholder: next_observations,
                    model.actions_placeholder: actions,
                    model.rewards_placeholder: rewards,
                    model.terminals_placeholder: terminals,

                    # to visualize the mean reward in TensorBoard
                    episode_reward_placeholder: mean_episode_reward
                })

                # add summary if requested by the user
                if summary_path is not None:
                    summary_writer.add_summary(summary, step)
                    # write through summaries every 10th step to get summaries faster into TensorBoard
                    if step % 10 == 0:
                        summary_writer.flush()

                # save model every 100th step
                if step % 100 == 0 and step > 0:
                    save_model(saver, checkpoint_path, model_name, step, session)

        except KeyboardInterrupt:
            print('Stop requested')

            # save the model if interrupted
            save_model(saver, checkpoint_path, model_name, step, session)

        finally:
            # end all subprocesses
            multi_env.close()


def create_environments(env_id, num_envs):
    """Creates multiple Atari environments that run in subprocesses.

    Args:
        env_id (:obj:`string`):
            An id passed to :meth:`gym.make` to create the environments.

        num_envs (:obj:`int`):
            The number of environments (and subprocesses) that will be created.

    Returns:
        :obj:`list` of :obj:`gym.Wrapper`:
            The environments.
    """

    # creates functions to create environments (binds values to make_atari_env)
    # render first environment to visualize the learning progress
    env_fns = [functools.partial(make_atari_env, env_id, render=i == 0) for i in range(num_envs)]
    envs = create_subprocess_envs(env_fns)

    # stacking frames inside the subprocesses would cause the frames to be passed between processes multiple times
    envs = [wrappers.FrameStackWrapper(env, 4) for env in envs]
    return envs


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
    env = wrappers.EpisodeInfoWrapper(env)  # stores episode info in 'info' at the end of an episode
    env = wrappers.AtariEpisodicLifeWrapper(env)  # terminate episodes after a life has been lost inside the game

    # execute the 'FIRE' action after a reset (at start and after a life has been lost)
    # this is required for most games to start
    env = wrappers.AtariFireResetWrapper(env)

    env = wrappers.AtariClipRewardWrapper(env)  # clips the rewards between -1 and 1

    if render:
        env = wrappers.RenderWrapper(env)

    env = wrappers.AtariInfoClearWrapper(env)  # removes redundant info to reduce inter-process data

    return env


def create_optimizer(acktr, model, learning_rate):
    """Creates an optimizer based on whether `ACKTR` or `A2C` is used. `A2C` uses the RMSProp optimizer, `ACKTR` uses
    the K-FAC optimizer. This function is not restricted to Atari models and can be used generally.

    Args:
        acktr (:obj:`bool`):
            Whether to use the optimizer of `ACKTR` or `A2C`.

        model (:obj:`~actorcritic.model.ActorCriticModel`):
            A model that is needed for K-FAC to register the neural network layers and the predictive distributions.

        learning_rate (:obj:`float` or :obj:`tf.Tensor`):
            A learning rate for the optimizer.
    """

    if acktr:
        # required for the K-FAC optimizer
        layer_collection = kfac.LayerCollection()
        model.register_layers(layer_collection)
        model.register_predictive_distributions(layer_collection)

        # use SGD optimizer for the first few iterations, to prevent NaN values  # TODO
        cold_optimizer = tf.train.MomentumOptimizer(learning_rate=0.0003, momentum=0.9)
        cold_optimizer = ClipGlobalNormOptimizer(cold_optimizer, clip_norm=0.5)

        optimizer = ColdStartPeriodicInvUpdateKfacOpt(
            num_cold_updates=30, cold_optimizer=cold_optimizer,
            invert_every=10, learning_rate=learning_rate, cov_ema_decay=0.99, damping=0.01,
            layer_collection=layer_collection, momentum=0.9, norm_constraint=0.0001,  # trust region radius
            cov_devices=['/gpu:0'], inv_devices=['/gpu:0'])

    else:
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        optimizer = ClipGlobalNormOptimizer(optimizer, clip_norm=0.5)  # clip the gradients

    return optimizer


def load_model(saver, checkpoint_path, session):
    """Loads the latest model checkpoint (with the neural network parameters) from a directory.

    Args:
        saver (:obj:`tf.train.Saver`):
            A saver object to restore the model.

        checkpoint_path (:obj:`string`):
            A directory where the checkpoint is loaded from.

        session (:obj:`tf.Session`):
            A session which will contain the loaded variable values.
    """

    try:
        latest_checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
        if latest_checkpoint_path is None:
            raise FileNotFoundError()

        saver.restore(session, latest_checkpoint_path)
        print('Loaded model')

    except (tf.errors.NotFoundError, FileNotFoundError):
        print('No model loaded')


def save_model(saver, checkpoint_path, model_name, step, session):
    """Saves a model checkpoint to a directory.

    Args:
        saver (:obj:`tf.train.Saver`):
            A saver object to save the model.

        checkpoint_path (:obj:`string`):
            A directory where the model checkpoint will be saved.

        model_name (:obj:`string`):
            A name of the model. The checkpoint file in the `checkpoint_path` directory will have this name.

        step (:obj:`int` or :obj:`tf.Tensor`):
            A number that is appended to the checkpoint file name.

        session (:obj:`tf.Session`):
            A session whose variables will be saved.
    """

    saver.save(session, checkpoint_path + '/' + model_name, step)
    print('Saved model (step {})'.format(step))


if __name__ == '__main__':
    acktr = True  # whether to use ACKTR or A2C
    env_id = 'SeaquestNoFrameskip-v4'  # id of the gym environment
    num_envs = 32  # number of multiple environments
    num_steps = 20  # number of steps per update

    # save results in current directory
    results_path = os.path.abspath('./results')
    checkpoint_path = results_path + '/checkpoints/' + env_id
    summary_path = results_path + '/summaries/' + env_id

    # make sure the directories exist
    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(summary_path, exist_ok=True)

    model_name = 'Atari-' + env_id

    train_a2c_acktr(acktr, env_id, num_envs, num_steps, checkpoint_path, model_name, summary_path)

    # If you encounter an InvalidArgumentError 'Received a label value of x which is outside the valid range of [0, x)',
    # just restart the program until it works. This should only happen at the beginning of the learning process. This is
    # not intended and hopefully will be fixed in the future.
