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
    # creates functions to create environments (binds values to make_atari_env)
    # render first environment to visualize the learning progress
    env_fns = [functools.partial(make_atari_env, env_id, render=i == 0) for i in range(num_envs)]
    envs = create_subprocess_envs(env_fns)

    # stacking frames inside the subprocesses would cause the frames to be passed between processes multiple times
    envs = [wrappers.FrameStackWrapper(env, 4) for env in envs]
    multi_env = MultiEnv(envs)

    # acktr uses only 32 filters in the last layer
    model = AtariModel(multi_env.observation_space, multi_env.action_space, 32 if acktr else 64)

    objective = A2CObjective(model, discount_factor=0.99, entropy_regularization_strength=0.01)

    if acktr:
        # required for the K-FAC optimizer
        layer_collection = kfac.LayerCollection()
        model.register_layers(layer_collection)
        model.register_predictive_distributions(layer_collection)

        # use SGD optimizer for the first few iterations, to prevent NaN values # TODO
        cold_optimizer = tf.train.MomentumOptimizer(learning_rate=0.0007, momentum=0.9)
        cold_optimizer = ClipGlobalNormOptimizer(cold_optimizer, clip_norm=0.5)

        optimizer = ColdStartPeriodicInvUpdateKfacOpt(
            cold_updates=20, cold_optimizer=cold_optimizer,
            invert_every=10, learning_rate=0.25, cov_ema_decay=0.99, damping=0.01,
            layer_collection=layer_collection, momentum=0.9, norm_constraint=0.0001,  # trust region radius
            cov_devices=['/gpu:0'], inv_devices=['/gpu:0'])

    else:
        optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0007)
        optimizer = ClipGlobalNormOptimizer(optimizer, clip_norm=0.5)  # clip the gradients

    global_step = tf.train.get_or_create_global_step()

    # create optimizer operation for shared parameters
    optimize_op = objective.minimize_shared(optimizer, baseline_loss_weight=0.5, global_step=global_step)

    agent = MultiEnvAgent(multi_env, model, num_steps)

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
                # sample trajectory batch
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
            # save when interrupted
            if step is not None:
                #saver.save(session, save_path + '/' + model_name, step)
                print('Saved model (step {})'.format(step))


def make_atari_env(env_id, render):
    env = gym.make(env_id)

    env = wrappers.AtariNoopResetWrapper(env, noop_max=30)  # sends the 'NOOP' action 30 times after a reset

    # use only 4th frame while repeating the action on the remaining 3 frames
    env = wrappers.AtariFrameskipWrapper(env, frameskip=4)

    env = wrappers.AtariPreprocessFrameWrapper(env)  # extract luminance and scale down
    env = wrappers.EpisodeInfoWrapper(env)  # stores episode info in 'info'
    env = wrappers.AtariEpisodicLifeWrapper(env)  # terminate episodes after a life has been lost inside the game

    # sends the 'FIRE' action after a reset (at start and after a life has been lost)
    # this is required for most games to start
    env = wrappers.AtariFireResetWrapper(env)

    env = wrappers.AtariClipRewardWrapper(env)  # clips the rewards between -1 and 1

    if render:
        env = wrappers.RenderWrapper(env)

    env = wrappers.AtariInfoClearWrapper(env)  # removes redundant info to reduce inter-process data

    return env


if __name__ == '__main__':
    acktr = True  # whether to use the K-FAC optimizer
    env_id = 'SeaquestNoFrameskip-v4'  # id of the gym environment
    num_envs = 32  # number of multiple environments
    num_steps = 20  # number of steps per update

    # save in project root directory
    save_path = os.path.abspath('./model')
    os.makedirs(save_path, exist_ok=True)
    model_name = 'atari'

    train_a2c_acktr(acktr, env_id, num_envs, num_steps, save_path, model_name)
