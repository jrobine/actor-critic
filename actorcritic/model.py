from abc import ABCMeta

import gym
import numpy as np
import tensorflow as tf


class ActorCriticModel(object, metaclass=ABCMeta):

    def __init__(self, observation_space, action_space):
        self._observations_placeholder = None
        self._bootstrap_observations_placeholder = None
        self._actions_placeholder = None
        self._rewards_placeholder = None
        self._terminals_placeholder = None

        self._setup_placeholders(observation_space, action_space)

        self._policy = None
        self._baseline = None
        self._bootstrap_values = None

    @property
    def observations_placeholder(self):
        return self._observations_placeholder

    @property
    def bootstrap_observations_placeholder(self):
        return self._bootstrap_observations_placeholder

    @property
    def actions_placeholder(self):
        return self._actions_placeholder

    @property
    def rewards_placeholder(self):
        return self._rewards_placeholder

    @property
    def terminals_placeholder(self):
        return self._terminals_placeholder

    @property
    def policy(self):
        return self._policy

    @property
    def baseline(self):
        return self._baseline

    @property
    def bootstrap_values(self):
        return self._bootstrap_values

    def _setup_placeholders(self, observation_space, action_space):
        with tf.name_scope('placeholders'):
            self._observations_placeholder = _space_placeholder(observation_space, batch_shape=[None, None],
                                                                name='observations')
            self._bootstrap_observations_placeholder = _space_placeholder(observation_space, batch_shape=[None],
                                                                          name='bootstrap_observations')
            self._actions_placeholder = _space_placeholder(action_space, batch_shape=[None, None], name='actions')
            self._rewards_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, None], name='rewards')
            self._terminals_placeholder = tf.placeholder(dtype=tf.bool, shape=[None, None], name='terminals')

    def register_layers(self, layer_collection):
        raise NotImplementedError()

    def register_losses(self, layer_collection, random_seed=None):
        self._policy.register_loss(layer_collection, random_seed)
        self._baseline.register_loss(layer_collection, random_seed)

    def sample_actions(self, observations, session):
        return session.run(self.policy.sample, feed_dict={
            self.observations_placeholder: observations
        }).tolist()

    def select_max_actions(self, observations, session):
        return session.run(self.policy.mode, feed_dict={
            self.observations_placeholder: observations
        }).tolist()


def _space_placeholder(space, batch_shape=None, name=None):
    if batch_shape is None:
        batch_shape = [None]

    if isinstance(space, gym.spaces.Discrete):
        min_dtype = tf.as_dtype(np.min_scalar_type(space.n))
        return tf.placeholder(min_dtype, shape=batch_shape, name=name)
    elif isinstance(space, gym.spaces.Box):
        if space.low.dtype != space.high.dtype or space.low.shape != space.high.shape:
            raise TypeError()

        return tf.placeholder(space.low.dtype, shape=batch_shape + list(space.low.shape), name=name)

    # TODO support more spaces
    raise TypeError('Unsupported space')
