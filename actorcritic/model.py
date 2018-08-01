"""Contains the base class of actor-critic models."""


from abc import ABCMeta

import gym
import numpy as np
import tensorflow as tf


class ActorCriticModel(object, metaclass=ABCMeta):
    """Represents a model (e.g. a neural net) that provides the functionalities required for actor-critic algorithms.
    Provides a policy, a baseline (that is subtracted from the target values to compute the advantage) and the values
    used for bootstrapping from next observations (ideally the values of the baseline), and the placeholders.
    """

    def __init__(self, observation_space, action_space):
        """
        Args:
            observation_space (:obj:`gym.spaces.Space`):
                A space that determines the shape of the :attr:`observations_placeholder` and the
                :attr:`bootstrap_observations_placeholder`.

            action_space (:obj:`gym.spaces.Space`):
                A space that determines the shape of the :attr:`actions_placeholder`.
        """
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
        """:obj:`tf.Tensor`:
            The placeholder for the sampled observations.
        """
        return self._observations_placeholder

    @property
    def bootstrap_observations_placeholder(self):
        """:obj:`tf.Tensor`:
            The placeholder for the sampled next observations. These are used to compute the :attr:`bootstrap_values`.
        """
        return self._bootstrap_observations_placeholder

    @property
    def actions_placeholder(self):
        """:obj:`tf.Tensor`:
            The placeholder for the sampled actions.
        """
        return self._actions_placeholder

    @property
    def rewards_placeholder(self):
        """:obj:`tf.Tensor`:
            The placeholder for the sampled rewards (scalars).
        """
        return self._rewards_placeholder

    @property
    def terminals_placeholder(self):
        """:obj:`tf.Tensor`:
            The placeholder for the sampled terminals (booleans).
        """
        return self._terminals_placeholder

    @property
    def policy(self):
        """:obj:`~actorcritic.policies.Policy`:
            The policy used by this model.
        """
        return self._policy

    @property
    def baseline(self):
        """:obj:`~actorcritic.baselines.Baseline`:
            The baseline used by this model.
        """
        return self._baseline

    @property
    def bootstrap_values(self):
        """:obj:`tf.Tensor`:
            The bootstrapped values that are computed based on the observations passed to
            the :attr:`bootstrap_observations_placeholder`.
        """
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
        """Registers the layers of this model (neural net) in the specified :obj:`kfac.LayerCollection`
        (required for K-FAC).

        Args:
            layer_collection (:obj:`kfac.LayerCollection`):
                A layer collection used by the :obj:`~kfac.KfacOptimizer`.

        Raises:
            :obj:`NotImplementedError`:
                If this model does not support K-FAC.
        """
        raise NotImplementedError()

    def register_predictive_distributions(self, layer_collection, random_seed=None):
        """Registers the predictive distributions of the policy and the baseline in the specified
        :obj:`kfac.LayerCollection` (required for K-FAC).

        Args:
            layer_collection (:obj:`kfac.LayerCollection`):
                A layer collection used by the :obj:`~kfac.KfacOptimizer`.

            random_seed (:obj:`int`, optional):
                A random seed used for sampling from the predictive distributions.
        """
        self._policy.register_predictive_distribution(layer_collection, random_seed)
        self._baseline.register_predictive_distribution(layer_collection, random_seed)

    def sample_actions(self, observations, session):
        """Samples actions from the policy based on the specified observations.

        Args:
            observations:
                The observations that will be passed to the :attr:`observations_placeholder`.

            session (:obj:`tf.Session`):
                A session that will be used to compute the values.

        Returns:
            :obj:`list` of :obj:`list`:
                A list of lists of actions. The shape equals the shape of `observations`.
        """
        return session.run(self.policy.sample, feed_dict={
            self.observations_placeholder: observations
        }).tolist()

    def select_max_actions(self, observations, session):
        """Selects actions from the policy that have the highest probability (mode) based on the specified observations.

        Args:
            observations:
                The observations that will be passed to the :attr:`observations_placeholder`.

            session (:obj:`tf.Session`):
                A session that will be used to compute the values.

        Returns:
            :obj:`list` of :obj:`list`:
                A list of lists of actions. The shape equals the shape of `observations`.
        """
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
