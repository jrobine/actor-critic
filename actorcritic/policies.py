"""Contains `policies` that determine the behavior of an `agent`."""


from abc import ABCMeta, abstractmethod

import tensorflow as tf


class Policy(object, metaclass=ABCMeta):
    """Base class for stochastic policies.
    """

    @property
    @abstractmethod
    def sample(self):
        """:obj:`tf.Tensor`:
            Samples actions from this policy based on the inputs that are provided for computing the probabilities. The
            shape equals the shape of the inputs.
        """
        pass

    @property
    @abstractmethod
    def mode(self):
        """:obj:`tf.Tensor`:
            Selects actions from this policy which have the highest probability (mode) based on the inputs that are
            provided for computing the probabilities. The shape equals the shape of the inputs.
        """
        pass

    @property
    @abstractmethod
    def entropy(self):
        """:obj:`tf.Tensor`:
            Computes the entropy of this policy based on the inputs that are provided for computing the probabilities.
            The shape equals the shape of the inputs.
        """
        pass

    @property
    @abstractmethod
    def log_prob(self):
        """:obj:`tf.Tensor`:
            Computes the log-probability of the given actions based on the inputs that are provided for computing the
            probabilities. The shape equals the shape of the actions and the inputs.
        """
        pass

    def register_predictive_distribution(self, layer_collection, random_seed=None):
        """Registers the predictive distribution of this policy in the specified :obj:`kfac.LayerCollection`
        (required for K-FAC).

        Args:
            layer_collection (:obj:`kfac.LayerCollection`):
                A layer collection used by the :obj:`~kfac.KfacOptimizer`.

            random_seed (:obj:`int`, optional):
                A random seed for sampling from the predictive distribution.

        Raises:
            :obj:`NotImplementedError`:
                If this policy does not support K-FAC.
        """
        raise NotImplementedError()


class DistributionPolicy(Policy, metaclass=ABCMeta):
    """Base class for stochastic policies that follow a concrete :obj:`tf.distributions.Distribution`. Implements the
    required methods based on this distribution.
    """

    def __init__(self, distribution, actions, random_seed=None):
        """
        Args:
            distribution (:obj:`tf.distributions.Distribution`):
                The distribution.

            actions (:obj:`tf.Tensor`):
                The input actions used to compute the log-probabilities. Must have the same shape as the inputs.

            random_seed (:obj:`int`, optional):
                A random seed used for sampling.
        """
        self._distribution = distribution

        self._sample = tf.squeeze(distribution.sample(sample_shape=[], seed=random_seed, name='sample'), axis=-1)
        self._mode = tf.squeeze(distribution.mode(name='mode'), axis=-1)
        self._entropy = distribution.entropy(name='entropy')
        self._log_prob = distribution.log_prob(tf.stop_gradient(tf.cast(actions, tf.int32)), name='log_prob')

    @property
    def sample(self):
        """:obj:`tf.Tensor`:
            Samples actions from this policy based on the inputs that are provided for computing the probabilities. The
            shape equals the shape of the inputs.
        """
        return self._sample

    @property
    def mode(self):
        """:obj:`tf.Tensor`:
            Selects actions from this policy which have the highest probability (mode) based on the inputs that are
            provided for computing the probabilities. The shape equals the shape of the inputs.
        """
        return self._mode

    @property
    def entropy(self):
        """:obj:`tf.Tensor`:
            Computes the entropy of this policy based on the inputs that are provided for computing the probabilities.
            The shape equals the shape of the inputs.
        """
        return self._entropy

    @property
    def log_prob(self):
        """:obj:`tf.Tensor`:
            Computes the log-probability of the given actions based on the inputs that are provided for computing the
            probabilities. The shape equals the shape of the actions and the inputs.
        """
        return self._log_prob


class SoftmaxPolicy(DistributionPolicy):
    """A stochastic policy that follows a categorical distribution.
    """

    def __init__(self, logits, actions, random_seed=None, name=None):
        """
        Args:
            logits (:obj:`tf.Tensor`):
                The input logits (or 'scores') used to compute the probabilities.

            actions (:obj:`tf.Tensor`):
                The input actions used to compute the log-probabilities. Must have the same shape as `logits`.

            random_seed (:obj:`int`, optional):
                A random seed used for sampling.

            name (:obj:`string`, optional):
                A name for this policy.
        """
        with tf.name_scope(name, 'SoftmaxPolicy'):
            super().__init__(tf.distributions.Categorical(logits, name='distribution'), actions, random_seed)

    def register_predictive_distribution(self, layer_collection, random_seed=None):
        """Registers the predictive distribution of this policy in the specified :obj:`kfac.LayerCollection`
        (required for K-FAC).

        Args:
            layer_collection (:obj:`kfac.LayerCollection`):
                A layer collection used by the :obj:`~kfac.KfacOptimizer`.

            random_seed (:obj:`int`, optional):
                A random seed for sampling from the predictive distribution.
        """
        return layer_collection.register_categorical_predictive_distribution(
            logits=self._distribution.logits, seed=random_seed)
