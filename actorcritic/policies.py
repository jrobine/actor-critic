from abc import ABCMeta, abstractmethod

import tensorflow as tf


class Policy(object, metaclass=ABCMeta):
    """Abstract class for stochastic policies.
    """

    @property
    @abstractmethod
    def sample(self):
        """Samples actions from this policy based on the inputs that are provided for computing the probabilities.

        Returns:
            A tensor that samples the actions. The shape equals the shape of the inputs.
        """
        pass

    @property
    @abstractmethod
    def mode(self):
        """Selects actions from this policy which have the highest probability (mode) based on the inputs that are
        provided for computing the probabilities.

        Returns:
            A tensor that selects the actions. The shape equals the shape of the inputs.
        """
        pass

    @property
    @abstractmethod
    def entropy(self):
        """Computes the entropy of this policy based on the inputs that are provided for computing the probabilities.

        Returns:
            A tensor computing the entropy values. The shape equals the shape of the inputs.
        """
        pass

    @abstractmethod
    def log_prob(self, actions, name=None):
        """Computes the log-probability of the given actions based on the inputs that are provided for computing the
        probabilities.

        Args:
            actions: The actions. Must be of the same shape as the provided inputs.
            name: Optional name of the operation.

        Returns:
            A tensor containing the log-probabilities. The shape equals the shape of the actions and the inputs.
        """
        pass

    def register_predictive_distribution(self, layer_collection, random_seed=None):
        """Registers the predictive distribution of this policy in the specified LayerCollection (required for K-FAC).

        Policies that do not support K-FAC do not have to override this method.
        In this case raises a NotImplementedError.

        Args:
            layer_collection: A `kfac.LayerCollection`.
            random_seed: An optional random seed for sampling from the predictive distribution.
        """

        raise NotImplementedError()


class DistributionPolicy(Policy, metaclass=ABCMeta):
    """Abstract class for stochastic policies that follow a specific `tf.distributions.Distribution`.
    Implements the required methods based on this distribution.
    """

    def __init__(self, distribution, random_seed=None):
        """Creates a policy following the specified distribution.

        Args:
             distribution: The distribution, subclass of `tf.distributions.Distribution`.
             random_seed: An optional random seed used for sampling.
        """

        self._distribution = distribution

        self._sample = tf.squeeze(distribution.sample(sample_shape=[], seed=random_seed, name='sample'), axis=-1)
        self._mode = tf.squeeze(distribution.mode(name='mode'), axis=-1)
        self._entropy = distribution.entropy(name='entropy')

    @property
    def sample(self):
        """Samples actions from this policy based on the inputs that are provided for computing the probabilities.

        Returns:
            A tensor that samples the actions. The shape equals the shape of the inputs.
        """
        return self._sample

    @property
    def mode(self):
        """Selects actions from this policy which have the highest probability (mode) based on the inputs that are
        provided for computing the probabilities.

        Returns:
            A tensor that selects the actions. The shape equals the shape of the inputs.
        """
        return self._mode

    @property
    def entropy(self, name='entropy'):
        """Computes the entropy of this policy based on the inputs that are provided for computing the probabilities.

        Returns:
            A tensor computing the entropy values. The shape equals the shape of the inputs.
        """
        return self._entropy

    def log_prob(self, actions, name='log_prob'):
        """Computes the log-probability of the given actions based on the inputs that are provided for computing the
        probabilities.

        Args:
            actions: The actions. Must be of the same shape as the provided inputs.
            name: Optional name of the operation.

        Returns:
            A tensor containing the log-probabilities. The shape equals the shape of the actions and the inputs.
        """
        return self._distribution.log_prob(tf.cast(actions, tf.int32), name=name)


class SoftmaxPolicy(DistributionPolicy):

    def __init__(self, logits, random_seed=None, name=None):
        """Creates a new policy following a categorical distribution.

        Args:
             logits: The input logits (or 'scores') used to compute the probabilities.
        """

        with tf.variable_scope(name, 'SoftmaxPolicy'):
            super().__init__(tf.distributions.Categorical(logits, name='distribution'), random_seed)

    def register_predictive_distribution(self, layer_collection, random_seed=None):
        """Registers the predictive distribution (categorical distribution) of this policy in the specified
        LayerCollection (required for K-FAC).

        Args:
            layer_collection: A `kfac.LayerCollection`.
            random_seed: An optional random seed for sampling from the predictive distribution.
        """

        return layer_collection.register_categorical_predictive_distribution(
            logits=self._distribution.logits, seed=random_seed)
