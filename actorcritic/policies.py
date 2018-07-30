from abc import ABCMeta, abstractmethod

import tensorflow as tf


class Policy(object, metaclass=ABCMeta):

    @property
    @abstractmethod
    def sample(self):
        pass

    @property
    @abstractmethod
    def mode(self):
        pass

    @property
    @abstractmethod
    def entropy(self):
        pass

    @abstractmethod
    def log_prob(self, actions, name=None):
        pass

    def register_loss(self, layer_collection, random_seed=None):
        raise NotImplementedError()


class DistributionPolicy(Policy, metaclass=ABCMeta):

    def __init__(self, distribution, random_seed=None):
        self._distribution = distribution

        self._sample = tf.squeeze(distribution.sample(sample_shape=[], seed=random_seed, name='sample'), axis=-1)
        self._mode = tf.squeeze(distribution.mode(name='mode'), axis=-1)
        self._entropy = tf.reduce_mean(distribution.entropy(name='entropy'), name='mean_entropy')

    @property
    def sample(self):
        return self._sample

    @property
    def mode(self):
        return self._mode

    @property
    def entropy(self, name='entropy'):
        return self._entropy

    def log_prob(self, actions, name='log_prob'):
        return self._distribution.log_prob(tf.cast(actions, tf.int32), name=name)


class SoftmaxPolicy(DistributionPolicy):

    def __init__(self, logits, random_seed=None, name=None):
        with tf.variable_scope(name, 'SoftmaxPolicy'):
            super().__init__(tf.distributions.Categorical(logits, name='distribution'), random_seed)

    def register_loss(self, layer_collection, random_seed=None):
        return layer_collection.register_categorical_predictive_distribution(logits=self._distribution.logits,
                                                                             seed=random_seed)
