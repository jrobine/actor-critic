from abc import ABCMeta, abstractmethod


class Baseline(object, metaclass=ABCMeta):
    """A wrapper class for the baseline that is subtracted from the target values (advantage).
    """

    @abstractmethod
    def value(self):
        """The output values of this baseline.

        Returns:
            A `tf.Tensor` that contains the values.
        """
        pass

    def register_predictive_distribution(self, layer_collection, random_seed=None):
        """Registers the predictive distribution of this baseline in the specified `kfac.LayerCollection`
        (required for K-FAC).

        Baselines that do not support K-FAC do not have to override this method.
        In this case a `NotImplementedError` is raised.

        Args:
            layer_collection: A `kfac.LayerCollection`.
            random_seed: An optional random seed for sampling from the predictive distribution.
        """
        raise NotImplementedError()


class StateValueFunction(Baseline):
    """A baseline defined by a state-value function.
    """

    def __init__(self, value):
        """Creates a new `StateValueFunction`.

        Args:
             value: A `tf.Tensor` that contains the output values of this state-value function.
        """
        self._value = value

    @property
    def value(self):
        """The output values of this state-value function.

        Returns:
            A `tf.Tensor` that contains the values.
        """
        return self._value

    def register_predictive_distribution(self, layer_collection, random_seed=None):
        """Registers the predictive distribution (normal distribution) of this state-value function in the specified
        `kfac.LayerCollection` (required for K-FAC).

        Args:
            layer_collection: A `kfac.LayerCollection`.
            random_seed: An optional random seed for sampling from the predictive distribution.
        """
        layer_collection.register_normal_predictive_distribution(mean=self._value, var=1.0, seed=random_seed)
        # var=0.5 => squared error loss, var=1.0 => half squared error loss
        # var=1.0 => vanilla Gauss Newton, see ACKTR 3.1 "Natural gradient in actor-critic"
        # NormalMeanVarianceNegativeLogProbLoss for adaptive gauss newton?
