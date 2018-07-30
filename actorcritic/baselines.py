from abc import ABCMeta, abstractmethod


class Baseline(object, metaclass=ABCMeta):
    """A wrapper class for the baseline that is subtracted from the target values (advantage function).
    """

    @abstractmethod
    def value(self):
        """
        Returns:
            A tensor containing the output values of this baseline.
        """
        pass

    def register_predictive_distribution(self, layer_collection, random_seed=None):
        """Registers the predictive distribution of this baseline in the specified LayerCollection (required for K-FAC).

        Baselines that do not support K-FAC do not have to override this method.
        In this case raises a NotImplementedError.

        Args:
            layer_collection: A `kfac.LayerCollection`.
            random_seed: An optional random seed for sampling from the predictive distribution.
        """

        raise NotImplementedError()


class StateValueFunction(Baseline):

    def __init__(self, value):
        """Creates a new state-value function baseline with the specified values.

        Args:
             value: A tensor containing the output values of this state-value function
        """
        self._value = value

    @property
    def value(self):
        """
        Returns:
            A tensor containing the output values of this state-value function.
        """
        return self._value

    def register_predictive_distribution(self, layer_collection, random_seed=None):
        """Registers the predictive distribution (normal distribution) of this baseline in the specified LayerCollection
        (required for K-FAC).

        Args:
            layer_collection: A `kfac.LayerCollection`.
            random_seed: An optional random seed for sampling from the predictive distribution.
        """

        # var=0.5 => squared error loss, var=1.0 => half squared error loss
        # var=1.0 => vanilla Gauss Newton, see ACKTR 3.1 "Natural gradient in actor-critic"
        layer_collection.register_normal_predictive_distribution(mean=self._value, var=1.0, seed=random_seed)

        # NormalMeanVarianceNegativeLogProbLoss for adaptive gauss newton?
