"""Contains `baselines`, which are used to compute the `advantage`."""

from abc import ABCMeta, abstractmethod


class Baseline(object, metaclass=ABCMeta):
    """A wrapper class for the baseline that is subtracted from the target values to get the `advantage`.
    """

    @property
    @abstractmethod
    def value(self):
        """:obj:`tf.Tensor`:
            The output values of this baseline.
        """
        pass

    def register_predictive_distribution(self, layer_collection, random_seed=None):
        """Registers the predictive distribution of this baseline in the specified :obj:`kfac.LayerCollection`
        (required for K-FAC).

        Args:
            layer_collection (:obj:`kfac.LayerCollection`):
                A layer collection used by the :obj:`~kfac.KfacOptimizer`.

            random_seed (:obj:`int`, optional):
                A random seed for sampling from the predictive distribution.

        Raises:
            :obj:`NotImplementedError`:
                If this baseline does not support K-FAC.
        """
        raise NotImplementedError()


class StateValueFunction(Baseline):
    """A baseline defined by a state-value function.
    """

    def __init__(self, value):
        """
        Args:
             value (:obj:`tf.Tensor`):
                The output values of this state-value function.
        """
        self._value = value

    @property
    def value(self):
        """:obj:`tf.Tensor`:
            The output values of this state-value function.
        """
        return self._value

    def register_predictive_distribution(self, layer_collection, random_seed=None):
        """Registers the predictive distribution (normal distribution) of this state-value function in the specified
        :obj:`kfac.LayerCollection` (required for K-FAC).

        Args:
            layer_collection (:obj:`kfac.LayerCollection`):
                A layer collection used by the :obj:`~kfac.KfacOptimizer`.

            random_seed (:obj:`int`, optional):
                A random seed for sampling from the predictive distribution.
        """
        layer_collection.register_normal_predictive_distribution(mean=self._value, var=1.0, seed=random_seed)
        # var=0.5 => squared error loss, var=1.0 => half squared error loss
        # var=1.0 => vanilla Gauss Newton, see ACKTR 3.1 "Natural gradient in actor-critic"
        # NormalMeanVarianceNegativeLogProbLoss for adaptive gauss newton?
