from abc import ABCMeta, abstractmethod


class Baseline(object, metaclass=ABCMeta):

    @abstractmethod
    def value(self):
        pass

    def register_loss(self, layer_collection, random_seed=None):
        raise NotImplementedError()


class StateValueFunction(Baseline):

    def __init__(self, value):
        self._value = value

    @property
    def value(self):
        return self._value

    def register_loss(self, layer_collection, random_seed=None):
        # var=0.5 => squared error loss, var=1.0 => half squared error loss
        # var=1.0 => vanilla Gauss Newton, see ACKTR 3.1 "Natural gradient in actor-critic"
        layer_collection.register_normal_predictive_distribution(mean=self._value, var=1.0, seed=random_seed)

        # NormalMeanVarianceNegativeLogProbLoss for adaptive gauss newton?
