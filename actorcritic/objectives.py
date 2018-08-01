"""Contains `objectives` that are used to optimize actor-critic models."""


from abc import ABCMeta, abstractmethod

import numpy as np
import tensorflow as tf


class ActorCriticObjective(object, metaclass=ABCMeta):
    """An objective takes an :obj:`~actorcritic.model.ActorCriticModel` and determines how it is optimized. It defines
    the loss of the policy and the loss of the baseline, and can create train operations based on these losses.
    """

    @property
    @abstractmethod
    def policy_loss(self):
        """:obj:`tf.Tensor`:
            The loss of the policy of the model.
        """
        pass

    @property
    @abstractmethod
    def baseline_loss(self):
        """:obj:`tf.Tensor`:
            The loss of the baseline of the model.
        """
        pass

    def optimize_separate(self, policy_optimizer, baseline_optimizer, policy_kwargs=None, baseline_kwargs=None):
        """Creates an operation that minimizes the policy loss and the baseline loss separately. This means that it
        minimizes the losses using two different optimizers.

        Args:
            policy_optimizer (:obj:`tf.train.Optimizer`):
                An optimizer that is used for the policy loss.

            baseline_optimizer (:obj:`tf.train.Optimizer`):
                An optimizer that is used for the baseline loss.

            policy_kwargs (:obj:`dict`, optional):
                Keyword arguments passed to the :meth:`minimize` method of the `policy_optimizer`.

            baseline_kwargs (:obj:`dict`, optional):
                Keyword arguments passed to the :meth:`minimize` method of the `baseline_optimizer`.

        Returns:
            :obj:`tf.Operation`:
                An operation that updates both the policy and the baseline.
        """
        policy_op = policy_optimizer.minimize(self.policy_loss, **policy_kwargs)
        baseline_op = baseline_optimizer.minimize(self.baseline_loss, **baseline_kwargs)
        return tf.group([policy_op, baseline_op])

    def optimize_shared(self, optimizer, baseline_loss_weight=0.5, **kwargs):
        """Creates an operation that minimizes both the policy loss and the baseline loss using the same optimizer. This
        is used for models that share parameters between the policy and the baseline. The shared loss is defined as::

            shared_loss = policy_loss + baseline_loss_weight * baseline_loss

        where `baseline_loss_weight` determines the `'learning rate'` relative to the policy loss.

        Args:
            optimizer (:obj:`tf.train.Optimizer`):
                An optimizer that is used for both the policy loss and the baseline loss.

            baseline_loss_weight (:obj:`float` or :obj:`tf.Tensor`):
                Determines the relative `'learning rate'`.

            kwargs (:obj:`dict`, optional):
                Keyword arguments passed to the :meth:`minimize` method of the optimizer.

        Returns:
            :obj:`tf.Operation`:
                An operation that updates both the policy and the baseline.
        """
        shared_loss = self.policy_loss + baseline_loss_weight * self.baseline_loss
        return optimizer.minimize(shared_loss, **kwargs)


class A2CObjective(ActorCriticObjective):
    """An objective that defines the loss of the policy and the baseline according to the A3C and A2C/ACKTR papers.

    The rewards are discounted and the policy loss uses entropy regularization. The baseline is optimized using a
    squared error loss.

    The policy objective uses entropy regularization::

        J(theta) = log(policy(state, action | theta)) * (target_values - baseline) + beta * entropy(policy)

    where `beta` determines the strength of the entropy regularization.

    See Also:

        * https://arxiv.org/pdf/1602.01783.pdf  (A3C)
        * https://arxiv.org/pdf/1708.05144.pdf  (A2C/ACKTR)
    """

    def __init__(self, model, discount_factor=0.99, entropy_regularization_strength=0.01, name=None):
        """
        Args:
            model (:obj:`~actorcritic.model.ActorCriticModel`):
                A model that provides the policy and the baseline that will be optimized.

            discount_factor (:obj:`float`):
                Used for discounting the rewards. Should be between [0, 1].

            entropy_regularization_strength (:obj:`float` or :obj:`tf.Tensor`):
                Determining the strength of the entropy regularization. Corresponds to the `beta` parameter in A3C.

            name (:obj:`string`, optional):
                A name for this objective.
        """
        bootstrap_values = model.bootstrap_values
        rewards = model.rewards_placeholder
        terminals = model.terminals_placeholder

        policy = model.policy
        baseline = model.baseline

        with tf.name_scope(name, 'A2CObjective'):
            with tf.name_scope('target_values'):
                discounted_rewards = _discount(rewards, terminals, discount_factor)
                discounted_bootstrap_values = _discount_bootstrap(bootstrap_values, terminals, discount_factor)
                target_values = tf.stop_gradient(discounted_rewards + discounted_bootstrap_values)

            with tf.name_scope('advantage'):
                # advantage = target_values - baseline
                advantage = tf.stop_gradient(target_values - baseline.value)

            with tf.name_scope('standard_policy_objective'):
                # J(theta) = log(policy(state, action | theta)) * advantage
                # TODO reduce_sum axis=1 ?
                standard_policy_objective = tf.reduce_mean(advantage * policy.log_prob)

            with tf.name_scope('entropy_regularization'):
                with tf.name_scope('mean_entropy'):
                    mean_entropy = tf.reduce_mean(policy.entropy)
                entropy_regularization = entropy_regularization_strength * mean_entropy

            with tf.name_scope('policy_objective'):
                # full policy objective with entropy regularization:
                # J(theta) = log(policy(state, action | theta)) * advantage + beta * entropy(policy)
                policy_objective = standard_policy_objective + entropy_regularization

            with tf.name_scope('policy_loss'):
                # maximize policy objective = minimizing the negative
                self._policy_loss = -policy_objective

            with tf.name_scope('baseline_loss'):
                # squared error loss for baseline (actually half squared error loss)
                # TODO value_function_loss = -tf.reduce_mean(advantage_function * value_function_gradient) ?
                self._baseline_loss = tf.reduce_mean(tf.square(target_values - baseline.value) / 2.)

    @property
    def policy_loss(self):
        """:obj:`tf.Tensor`:
            The loss of the policy of the model.
        """
        return self._policy_loss

    @property
    def baseline_loss(self):
        """:obj:`tf.Tensor`:
            The loss of the baseline of the model.
        """
        return self._baseline_loss


def _discount(values, terminals, discount_factor):

    def fn(terminals, discount_factor):
        batch_size, num_steps = terminals.shape

        discount_factors = np.triu(np.ones((num_steps, num_steps), dtype=np.float32), k=1)
        discount_factors = np.cumsum(discount_factors, axis=1)
        discount_factors = np.transpose(discount_factors)
        discount_factors = discount_factor ** discount_factors
        discount_factors = np.tril(discount_factors, k=0)
        discount_factors = np.expand_dims(discount_factors, axis=0)
        discount_factors = np.repeat(discount_factors, batch_size, axis=0)

        indices = np.where(terminals)
        indices = np.transpose(indices)
        for batch_index, time_index in indices:
            discount_factors[batch_index, time_index + 1:, :time_index + 1] = 0.

        return discount_factors

    discount_matrices = tf.py_func(fn, [terminals, discount_factor], tf.float32, stateful=True)

    values = tf.expand_dims(values, axis=1)
    discounted_values = tf.matmul(values, discount_matrices)
    return tf.squeeze(discounted_values, axis=1)


def _discount_bootstrap(values, terminals, discount_factor):
    # terminal trajectories do not need bootstrapping, so returns discounted bootstrapped values
    # for all non-terminal sub-trajectories

    def fn(terminals):
        # returns exponentiated discount factors for all non-terminal sub-trajectories
        return np.flip(np.cumprod(np.cumprod(np.flip(np.invert(terminals), axis=1), axis=1, dtype=np.int32) * discount_factor, axis=1, dtype=np.float32), axis=1)

    discount_factors = tf.py_func(fn, [terminals], tf.float32, stateful=True)
    return discount_factors * tf.expand_dims(values, axis=-1)  # element-wise multiplication
