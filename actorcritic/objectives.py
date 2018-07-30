import numpy as np
import tensorflow as tf

from abc import ABCMeta, abstractmethod


class ActorCriticObjective(object, metaclass=ABCMeta):

    @property
    @abstractmethod
    def policy_loss(self):
        pass

    @property
    @abstractmethod
    def baseline_loss(self):
        pass

    def minimize_separate(self, policy_optimizer, baseline_optimizer, global_step=None):
        policy_op = policy_optimizer.minimize(self.policy_loss, global_step=global_step)
        baseline_op = baseline_optimizer.minimize(self.baseline_loss, global_step=global_step)
        return tf.group([policy_op, baseline_op])

    def minimize_shared(self, optimizer, baseline_loss_weight, global_step=None):
        shared_loss = self.policy_loss + baseline_loss_weight * self.baseline_loss
        return optimizer.minimize(shared_loss, global_step=global_step)


class A2CObjective(ActorCriticObjective):

    def __init__(self, model, discount_factor, entropy_regularization_strength=0.01, name=None):
        bootstrap_values = model.bootstrap_values
        actions = model.actions_placeholder
        rewards = model.rewards_placeholder
        terminals = model.terminals_placeholder

        policy = model.policy
        baseline = model.baseline

        with tf.name_scope(name, 'A2CObjective'):
            with tf.name_scope('discounted_rewards'):
                discounted_rewards = _discount(rewards, terminals, discount_factor)

            with tf.name_scope('discounted_bootstrap_values'):
                discounted_bootstrap_values = _discount_bootstrap(bootstrap_values, terminals, discount_factor)

            with tf.name_scope('target_values'):
                target_values = tf.stop_gradient(discounted_rewards + discounted_bootstrap_values)

            with tf.name_scope('advantage_function'):
                advantage_function = tf.stop_gradient(target_values - baseline.value)

            with tf.name_scope('log_prob'):
                log_prob = policy.log_prob(tf.stop_gradient(actions))

            with tf.name_scope('policy_objective'):
                # policy objective function:
                # J(θ) = log(π(s, a | θ)) * advantages
                # TODO reduce_sum axis=1 ?
                policy_objective = tf.reduce_mean(advantage_function * log_prob)

            with tf.name_scope('entropy_regularization'):
                entropy_regularization = entropy_regularization_strength * policy.entropy

            # full policy objective function with entropy regularization:
            # J(θ) = log(π(s, a | θ)) * advantages + beta * entropy(π(θ))
            with tf.name_scope('full_policy_objective'):
                full_policy_objective = policy_objective + entropy_regularization

            with tf.name_scope('policy_loss'):
                # maximize policy objective = minimizing the negative
                self._policy_loss = -full_policy_objective

            with tf.name_scope('baseline_loss'):
                # TODO value_function_loss = -tf.reduce_mean(advantage_function * value_function_gradient) ?
                self._baseline_loss = tf.reduce_mean(tf.square(target_values - baseline.value) / 2.)

    @property
    def policy_loss(self):
        return self._policy_loss

    @property
    def baseline_loss(self):
        return self._baseline_loss


def _discount_matrices(terminals, discount_factor):
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

    return tf.py_func(fn, [terminals, discount_factor], tf.float32, stateful=True)


def _discount_with_matrices(values, matrices):
    values = tf.expand_dims(values, axis=1)
    discounted_values = tf.matmul(values, matrices)
    return tf.squeeze(discounted_values, axis=1)


def _discount(values, terminals, discount_factor):
    matrices = _discount_matrices(terminals, discount_factor)
    return _discount_with_matrices(values, matrices)


def _discount_bootstrap(values, terminals, discount_factor):
    def fn(terminals):
        num_trajectories, num_steps = terminals.shape
        discount_factors = np.zeros_like(terminals, dtype=np.float32)

        # TODO improve
        for i in range(num_trajectories):
            last_terminal = -1
            for t in reversed(range(num_steps)):
                if terminals[i, t]:
                    last_terminal = t
                    break

            for t in range(last_terminal + 1, num_steps):
                discount_factors[i, t] = np.power(discount_factor, num_steps - t)

        return discount_factors

    matrix = tf.py_func(fn, [terminals], tf.float32, stateful=True)
    return matrix * tf.expand_dims(values, axis=-1)
