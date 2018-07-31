import kfac
import tensorflow as tf


class ColdStartPeriodicInvUpdateKfacOpt(kfac.KfacOptimizer):
    """A modified `kfac.KfacOptimizer`, based on the `kfac.PeriodicInvCovUpdateKfacOpt`, that runs the inverse operation
    periodically and uses a standard SGD optimizer for a few updates in the beginning, called 'cold updates' using a
    'cold optimizer'.
    This can be used to slowly initialize the parameters in the beginning before using the heavy K-FAC optimizer.
    The covariances get updated every step (after the 'cold updates').

    The idea is taken from the original ACKTR implementation:

        https://github.com/openai/baselines/blob/master/baselines/acktr/kfac.py
    """

    def __init__(self, num_cold_updates, cold_optimizer, invert_every, **kwargs):
        """Creates a new `ColdStartPeriodicInvUpdateKfacOpt`.

        Args:
            num_cold_updates: The number of 'cold updates' in the beginning before using the actual K-FAC optimizer.
            cold_optimizer: The `tf.train.Optimizer` to use for the 'cold updates'.
            invert_every: The inverse operation get called every `invert_every` steps (after the 'cold updates' have
                finished).
        """
        self._num_cold_updates = num_cold_updates
        self._cold_optimizer = cold_optimizer
        self._invert_every = invert_every
        self._counter = None
        super().__init__(**kwargs)

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        cov_update_thunks, inv_update_thunks = self.make_vars_and_create_op_thunks()

        with tf.control_dependencies([global_step]):
            do_cold_or_cov_updates = tf.cond(tf.less(global_step, self._num_cold_updates),
                                             lambda: self._cold_optimizer.apply_gradients(grads_and_vars, global_step),
                                             lambda: tf.group([thunk() for thunk in cov_update_thunks]))

        with tf.control_dependencies([do_cold_or_cov_updates]):
            do_inv_updates = tf.cond(tf.logical_and(tf.greater(global_step, self._num_cold_updates),
                                                    tf.equal(tf.mod(global_step - self._num_cold_updates,
                                                                    self._invert_every), 0)),
                                     lambda: tf.group([thunk() for thunk in inv_update_thunks]), tf.no_op)

            with tf.control_dependencies([do_inv_updates]):
                return super().apply_gradients(grads_and_vars=grads_and_vars, global_step=global_step, name=name)
