import kfac
import tensorflow as tf


# TODO

# modified version of PeriodicInvCovUpdateKfacOpt from tensorflow kfac implementation
class ColdStartPeriodicInvUpdateKfacOpt(kfac.KfacOptimizer):

    def __init__(self, cold_updates, cold_optimizer, invert_every, **kwargs):
        self._cold_updates = cold_updates
        self._cold_optimizer = cold_optimizer
        self._invert_every = invert_every
        self._counter = None
        super().__init__(**kwargs)

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        cov_update_thunks, inv_update_thunks = self.make_vars_and_create_op_thunks()

        with tf.control_dependencies([global_step]):
            do_cold_or_cov_updates = tf.cond(tf.less(global_step, self._cold_updates),
                                             lambda: self._cold_optimizer.apply_gradients(grads_and_vars, global_step),
                                             lambda: tf.group([thunk() for thunk in cov_update_thunks]))

        with tf.control_dependencies([do_cold_or_cov_updates]):
            do_inv_updates = tf.cond(tf.logical_and(tf.greater(global_step, self._cold_updates),
                                                    tf.equal(tf.mod(global_step - self._cold_updates,
                                                                    self._invert_every), 0)),
                                     lambda: tf.group([thunk() for thunk in inv_update_thunks]), tf.no_op)

            with tf.control_dependencies([do_inv_updates]):
                return super().apply_gradients(grads_and_vars=grads_and_vars, global_step=global_step, name=name)
