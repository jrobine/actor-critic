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
        counter = self.counter
        prev_counter = tf.assign(tf.get_variable('prev_counter', shape=(), initializer=tf.zeros_initializer), counter)

        with tf.control_dependencies([prev_counter]):
            update_counter = tf.assign_add(counter, 1, name='update_counter')
            do_cold_or_cov_updates = tf.cond(tf.less_equal(prev_counter, self._cold_updates),
                                             lambda: self._cold_optimizer.apply_gradients(grads_and_vars, global_step),
                                             lambda: tf.group([thunk() for thunk in cov_update_thunks]))

        with tf.control_dependencies([do_cold_or_cov_updates, update_counter]):
            do_inv_updates = tf.cond(tf.logical_and(tf.greater(prev_counter, self._cold_updates),
                                                    tf.equal(tf.mod(prev_counter - self._cold_updates,
                                                                    self._invert_every), 0)),
                                     lambda: tf.group([thunk() for thunk in inv_update_thunks]), tf.no_op)

            with tf.control_dependencies([do_inv_updates]):
                return super().apply_gradients(grads_and_vars=grads_and_vars, global_step=global_step, name=name)

    @property
    def counter(self):
        if self._counter is None:
            with tf.variable_scope('periodic_counter', reuse=tf.AUTO_REUSE):
                self._counter = tf.get_variable('counter', shape=(), initializer=tf.zeros_initializer)

        return self._counter
