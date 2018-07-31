import numpy as np
import tensorflow as tf


def fully_connected_params(input_size, output_size, dtype, weights_initializer, bias_initializer):
    """Creates weights and bias variables for a fully connected layer. These can be used in `fully_connected()`
    afterwards.

    Args:
        input_size: The size of the input layer.
        output_size: The output size. Number of units.
        dtype: The data type of the variables.
        weights_initializer: Initializer for weights variable.
        bias_initializer: Initializer for bias variable.

    Returns:
        A tuple of (weights, bias), where `weights` and `bias` are `tf.Variable`s.
    """
    weights = tf.get_variable('weights', (input_size, output_size), dtype, weights_initializer)
    bias = tf.get_variable('bias', (output_size,), dtype, bias_initializer)
    return weights, bias


# noinspection PyShadowingBuiltins
def fully_connected(input, params):
    """Creates a fully connected layer with bias (without activation).

    Args:
        input: A `tf.Tensor` that contains the input values.
        params: Tuple of (weights, bias), where `weights` and `bias` are `tf.Variable`s.

    Returns:
        A `tf.Tensor` that contains the output of the operation.
    """
    weights, bias = params
    return input @ weights + bias


def conv2d_params(num_input_channels, num_filters, filter_extent, dtype, weights_initializer, bias_initializer):
    """Created weights and bias variables for a 2D convolutional layer. These can be used in `conv2d()` afterwards.

    Args:
        num_input_channels: The size of the input layer.
        num_filters: The output size. Number of filters to apply.
        filter_extent: The spatial extent of the filters. Determines the size of the weights.
        dtype: The data type of the variables.
        weights_initializer: Initializer for weights variable.
        bias_initializer: Initializer for bias variable.

    Returns:
        A tuple of (weights, bias), where `weights` and `bias` are `tf.Variable`s.
    """
    weights = tf.get_variable(
        'weights', (filter_extent, filter_extent, num_input_channels, num_filters), dtype, weights_initializer)
    bias = tf.get_variable('bias', (num_filters,), dtype, bias_initializer)
    return weights, bias


# noinspection PyShadowingBuiltins
def conv2d(input, params, stride, padding):
    """Creates a 2D convolutional layer with bias (without activation).

    Args:
        input: A `tf.Tensor` that contains the input values.
        params: Tuple of (weights, bias), where `weights` and `bias` are `tf.Variable`s.
        stride: The stride.
        padding: The padding. One of 'VALID', 'SAME'.

    Returns:
        A `tf.Tensor` that contains the output of the operation.
    """
    strides = (1, stride, stride, 1)
    weights, bias = params
    return tf.nn.conv2d(input, weights, strides, padding, data_format='NHWC') + bias


# noinspection PyShadowingBuiltins
def flatten(input):
    """Flattens a tensor but keeps the batch size.

    Args:
        input: A `tf.Tensor` of shape [`batch_size`, `d_1`, ..., `d_n`].

    Returns:
        A `tf.Tensor` of shape [`batch_size`, `d1` * ... * `d_n`] that contains the flattenend input.
    """
    flat_size = np.prod(input.get_shape().as_list()[1:])
    return tf.reshape(input, [-1, flat_size])


def linear_decay(start_value, end_value, step, total_steps, name=None):
    """Applies linear decay from start_value to end_value.

    ```python
    value = (start_value - end_value) * (1 - step / total_steps) + end_value
    ```

    Args:
        start_value: A `tf.Tensor` or scalar that contains the start value.
        end_value: A `tf.Tensor` or scalar that contains the end value.
        step: A `tf.Tensor` that contains the current step (e.g. global_step).
        total_step: The total number of steps. Steps to reach end_value.
        name: An optional name of the operation.

    Returns:
        A `tf.Tensor` that contains the linear decayed value.
    """
    return tf.train.polynomial_decay(
        learning_rate=start_value, global_step=step, decay_steps=total_steps, end_learning_rate=end_value,
        power=1., cycle=False, name=name)


class ClipGlobalNormOptimizer(tf.train.Optimizer):
    """An optimizer that minimizes the loss by clipping gradients using global norm (tf.clip_by_global_norm).

        see also:
        * https://www.tensorflow.org/versions/r1.2/api_docs/python/tf/clip_by_global_norm
        * https://stackoverflow.com/questions/36498127/how-to-apply-gradient-clipping-in-tensorflow/43486487#43486487
    """

    def __init__(self, optimizer, clip_norm, name=None):
        """Creates a new `ClipGlobalNormOptimizer`.

        Args:
            optimizer: The original `tf.train.Optimizer` to clip gradients on.
            clip_norm: A `tf.Tensor` or scalar that is used as value for global norm (tf.clip_by_global_norm).
            name: An optional name for this optimizer.
        """
        super().__init__(use_locking=False, name='ClipGlobalNormOptimizer' if name is None else name)
        self._optimizer = optimizer
        self._clip_norm = clip_norm

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        gradients, variables = zip(*grads_and_vars)
        gradients, _ = tf.clip_by_global_norm(gradients, self._clip_norm)
        optimize_op = self._optimizer.apply_gradients(zip(gradients, variables), global_step=global_step, name=name)
        return optimize_op
