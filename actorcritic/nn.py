"""Contains utilities that concern TensorFlow and neural networks."""


import numpy as np
import tensorflow as tf


def fully_connected_params(input_size, output_size, dtype, weights_initializer, bias_initializer):
    """Creates weights and bias variables for a fully connected layer. These can be used in :meth:`fully_connected`.

    Args:
        input_size (:obj:`int`):
            The size of the input layer.

        output_size (:obj:`int`):
            The output size. Number of units.

        dtype (:obj:`tf.DType`):
            The data type of the variables.

        weights_initializer (:obj:`tf.keras.initializers.Initializer`):
            An initializer for the weights.

        bias_initializer (:obj:`tf.keras.initializers.Initializer`):
            An initializer for the bias.

    Returns:
        :obj:`tuple` of (:obj:`tf.Variable`, :obj:`tf.Variable`):
            A tuple of (`weights`, `bias`).
    """
    weights = tf.get_variable('weights', (input_size, output_size), dtype, weights_initializer)
    bias = tf.get_variable('bias', (output_size,), dtype, bias_initializer)
    return weights, bias


# noinspection PyShadowingBuiltins
def fully_connected(input, params):
    """Creates a fully connected layer with bias (without activation).

    Args:
        input (:obj:`tf.Tensor`):
            The input values.

        params (:obj:`tuple` of (:obj:`tf.Variable`, :obj:`tf.Variable`)):
            A tuple of (`weights`, `bias`). Probably obtained by :meth:`fully_connected_params`.

    Returns:
        :obj:`tf.Tensor`:
            The output values.
    """
    weights, bias = params
    return input @ weights + bias


def conv2d_params(num_input_channels, num_filters, filter_extent, dtype, weights_initializer, bias_initializer):
    """Creates weights and bias variables for a 2D convolutional layer. These can be used in :meth:`conv2d`.

    Args:
        num_input_channels (:obj:`int`):
            The size of the input layer.

        num_filters (:obj:`int`):
            The output size. Number of filters to apply.

        filter_extent (:obj:`int`):
            The spatial extent of the filters. Determines the size of the weights.

        dtype (:obj:`tf.DType`):
            The data type of the variables.

        weights_initializer (:obj:`tf.keras.initializers.Initializer`):
            An initializer for the weights.

        bias_initializer (:obj:`tf.keras.initializers.Initializer`):
            An initializer for the bias.

    Returns:
        :obj:`tuple` of (:obj:`tf.Variable`, :obj:`tf.Variable`):
            A tuple of (`weights`, `bias`).
    """
    weights = tf.get_variable(
        'weights', (filter_extent, filter_extent, num_input_channels, num_filters), dtype, weights_initializer)
    bias = tf.get_variable('bias', (num_filters,), dtype, bias_initializer)
    return weights, bias


# noinspection PyShadowingBuiltins
def conv2d(input, params, stride, padding):
    """Creates a 2D convolutional layer with bias (without activation).

    Args:
        input (:obj:`tf.Tensor`):
            The input values.

        params (:obj:`tuple` of (:obj:`tf.Variable`, :obj:`tf.Variable`)):
            A tuple of (`weights`, `bias`). Probably obtained by :meth:`conv2d_params`.

        stride (:obj:`int`):
            The stride of the convolution.

        padding (:obj:`string`):
            The padding of the convolution. One of `'VALID'`, `'SAME'`.

    Returns:
        :obj:`tf.Tensor`:
            The output values.
    """
    strides = (1, stride, stride, 1)
    weights, bias = params
    return tf.nn.conv2d(input, weights, strides, padding, data_format='NHWC') + bias


# noinspection PyShadowingBuiltins
def flatten(input):
    """Flattens inputs but keeps the batch size.

    Args:
        input (:obj:`tf.Tensor`):
            Input values of shape [`batch_size`, `d_1`, ..., `d_n`].

    Returns:
        :obj:`tf.Tensor`:
            Flattened input values of shape [`batch_size`, `d1` * ... * `d_n`].
    """
    flat_size = np.prod(input.get_shape().as_list()[1:])
    return tf.reshape(input, [-1, flat_size])


def linear_decay(start_value, end_value, step, total_steps, name=None):
    """Applies linear decay from `start_value` to `end_value`. The value at a specific step is computed as::

        value = (start_value - end_value) * (1 - step / total_steps) + end_value

    Args:
        start_value (:obj:`tf.Tensor` or :obj:`float`):
            The start value.

        end_value (:obj:`tf.Tensor` or :obj:`float`):
            The end value.

        step (:obj:`tf.Tensor`):
            The current step (e.g. global_step).

        total_step (:obj:`int` or :obj:`tf.Tensor`):
            The total number of steps. Steps to reach end_value.

        name (:obj:`string`, optional):
            A name for the operation.

    Returns:
        :obj:`tf.Tensor`:
            The linear decayed value.
    """
    return tf.train.polynomial_decay(
        learning_rate=start_value, global_step=step, decay_steps=total_steps, end_learning_rate=end_value,
        power=1., cycle=False, name=name)


class ClipGlobalNormOptimizer(tf.train.Optimizer):
    """A :obj:`tf.train.Optimizer` that wraps around another optimizer and minimizes the loss by clipping gradients
    using the global norm (:meth:`tf.clip_by_global_norm`).

    See Also:

        * https://www.tensorflow.org/versions/r1.2/api_docs/python/tf/clip_by_global_norm
        * https://stackoverflow.com/questions/36498127/how-to-apply-gradient-clipping-in-tensorflow/43486487#43486487
    """

    def __init__(self, optimizer, clip_norm, name=None):
        """
        Args:
            optimizer (:obj:`tf.train.Optimizer`):
                An optimizer whose gradients will be clipped.

            clip_norm (:obj:`tf.Tensor` or :obj:`float`):
                Value for the global norm (passed to :meth:`tf.clip_by_global_norm`).

            name (:obj:`string`, optional):
                A name for this optimizer.
        """
        super().__init__(use_locking=False, name='ClipGlobalNormOptimizer' if name is None else name)
        self._optimizer = optimizer
        self._clip_norm = clip_norm

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        gradients, variables = zip(*grads_and_vars)
        gradients, _ = tf.clip_by_global_norm(gradients, self._clip_norm)
        optimize_op = self._optimizer.apply_gradients(zip(gradients, variables), global_step=global_step, name=name)
        return optimize_op
