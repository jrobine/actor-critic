import gym
import numpy as np
import tensorflow as tf

import actorcritic.nn as nn
from actorcritic.baselines import StateValueFunction
from actorcritic.model import ActorCriticModel
from actorcritic.policies import SoftmaxPolicy


class AtariModel(ActorCriticModel):
    """An `actorcritic.model.ActorCriticModel` that follows the A3C and ACKTR paper. The network is originally used in:

        https://www.nature.com/articles/nature14236

    The observations are sent to three convolutional layers followed by a fully connected layer, each using rectifier
    activation functions (ReLU). The policy and the baseline use fully connected layers built on top of the last hidden
    fully connected layer separately. The policy layer has one unit for each action and its outputs are used as logits
    for a categorical distribution (SoftmaxPolicy). The baseline layer has only one unit which represents its value.
    The weights of the layers are orthogonally initialized.

    Detailed network structure:
    - Conv2D: 32 filters 8x8, stride 4
    - ReLU
    - Conv2D: 64 filters 4x4, stride 2
    - ReLU
    - Conv2D: 64 filters 3x3, stride 1  (number of filters based on argument `conv3_num_filters`)
    - Flatten
    - Fully connected: 512 units
    - ReLU
    - Fully connected (policy): units = number of actions / Fully connected (baseline): 1 unit

    A2C uses 64 filters in the third convolutional layer. ACKTR uses 32.

    The policy is a `actorcritic.policies.SoftmaxPolicy`. The baseline is a `actorcritic.baselines.StateValueFunction`.
    """

    def __init__(self, observation_space, action_space, conv3_num_filters=64, random_seed=None, name=None):
        """Creates a new `AtariModel`.

        observation_space: A `gym.spaces.Box` determining the shape of the observations that will be passed to the
            `observations_placeholder` and the `bootstrap_observations_placeholder`. Used to create these placeholders.
        action_space: A `gym.spaces.Discrete` determining the shape of the actions that will be passed to the
            `actions_placeholder`. Used to create this placeholder.
        conv3_num_filters: An optional number of filters used for the third convolutional layer. ACKTR uses 32 instead
            of 64.
        random_seed: An optional random seed used for the `actorcritic.policies.SoftmaxPolicy`.
        name: An optional name of this model.
        """
        super().__init__(observation_space, action_space)

        assert isinstance(action_space, gym.spaces.Discrete)
        assert isinstance(observation_space, gym.spaces.Box)

        self._num_actions = action_space.n
        self._conv3_num_filters = conv3_num_filters
        self._name = name

        # TODO

        # used to convert the outputs of the policy and the baseline back to the batch-major format of the inputs
        # because the values are flattened in between
        with tf.name_scope('shapes'):
            observations_shape = tf.shape(self._observations_placeholder)
            with tf.name_scope('input_shape'):
                input_shape = observations_shape[:2]
                with tf.name_scope('batch_size'):
                    batch_size = input_shape[0]
                with tf.name_scope('num_steps'):
                    num_steps = input_shape[1]
            with tf.name_scope('bootstrap_input_shape'):
                bootstrap_input_shape = tf.shape(self._bootstrap_observations_placeholder)[:1]

        num_stack = observation_space.shape[-1]

        # the observations are passed in uint8 to save memory and then converted to scalars in range [0,1] on the gpu
        # by dividing by 255
        with tf.name_scope('normalized_observations'):
            normalized_observations = tf.cast(self._observations_placeholder, dtype=tf.float32) / 255.0
            normalized_bootstrap_observations = tf.cast(self._bootstrap_observations_placeholder,
                                                        dtype=tf.float32) / 255.0

        # convert from batch-major format [environment, step] to one flat vector [environment * step] by stacking the
        # steps of each environment
        # this is necessary since the neural network operations only support batch inputs
        with tf.name_scope('flat_observations'):
            self._flat_observations = tf.stop_gradient(
                tf.reshape(normalized_observations, (-1,) + observation_space.shape))
            flat_bootstrap_observations = tf.stop_gradient(
                tf.reshape(normalized_bootstrap_observations, (-1,) + observation_space.shape))

        with tf.variable_scope(self._name, 'AtariModel'):
            self._params = dict()

            # create parameters for all layers
            self._build_params(num_input_channels=num_stack)

            # create layers for the policy and the baseline that use the standard observations as input
            self._preactivations, self._activations = self._build_layers(self._flat_observations, build_policy=True)

            # create layers for the bootstrap values that use the next observations as input
            _, bootstrap_activations = self._build_layers(flat_bootstrap_observations, build_policy=False)

            with tf.name_scope('policy'):
                policy_logits = tf.reshape(self._activations['fc_policy'], [batch_size, num_steps, self._num_actions])
                self._policy = SoftmaxPolicy(policy_logits, random_seed)

            with tf.name_scope('baseline'):
                baseline_logits = tf.reshape(self._activations['fc_baseline'], input_shape)
                self._baseline = StateValueFunction(baseline_logits)

            with tf.name_scope('bootstrap_values'):
                self._bootstrap_values = tf.reshape(bootstrap_activations['fc_baseline'], bootstrap_input_shape)

    def _build_params(self, num_input_channels):
        with tf.name_scope('initializers'):
            # values of the initializers taken from original a2c implementation
            weights_initializer = tf.orthogonal_initializer(np.sqrt(2.), dtype=tf.float32)
            bias_initializer = tf.zeros_initializer(dtype=tf.float32)
            policy_weights_initializer = tf.orthogonal_initializer(0.01, dtype=tf.float32)
            baseline_weights_initializer = tf.orthogonal_initializer(1., dtype=tf.float32)

        with tf.variable_scope('conv1'):
            conv1_num_filters = 32
            conv1_filter_extent = 8
            self._params['conv1'] = nn.conv2d_params(
                num_input_channels, conv1_num_filters, conv1_filter_extent, tf.float32,
                weights_initializer, bias_initializer)

        with tf.variable_scope('conv2'):
            conv2_num_filters = 64
            conv2_filter_extent = 4
            self._params['conv2'] = nn.conv2d_params(
                conv1_num_filters, conv2_num_filters, conv2_filter_extent, tf.float32,
                weights_initializer, bias_initializer)

        with tf.variable_scope('conv3'):
            conv3_filter_extent = 3
            self._params['conv3'] = nn.conv2d_params(
                conv2_num_filters, self._conv3_num_filters, conv3_filter_extent, tf.float32,
                weights_initializer, bias_initializer)

        conv3_flat_size = 49 * self._conv3_num_filters  # TODO don't hardcode

        with tf.variable_scope('fc4'):
            fc4_output_size = 512
            self._params['fc4'] = nn.fully_connected_params(
                conv3_flat_size, fc4_output_size, tf.float32, weights_initializer, bias_initializer)

        with tf.variable_scope('fc_policy'):
            self._params['fc_policy'] = nn.fully_connected_params(
                fc4_output_size, self._num_actions, tf.float32, policy_weights_initializer, bias_initializer)

        with tf.variable_scope('fc_baseline'):
            self._params['fc_baseline'] = nn.fully_connected_params(
                fc4_output_size, 1, tf.float32, baseline_weights_initializer, bias_initializer)

    # noinspection PyShadowingBuiltins
    def _build_layers(self, input, build_policy):
        preactivations = dict()
        activations = dict()

        with tf.variable_scope('conv1', reuse=True):
            conv1_pre = nn.conv2d(input, self._params['conv1'], stride=4, padding='VALID')
            conv1 = tf.nn.relu(conv1_pre)

            preactivations['conv1'] = conv1_pre
            activations['conv1'] = conv1

        with tf.variable_scope('conv2', reuse=True):
            conv2_pre = nn.conv2d(conv1, self._params['conv2'], stride=2, padding='VALID')
            conv2 = tf.nn.relu(conv2_pre)

            preactivations['conv2'] = conv2_pre
            activations['conv2'] = conv2

        with tf.variable_scope('conv3', reuse=True):
            conv3_pre = nn.conv2d(conv2, self._params['conv3'], stride=1, padding='VALID')
            conv3 = tf.nn.relu(conv3_pre)

            preactivations['conv3'] = conv3_pre

        with tf.name_scope('flat'):
            conv3_flat = nn.flatten(conv3)
            activations['conv3'] = conv3_flat

        with tf.variable_scope('fc4', reuse=True):
            fc4_pre = nn.fully_connected(conv3_flat, self._params['fc4'])
            fc4 = tf.nn.relu(fc4_pre)

            preactivations['fc4'] = fc4_pre
            activations['fc4'] = fc4

        if build_policy:
            with tf.variable_scope('fc_policy', reuse=True):
                fc_policy = nn.fully_connected(fc4, self._params['fc_policy'])
                activations['fc_policy'] = fc_policy

        with tf.variable_scope('fc_baseline', reuse=True):
            fc_baseline = nn.fully_connected(fc4, self._params['fc_baseline'])
            activations['fc_baseline'] = fc_baseline

        return preactivations, activations

    def register_layers(self, layer_collection):
        """Registers the layers of this model (neural net) in the specified `kfac.LayerCollection` (required for K-FAC).

        Args:
            layer_collection: A `kfac.LayerCollection`.
        """
        layer_collection.register_conv2d(
            self._params['conv1'], strides=[1, 4, 4, 1], padding='VALID',
            inputs=self._flat_observations, outputs=self._preactivations['conv1'])

        layer_collection.register_conv2d(
            self._params['conv2'], strides=[1, 2, 2, 1], padding='VALID',
            inputs=self._activations['conv1'], outputs=self._preactivations['conv2'])

        layer_collection.register_conv2d(
            self._params['conv3'], strides=[1, 1, 1, 1], padding='VALID',
            inputs=self._activations['conv2'], outputs=self._preactivations['conv3'])

        layer_collection.register_fully_connected(
            self._params['fc4'], inputs=self._activations['conv3'], outputs=self._preactivations['fc4'])

        layer_collection.register_fully_connected(
            self._params['fc_policy'], inputs=self._activations['fc4'], outputs=self._activations['fc_policy'])

        layer_collection.register_fully_connected(
            self._params['fc_baseline'], inputs=self._activations['fc4'], outputs=self._activations['fc_baseline'])
