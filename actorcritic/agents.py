"""Contains `agents`, which are an abstraction from environments."""

from abc import ABCMeta, abstractmethod


class Agent(object, metaclass=ABCMeta):
    """Takes environments and a model (containing a policy) and provides :meth:`interact`, which manages operations
    such as selecting actions from the model and stepping in the environments.

    See Also:
        This allows to create multi-step agents, like :obj:`SingleEnvAgent` and :obj:`MultiEnvAgent`.
    """

    @abstractmethod
    def interact(self, session):
        """Samples actions from the model, and steps in the environments.

        Args:
            session (:obj:`tf.Session`):
                A session that will be used to compute the actions.

        Returns:
            :obj:`tuple`:
                A tuple of (`observations`, `actions`, `rewards`, `terminals`, `next_observations`, `infos`).

                All values are in `batch-major` format, meaning that the rows determine the batch and the columns
                determine the time: [`batch`, `time`]. In our case the rows correspond to the environments and the
                columns correspond to the steps: [`environment`, `step`].
                The opposite is the `time-major` format: [`time`, `batch`] or [`step`, `environment`].

                Example:

                    If the agent maintains `3` environments and samples for `5` steps, the result would consist of a
                    matrix (:obj:`list` of :obj:`list`) with shape [`3`, `5`]::

                        [ [step 1, step 2, step 3, step 4, step 5],   # environment 1
                          [step 1, step 2, step 3, step 4, step 5],   # environment 2
                          [step 1, step 2, step 3, step 4, step 5] ]  # environment 3

                `observations`, `actions`, `rewards`, `terminals`, and `infos` are collected during sampling and have
                the shape [`environments`, `steps`].

                `next_observations` contains the observations that the agent received at last, but did not use for
                selecting actions yet. These e.g. can be used to bootstrap the remaining returns. Has the shape
                [`environments`, `1`].
        """
        pass


class SingleEnvAgent(Agent):
    """An agent that maintains a single environment and samples multiple steps.
    """

    def __init__(self, env, model, num_steps):
        """
        Args:
             env (:obj:`gym.Env`):
                An environment.

             model (:obj:`~actorcritic.model.ActorCriticModel`):
                A model to sample actions.

             num_steps (:obj:`int`):
                The number of steps to take in :meth:`interact`.
        """
        self._env = env
        self._model = model
        self._num_steps = num_steps

        self._observation = None  # stores observations between calls of `interact` to reuse `next_observations`

    def interact(self, session):
        """Samples actions from the model and steps in the environment.

        Args:
             session (:obj:`tf.Session`):
                A session that will be used to compute the actions.

        Returns:
            :obj:`tuple`:
                A tuple (`observations`, `actions`, `rewards`, `terminals`, `next_observations`, `infos`).

                All values are in `batch-major` format, meaning that the rows determine the batch and the columns
                determine the time: [`batch`, `time`]. In our case we have `one` environment so the row corresponds to
                the environment and the columns correspond to the steps: [`1`, `step`].
                The opposite is the `time-major` format: [`time`, `batch`] or [`step`, `1`].

                `observations`, `actions`, `rewards`, `terminals`, and `infos` are collected during sampling and have
                the shape [`1`, `steps`].

                `next_observations` contains the observation that the agent received at last, but did not use for
                selecting an action yet. This e.g. can be used to bootstrap the remaining return.
                Has the shape [`1`, `1`].
        """

        # setup time-major values [step]
        observation_steps = []
        action_steps = []
        reward_steps = []
        terminal_steps = []
        info_steps = []

        # get last observations
        next_observation = self._observation
        if next_observation is None:
            next_observation = self._env.reset()

        for _ in range(self._num_steps):
            # save current observations
            observation_steps.append(next_observation)

            action = self._model.sample_actions([[next_observation]], session)[0]
            next_observation, reward, terminal, info = self._env.step(action)

            # save current step
            action_steps.append(action)
            reward_steps.append(reward)
            terminal_steps.append(terminal)
            info_steps.append(info)

        # save for next call of `interact`
        self._observation = next_observation

        # convert from time-major [step] to batch-major values [1, step]
        observation_batch = [observation_steps]
        action_batch = [action_steps]
        reward_batch = [reward_steps]
        terminal_batch = [terminal_steps]
        info_batch = [info_steps]

        return observation_batch, action_batch, reward_batch, terminal_batch, [next_observation], info_batch


class MultiEnvAgent(Agent):
    """An agent that maintains multiple environments (via :obj:`~actorcritic.multi_env.MultiEnv`) and samples multiple
    steps.
    """

    def __init__(self, multi_env, model, num_steps):
        """
        Args:
             multi_env (:obj:`~actorcritic.multi_env.MultiEnv`):
                Multiple environments.

             model (:obj:`~actorcritic.model.ActorCriticModel`):
                A model to sample actions.

             num_steps (:obj:`int`):
                The number of steps to take in :meth:`interact`.
        """
        self._env = multi_env
        self._model = model
        self._num_steps = num_steps

        self._observations = None  # stores observations between calls of `interact` to reuse `next_observations`

    def interact(self, session):
        """Samples actions from the model, and steps in the environments.

        Args:
             session (:obj:`tf.Session`):
                A session that will be used to compute the actions.

        Returns:
            :obj:`tuple`:
                A tuple of (`observations`, `actions`, `rewards`, `terminals`, `next_observations`, `infos`).

                All values are in `batch-major` format, meaning that the rows determine the batch and the columns
                determine the time: [`batch`, `time`]. In our case the rows correspond to the environments and the
                columns correspond to the steps: [`environment`, `step`].
                The opposite is the `time-major` format: [`time`, `batch`] or [`step`, `environment`].

                Example:

                    If the agent maintains `3` environments and samples for `5` steps, the result would consist of a
                    matrix (:obj:`list` of :obj:`list`) with shape [`3`, `5`]::

                        [ [step 1, step 2, step 3, step 4, step 5],   # environment 1
                          [step 1, step 2, step 3, step 4, step 5],   # environment 2
                          [step 1, step 2, step 3, step 4, step 5] ]  # environment 3

                `observations`, `actions`, `rewards`, `terminals`, and `infos` are collected during sampling and have
                the shape [`environments`, `steps`].

                `next_observations` contains the observations that the agent received at last, but did not use for
                selecting actions yet. These e.g. can be used to bootstrap the remaining returns. Has the shape
                [`environments`, `1`].
        """

        # setup time-major values [step, env]
        observation_steps = []
        action_steps = []
        reward_steps = []
        terminal_steps = []
        info_steps = []

        # get last observations
        next_observations = self._observations
        if next_observations is None:
            next_observations = self._env.reset()

        for _ in range(self._num_steps):
            # save current observations
            observation_steps.append(next_observations)

            # convert `next_observations` from [env] to batch-major [env, 1] by transposing [1, env]
            batch_next_observations = transpose_list([next_observations])

            actions = self._model.sample_actions(batch_next_observations, session)
            next_observations, rewards, terminals, infos = self._env.step(actions)

            # save current step
            action_steps.append(actions)
            reward_steps.append(rewards)
            terminal_steps.append(terminals)
            info_steps.append(infos)

        # save for next call of `interact`
        self._observations = next_observations

        # convert from time-major [step, env] to batch-major values [env, step]
        observation_batch = transpose_list(observation_steps)
        action_batch = transpose_list(action_steps)
        reward_batch = transpose_list(reward_steps)
        terminal_batch = transpose_list(terminal_steps)
        info_batch = transpose_list(info_steps)

        return observation_batch, action_batch, reward_batch, terminal_batch, next_observations, info_batch


def transpose_list(values):
    """Transposes a list of lists. Can be used to convert from `time-major` format to `batch-major` format and vice
    versa.

    Example:
        Input::

            [[1, 2, 3, 4],
             [5, 6, 7, 8],
             [9, 10, 11, 12]]

        Output::

            [[1, 5, 9],
             [2, 6, 10],
             [3, 7, 11],
             [4, 8, 12]]

    Args:
        values (:obj:`list` of :obj:`list`):
            Values to transpose.

    Returns:
        :obj:`list` of :obj:`list`:
            The transposed values.
    """
    return list(map(list, zip(*values)))  # taken from: https://stackoverflow.com/a/6473724
