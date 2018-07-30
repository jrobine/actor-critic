from abc import ABCMeta, abstractmethod


class Agent(object, metaclass=ABCMeta):

    @abstractmethod
    def interact(self, session):
        pass


class SingleEnvAgent(Agent):

    def __init__(self, env, model, num_steps):
        self._env = env
        self._model = model
        self._num_steps = num_steps

        self._observation = None  # stores observations between calls of `interact` to reuse `next_observations`

    def interact(self, session):
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

    def __init__(self, multi_env, model, num_steps):
        self._env = multi_env
        self._model = model
        self._num_steps = num_steps

        self._observations = None  # stores observations between calls of `interact` to reuse `next_observations`

    def interact(self, session):
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
    """Transposes a list of lists.

    e.g.     [ [1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12] ]

    becomes  [ [1, 5, 9], [2, 6, 10], [3, 7, 11], [4, 8, 12] ]

    Args:
        values: The list of lists.

    Returns:
        A list containing the tranposed lists.
    """
    return list(map(list, zip(*values)))  # taken from: https://stackoverflow.com/a/6473724
