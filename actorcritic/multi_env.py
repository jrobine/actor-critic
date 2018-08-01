"""Contains classes that provide the ability to run multiple environments in subprocesses."""


import concurrent.futures
import multiprocessing
import multiprocessing.managers

import gym


class MultiEnv(object):
    """An environment that maintains multiple :obj:`SubprocessEnv`\s and executes them in parallel.

    The environments will be reset automatically when a terminal state is reached. That means that :meth:`reset`
    actually only has to be called once in the beginning.
    """

    def __init__(self, envs):
        """
        Args:
            envs (:obj:`list` of :obj:`SubprocessEnv`):
                The environments. The observation and action spaces must be equal across the environments.
        """
        super().__init__()
        self._envs = [_AutoResetWrapper(env) for env in envs]
        self._executor = concurrent.futures.ThreadPoolExecutor(len(self._envs))

    @property
    def envs(self):
        """:obj:`list` of :obj:`gym.Env`:
            The environments.
        """
        return self._envs

    @property
    def observation_space(self):
        """:obj:`gym.spaces.Space`:
            The observation space used by all environments.
        """
        return self._envs[0].observation_space

    @property
    def action_space(self):
        """:obj:`gym.spaces.Space`:
            The action space used by all environments.
        """
        return self._envs[0].action_space

    def reset(self):
        """Resets all environments.

         Returns:
            :obj:`list`:
                A list of observations received from each environment.
        """
        observations = list(self._executor.map(lambda env: env.reset(), self._envs))
        return observations

    def step(self, actions):
        """Proceeds one step in all environments.

        Args:
            actions (:obj:`list`):
                A list of actions to be executed in the environments.

        Returns:
            :obj:`tuple`:
                A tuple of (`observations`, `rewards`, `terminals`, `infos`). Each element is a list containing the
                values received from the environments.
        """

        def call_step(env_action):
            env, action = env_action
            if action is None:
                return None, None, None, None
            return env.step(action)

        observations, rewards, terminals, infos = zip(*list(
            self._executor.map(call_step, zip(self._envs, actions))))

        return list(observations), list(rewards), list(terminals), list(infos)

    def close(self):
        """Closes all environments.
        """
        for env in self._envs:
            self._executor.submit(env.close)

        self._executor.shutdown()


def create_subprocess_envs(env_fns):
    """Utility function that creates environments by calling the functions in `env_fns` and wrapping the returned
    environments in :obj:`SubprocessEnv`\s. They will be started and initialized in parallel.

    Args:
        env_fns (:obj:`list` of :obj:`callable`):
            A list of functions that return a :obj:`gym.Env`. They should not be instances of :obj:`SubprocessEnv`.

    Returns:
        :obj:`list` of :obj:`SubprocessEnv`:
            A list of the created environments.
    """

    # create processes and let them create the environments in parallel
    envs = []
    for env_fn in env_fns:
        env = SubprocessEnv(env_fn)
        env.start()
        envs.append(env)

    # call `initialize()`, which blocks the execution, in parallel using multiple threads
    # this also ensures that the creation of the environments is finished when returning from this function
    with concurrent.futures.ThreadPoolExecutor(len(envs)) as executor:
        for env in envs:
            executor.submit(env.initialize)

    return envs


class _AutoResetWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self._terminated = False

    def step(self, action):
        if self._terminated:
            self.env.reset()
        observation, reward, terminal, info = self.env.step(action)
        self._terminated = terminal
        return observation, reward, terminal, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self._terminated = False
        return observation


class SubprocessEnv(gym.Env):
    """Maintains a :obj:`gym.Env` inside a subprocess, so it can run concurrently. If the subprocess ends unexpectedly,
    it will be recreated automatically without interrupting the execution.

    To use the subprocess :meth:`start` has to be called first. After that :meth:`initialize` has to be called to
    retrieve the observation space and the action space from the underlying environment. The purpose of these methods is
    that multiple :obj:`SubprocessEnv`\s can be created and started in parallel without blocking the execution, which
    creates the underlying :obj:`gym.Env` already. Afterwards :meth:`start`, which blocks the execution, can be called
    in parallel. See :meth:`create_subprocess_envs` which implements this idea.
    """

    class _Command:
        INIT, STEP, RESET, RENDER, CLOSE = range(5)

    def __init__(self, env_fn):
        """
        Args:
            env_fn (:obj:`callable`):
                A function that returns a :obj:`gym.Env`. It will be called inside the subprocess, so watch out for
                referencing variables on the main process or the like. It possibly will be called multiple times, since
                the subprocess will be recreated when it unexpectedly ends.
        """
        self._env_fn = env_fn
        self._parent_connection, self._child_connection, self._process = None, None, None
        self._started = False
        self._initialized = False
        self._reset()

        self._action_space = None
        self._observation_space = None

    def _check_closed(self):
        if self._process is None:
            raise ValueError('The subprocess was closed already.')

    def _check_initialized(self, method):
        self._check_closed()

        if not self._started:
            raise ValueError('The subprocess is not started yet. '
                             'Call \'start()\' and \'initialize()\' before calling \'{}\'.'.format(method))

        if not self._initialized:
            raise ValueError('The subprocess is not initialized yet. '
                             'Call \'initialize()\' before calling \'{}\'.'.format(method))

    def _reset(self):
        self._started = False
        self._initialized = False
        self._parent_connection, self._child_connection = multiprocessing.Pipe(duplex=True)
        self._process = multiprocessing.Process(target=SubprocessEnv._child_worker,
                                                args=(self._child_connection, self._env_fn), name='SubprocessEnv')
        self._process.daemon = True

    def start(self):
        """Starts the subprocess. Does not block. You should call :meth:`initialize` afterwards.
        """
        self._check_closed()

        if not self._started:
            self._process.start()
            self._child_connection.close()
            self._started = True

    def initialize(self):
        """Retrieves the observation space and the action space from the environment in the subprocess. This method
        blocks until the execution is finished. :meth:`start` must have been called.
        """
        self._check_closed()

        if not self._started:
            raise ValueError('The subprocess is not started yet. Call \'start()\' before \'initialize()\'.')

        if not self._initialized:
            self._action_space, self._observation_space = self._communicate(SubprocessEnv._Command.INIT)
            self._initialized = True

    def _communicate(self, command, arg=None):
        try:
            return self._unsafe_communicate(command, arg)
        except (BrokenPipeError, ConnectionResetError, EOFError):
            # FIXME small bug: also restarts when exiting program with KeyboardInterrupt

            # restart process when connection is lost
            self._parent_connection.close()
            self._child_connection.close()
            self._process.terminate()
            self._process.join()

            self._reset()
            self.start()

            # initialize
            self._action_space, self._observation_space = self._unsafe_communicate(SubprocessEnv._Command.INIT)
            self._initialized = True

            if command != SubprocessEnv._Command.RESET:
                self._unsafe_communicate(SubprocessEnv._Command.RESET)

            return self._unsafe_communicate(command, arg)

    def _unsafe_communicate(self, command, arg=None):
        self._parent_connection.send((command, arg))
        return self._parent_connection.recv()

    @property
    def action_space(self):
        """:obj:`gym.spaces.Space`:
            The action space of the underlying environment. Does not block the execution. :meth:`start` and
            :meth:`initialize` must have been called.
        """
        self._check_initialized('action_space')
        return self._action_space

    @property
    def observation_space(self):
        """:obj:`gym.spaces.Space`:
            The observation space of the underlying environment. Does not block the execution. :meth:`start` and
            :meth:`initialize` must have been called.
        """
        self._check_initialized('observation_space')
        return self._observation_space

    def step(self, action):
        """Remotely calls :meth:`gym.Env.step` in the underlying environment. This method blocks until execution is
        finished. :meth:`start` and :meth:`initialize` must have been called.

        Args:
            action:
                The `action` argument passed to :meth:`gym.Env.step`.

        Returns:
            :obj:`tuple`:
                A tuple of (`observation`, `reward`, `terminal`, `info`). The values returned by
                :meth:`gym.Env.step`.
        """
        self._check_initialized('step()')
        return self._communicate(SubprocessEnv._Command.STEP, action)

    def reset(self, **kwargs):
        """Remotely calls :meth:`gym.Env.reset` in the underlying environment. This method blocks until execution is
        finished. :meth:`start` and :meth:`initialize` must have been called.

        Args:
            kwargs (:obj:`dict`):
                Keyword arguments passed to :meth:`gym.Env.reset`.

        Returns:
            The value returned by :meth:`gym.Env.reset`.
        """
        self._check_initialized('reset()')
        return self._communicate(SubprocessEnv._Command.RESET, kwargs)

    def render(self, mode='human'):
        """Remotely calls :meth:`gym.Env.render` in the subprocess. This methods blocks until execution is finished.
        :meth:`start` and :meth:`initialize` must have been called.

        Args:
            mode (:obj:`str`):
                The `mode` argument passed to :meth:`gym.Env.render`.

        Returns:
            The value returned by :meth:`gym.Env.render`.
        """
        self._check_initialized('render()')
        if mode == 'human':
            mode = None   # treat 'human' as default mode, so don't need to send
        result = self._communicate(SubprocessEnv._Command.RENDER, mode)
        return result

    def close(self):
        """Closes the subprocess.
        """
        self._check_closed()

        if self._started:
            try:
                if self._process.is_alive():
                    self._parent_connection.send((SubprocessEnv._Command.CLOSE, None))
                self._parent_connection.close()
                self._child_connection.close()
            except (BrokenPipeError, ConnectionResetError, EOFError):
                pass

        self._started = False
        self._initialized = False

        self._process.join()
        self._process = None

    @staticmethod
    def _child_worker(child_connection, env_fn):
        env = None
        try:
            while True:
                command, arg = child_connection.recv()
                response = None
                if command == SubprocessEnv._Command.INIT:
                    env = env_fn()
                    response = (env.action_space, env.observation_space)
                elif command == SubprocessEnv._Command.STEP:
                    action = arg
                    response = env.step(action)
                elif command == SubprocessEnv._Command.RESET:
                    kwargs = arg
                    if kwargs is None:
                        kwargs = dict()
                    response = env.reset(**kwargs)
                elif command == SubprocessEnv._Command.RENDER:
                    mode = arg
                    if mode is None:
                        mode = 'human'
                    response = env.render(mode)
                elif command == SubprocessEnv._Command.CLOSE:
                    break

                child_connection.send(response)
        except KeyboardInterrupt:
            pass

        if env is not None:
            env.close()
        child_connection.close()
