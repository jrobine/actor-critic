import concurrent.futures
import multiprocessing
import multiprocessing.managers

import gym


class MultiEnv(object):

    def __init__(self, envs):
        super().__init__()
        self._envs = [_AutoResetWrapper(env) for env in envs]
        self._executor = concurrent.futures.ThreadPoolExecutor(len(self._envs))

    @property
    def envs(self):
        return self._envs

    @property
    def observation_space(self):
        return self._envs[0].observation_space

    @property
    def action_space(self):
        return self._envs[0].action_space

    def reset(self):
        observations = list(self._executor.map(lambda env: env.reset(), self._envs))
        return observations

    def step(self, actions):
        observations, rewards, terminals, infos = zip(*list(
            self._executor.map(lambda x: x[0].step(x[1]) if x[1] is not None else (None, None, None, None),
                               zip(self._envs, actions))))

        return list(observations), list(rewards), list(terminals), list(infos)

    def close(self):
        for env in self._envs:
            self._executor.submit(env.close)

        # self._executor.shutdown(wait=False) # TODO


def create_subprocess_envs(env_fns):
    envs = []
    for env_fn in env_fns:
        env = SubprocessEnv(env_fn)
        env.start()
        envs.append(env)

    # initialize environments in subprocesses
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
    class _Command:
        INIT, STEP, RESET, RENDER, CLOSE = range(5)

    def __init__(self, env_fn):
        self._parent_connection, self._child_connection = multiprocessing.Pipe(duplex=True)
        self._process = multiprocessing.Process(target=SubprocessEnv._child_worker,
                                                args=(self._child_connection, env_fn), name='SubprocessEnv')
        self._process.daemon = True

        self._action_space = None
        self._observation_space = None
        self._started = False
        self._initialized = False

    # parent:

    def start(self):
        assert not self._started
        self._process.start()
        self._child_connection.close()
        self._started = True

    def initialize(self):
        assert self._started
        assert not self._initialized

        self._action_space, self._observation_space = self._communicate(SubprocessEnv._Command.INIT)
        self._initialized = True

    def _communicate(self, command, arg=None):
        # TODO handle BrokenPipeError on exit
        self._parent_connection.send((command, arg))
        return self._parent_connection.recv()

    @property
    def action_space(self):
        assert self._initialized
        return self._action_space

    @property
    def observation_space(self):
        assert self._initialized
        return self._observation_space

    def step(self, action):
        assert self._initialized
        return self._communicate(SubprocessEnv._Command.STEP, action)

    def reset(self, **kwargs):
        assert self._initialized
        return self._communicate(SubprocessEnv._Command.RESET, kwargs)

    def render(self, mode='human'):
        assert self._initialized
        if mode == 'human':
            mode = None
        result = self._communicate(SubprocessEnv._Command.RENDER, mode)
        return result

    def close(self):
        assert self._started
        self._started = False
        self._initialized = False
        self._parent_connection.send((SubprocessEnv._Command.CLOSE, None))
        self._parent_connection.close()
        self._process.join()

    # child:

    @staticmethod
    def _child_worker(child_connection, env_fn):
        env = None
        while True:
            try:
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
                break

        env.close()
        child_connection.close()
