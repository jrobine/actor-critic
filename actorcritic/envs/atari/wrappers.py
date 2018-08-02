"""Contains `wrappers` that can wrap around environments to modify their functionality.

The implementations of these wrappers are adopted from
`OpenAI <https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py>`_.
"""

import time

import cv2
import gym
import numpy as np

cv2.ocl.setUseOpenCL(False)  # do not use OpenCL


class AtariPreprocessFrameWrapper(gym.ObservationWrapper):
    """A wrapper that scales the observations from 210x160 down to 84x84 and converts from RGB to grayscale by
    extracting the luminance.
    """

    def __init__(self, env):
        """
        Args:
            env (:obj:`gym.Env`):
                An environment that will be wrapped.
        """
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # convert to grayscale
        frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)  # scale down
        return np.expand_dims(frame, axis=-1)


class AtariFrameskipWrapper(gym.Wrapper):
    """A wrapper that skips frames.
    """

    def __init__(self, env, frameskip):
        """
        Args:
            env (:obj:`gym.Env`):
                An environment that will be wrapped.

            frameskip (:obj:`int`):
                Every `frameskip`-th frame is used. The remaining frames are skipped.
        """
        super().__init__(env)
        self._frameskip = frameskip

    def step(self, action):
        frames = []
        total_reward = 0.0
        terminal = False
        info = None
        for i in range(self._frameskip):
            next_frame, reward, terminal, info = self.env.step(action)
            frames.append(next_frame)
            total_reward += reward
            if terminal:
                break

        if len(frames) >= 2:
            return np.amax((frames[-2], frames[-1]), axis=0), total_reward, terminal, info
        else:
            return frames[0], total_reward, terminal, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class AtariClipRewardWrapper(gym.RewardWrapper):
    """A wrapper that clips the rewards between -1 and 1.
    """

    def __init__(self, env):
        """
        Args:
            env (:obj:`gym.Env`):
                An environment that will be wrapped.
        """
        super().__init__(env)

    def reward(self, reward):
        return np.clip(reward, -1., 1.)


class AtariEpisodicLifeWrapper(gym.Wrapper):
    """A wrapper that ends episodes (returns `terminal` = True) after a life in the Atari game has been lost.
    """

    def __init__(self, env):
        """
        Args:
            env (:obj:`gym.Env`):
                An environment that will be wrapped.
        """
        super().__init__(env)
        self.lives = 0
        self.episode_terminal = True

    def step(self, action):
        next_observation, reward, terminal, info = self.env.step(action)
        self.episode_terminal = terminal
        next_lives = info['ale.lives']
        if next_lives < self.lives:
            terminal = True
        self.lives = next_lives
        return next_observation, reward, terminal, info

    def reset(self, **kwargs):
        if self.episode_terminal:
            self.env.reset(**kwargs)
        observation, _, terminal, info = self.env.step(0)  # ACTION_NOOP
        self.lives = info['ale.lives']
        return observation


class AtariFireResetWrapper(gym.Wrapper):
    """A wrapper that executes the `'FIRE'` action after the environment has been reset.
    """

    def __init__(self, env):
        """
        Args:
            env (:obj:`gym.Env`):
                An environment that will be wrapped.
        """
        super().__init__(env)

    def step(self, action):
        return self.env.step(action)

    def reset(self, **kwargs):
        # TODO
        self.env.reset(**kwargs)
        observation, _, terminal, _ = self.env.step(1)  # ACTION_FIRE
        if terminal:
            print('WARNING')
            observation = self.env.reset(**kwargs)
        return observation


class AtariNoopResetWrapper(gym.Wrapper):
    """A wrapper that executes a random number of `'NOOP'` actions.
    """

    def __init__(self, env, noop_max):
        """
        Args:
            env (:obj:`gym.Env`):
                An environment that will be wrapped.

            noop_max (:obj:`int`):
                The maximum number of `'NOOP'` actions. The number is selected randomly between 1 and `noop_max`.
        """
        super().__init__(env)
        self.noop_max = noop_max

    def step(self, action):
        return self.env.step(action)

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        num_noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)
        for _ in range(num_noops):
            observation, _, terminal, _ = self.env.step(0)  # ACTION_NOOP
            if terminal:
                observation = self.env.reset(**kwargs)
        return observation


class RenderWrapper(gym.Wrapper):
    """A wrapper that calls :meth:`gym.Env.render` every step.
    """

    def __init__(self, env, fps=None):
        """
        Args:
            env (:obj:`gym.Env`):
                An environment that will be wrapped.

            fps (:obj:`int`, :obj:`float`, optional):
                If it is not None, the steps will be slowed down to run at the specified frames per second by waiting
                1.0/`fps` seconds every step.
        """
        super().__init__(env)
        self._spf = 1.0 / fps if fps is not None else None

    def step(self, action):
        self.env.render()
        if self._spf is not None:
            time.sleep(self._spf)
        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class FrameStackWrapper(gym.Wrapper):
    """A wrapper that stacks the last observations. The observations returned by this wrapper consist of the last
    frames.
    """

    def __init__(self, env, num_stacked_frames):
        """
        Args:
            env (:obj:`gym.Env`):
                An environment that will be wrapped.

            num_stacked_frames (:obj:`int`):
                The number of frames that will be stacked.
        """
        super().__init__(env)
        self._num_stacked_frames = num_stacked_frames

        stacked_low = np.repeat(env.observation_space.low, num_stacked_frames, axis=-1)
        stacked_high = np.repeat(env.observation_space.low, num_stacked_frames, axis=-1)
        self.observation_space = gym.spaces.Box(low=stacked_low, high=stacked_high, dtype=env.observation_space.dtype)

        self._stacked_frames = np.zeros_like(stacked_low)

    def step(self, action):
        next_frame, reward, terminal, info = self.env.step(action)
        self._stacked_frames = np.roll(self._stacked_frames, shift=-1, axis=-1)
        if terminal:
            self._stacked_frames.fill(0.0)
        self._stacked_frames[..., -1:] = next_frame
        return self._stacked_frames, reward, terminal, info

    def reset(self, **kwargs):
        frame = self.env.reset(**kwargs)
        self._stacked_frames = np.repeat(frame, self._num_stacked_frames, axis=-1)
        return self._stacked_frames


class AtariInfoClearWrapper(gym.Wrapper):
    """A wrapper that removes unnecessary data in the `info` returned by :meth:`gym.Env.step`. This reduces the amount
    of inter-process data.

    Warning:
        :obj:`AtariEpisodicLifeWrapper` does not work afterwards, so it should be used `before`.
    """

    def __init__(self, env):
        """
        Args:
            env (:obj:`gym.Env`):
                An environment that will be wrapped.
        """
        super().__init__(env)

    def step(self, action):
        observation, reward, terminal, info = self.env.step(action)
        del info['ale.lives']
        return observation, reward, terminal, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class EpisodeInfoWrapper(gym.Wrapper):
    """A wrapper that stores episode information in the `info` returned by :meth:`gym.Env.step` at the end of an
    episode. More specifically, if an episode is terminal, `info` will contain the key `'episode'` which has a
    :obj:`dict` value containing the `'total_reward'`, which is the cumulative reward of the episode.

    Note:
        If you want to get the cumulative reward of the entire episode, :obj:`AtariEpisodicLifeWrapper` should be used
        `after` this wrapper.
    """

    def __init__(self, env):
        """
        Args:
            env (:obj:`gym.Env`):
                An environment that will be wrapped.
        """
        super().__init__(env)
        self.total_reward = 0.0

    def step(self, action):
        observation, reward, terminal, info = self.env.step(action)
        self.total_reward += reward
        if terminal:
            episode_info = dict()
            episode_info['total_reward'] = self.total_reward
            info['episode'] = episode_info
            self.total_reward = 0.0
        return observation, reward, terminal, info

    def reset(self, **kwargs):
        self.total_reward = 0.0
        return self.env.reset(**kwargs)

    @staticmethod
    def get_episode_rewards_from_info_batch(infos):
        """Utility function that extracts the episode rewards, that are inserted by the :obj:`EpisodeInfoWrapper`, out
        of the `infos`.

        Args:
            infos (:obj:`list` of :obj:`list`):
                A batch-major list of `infos` as returned by :meth:`~actorcritic.agents.Agent.interact`.

        Returns:
            :obj:`numpy.ndarray`:
                A batch-major array with the same shape as infos. It contains the episode reward of an `info` at the
                corresponding position. If no episode reward was in an `info`, the result will contain
                :obj:`numpy.nan` respectively.
        """

        rewards = np.full_like(infos, np.nan, np.float32)
        environments, steps = rewards.shape

        for environment in range(environments):
            for step in range(steps):
                info = infos[environment][step]

                if 'episode' in info:
                    reward = info['episode']['total_reward']
                    rewards[environment, step] = reward

        return rewards
