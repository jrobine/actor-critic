"""The implementations of these wrappers are adopted from:

    https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py

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
        """Creates a new `AtariPreprocessFrameWrapper`.

        Args:
            env: The environment that is wrapped.
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
        """Creates a new `AtariFrameskipWrapper`.

        Args:
            env: The environment that is wrapped.
            frameskip: Every `frameskip`-th is used, the remaining frames are skipped.
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
        """Creates a new `AtariClipRewardWrapper`.

        Args:
            env: The environment that is wrapped.
        """
        super().__init__(env)

    def reward(self, reward):
        return np.clip(reward, -1., 1.)


class AtariEpisodicLifeWrapper(gym.Wrapper):
    """A wrapper that ends episodes (returns terminal = True) after a life in the Atari game has been lost.
    """

    def __init__(self, env):
        """Creates a new `AtariEpisodicLifeWrapper`.

        Args:
            env: The environment that is wrapped.
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
    """A wrapper that executes the 'FIRE' action after the environment has been reset.
    """

    def __init__(self, env):
        """Creates a new `AtariFireResetWrapper`.

        Args:
            env: The environment that is wrapped.
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
    """A wrapper that execute a random number of 'NOOP' actions.
    """

    def __init__(self, env, noop_max):
        """Creates a new `AtariNoopResetWrapper`.

        Args:
            env: The environment that is wrapped.
            noop_max: The maximum number of 'NOOP' actions. The number is selected randomly between 0 and `noop_max`.
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
    """A wrapper that calls the `render()` method at each `step()` of the environment.
    """

    def __init__(self, env, fps=None):
        """Creates a new `AtariNoopResetWrapper`.

        Args:
            env: The environment that is wrapped.
            fps: An optional scalar. If it is not None, the steps will be slowed down to run at the specified frames
                per second by waiting 1.0/`fps` seconds after each `step()`.
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
    """A wrapper that stacks the last observations, so the observations returned by this wrapper consist of the last
    frames.
    """

    def __init__(self, env, num_stacked_frames):
        """Creates a new `FrameStackWrapper`.

        Args:
            env: The environment that is wrapped.
            fps: The number of frames that will be stacked.
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
    """A wrapper that removes unnecessary data in the `info` returned by `step()`. This reduces the amount of
    inter-process data. `AtariEpisodicLifeWrapper` does not work after that, so it should be used before.
    """

    def __init__(self, env):
        """Creates a new `AtariInfoClearWrapper`.

        Args:
            env: The environment that is wrapped.
        """
        super().__init__(env)

    def step(self, action):
        observation, reward, terminal, info = self.env.step(action)
        del info['ale.lives']
        return observation, reward, terminal, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class EpisodeInfoWrapper(gym.Wrapper):
    """A wrapper that stores episode information in the `info` returned by `step()` at the end of an episode. More
    specifically, if an episode is terminal, `info` will contain the key 'episode' which has a dict() value containing
    the 'total_reward', which is the cumulative reward of the episode.
    If you want to get the cumulative reward of the entire episode, `AtariEpisodicLifeWrapper` should be used after
    this wrapper.
    """

    def __init__(self, env):
        """Creates a new `EpisodeInfoWrapper`.

        Args:
            env: The environment that is wrapped.
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
