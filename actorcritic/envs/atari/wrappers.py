import time

import cv2
import gym
import numpy as np

cv2.ocl.setUseOpenCL(False)


# adopted from https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py


class AtariPreprocessFrameWrapper(gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
        return np.expand_dims(frame, axis=-1)


class AtariFrameskipWrapper(gym.Wrapper):

    def __init__(self, env, frameskip):
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

    def reward(self, reward):
        return np.clip(reward, -1., 1.)


class AtariEpisodicLifeWrapper(gym.Wrapper):

    def __init__(self, env):
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

    def __init__(self, env, noop_max):
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

    def __init__(self, env, fps=None):
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

    def __init__(self, env, num_stacked_frames):
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

    def step(self, action):
        observation, reward, terminal, info = self.env.step(action)
        del info['ale.lives']
        return observation, reward, terminal, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class EpisodeInfoWrapper(gym.Wrapper):

    def __init__(self, env):
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
