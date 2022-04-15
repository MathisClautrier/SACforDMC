
import gym
import numpy as np
from collections import deque
import os

def make_dir(dir_path):
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return 

def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )

class FrameStack(gym.Wrapper):
    """Modified version of the code taken from curl official implementation:
    https://github.com/MishaLaskin/curl/blob/master/utils.py
    """
    def __init__(self, env, k,img_size=100):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        self.img_size = img_size
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=env.observation_space.dtype
        )
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs = self.env.reset()
        if self.img_size !=100:
          obs = self.env.render('rgb_array', width=self.img_size, height=self.img_size).transpose(2,0,1)
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if self.img_size !=100:
          obs = self.env.render('rgb_array', width=self.img_size, height=self.img_size).transpose(2,0,1)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)