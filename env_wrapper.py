import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F


class PixelEnvWrapper(gym.Wrapper):
    """Wraps any pixel-based Gymnasium env: resizes obs to 64x64, normalizes to [0,1].
    Exposes action_dim, is_discrete, and obs_channels for use in train.py."""

    def __init__(self, env_id, size=64):
        env = gym.make(env_id)
        super().__init__(env)
        self.size = size

        if isinstance(env.action_space, gym.spaces.Discrete):
            self.is_discrete = True
            self.action_dim = env.action_space.n        # one-hot size
        else:
            self.is_discrete = False
            self.action_dim = env.action_space.shape[0]

        self.obs_channels = env.observation_space.shape[2]  # 1 or 3

    def _preprocess(self, obs):
        # obs: (H, W, C) uint8 -> (C, 64, 64) float32 in [0, 1]
        t = torch.from_numpy(obs).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        t = F.interpolate(t, size=(self.size, self.size), mode="bilinear", align_corners=False)
        return t.squeeze(0)  # (C, 64, 64)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._preprocess(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._preprocess(obs), reward, terminated, truncated, info
