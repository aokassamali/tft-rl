# Goal: expose the (55x38) masked action grid as a single Discrete(2090) action space.
# Why: makes masked PPO/BC much simpler than MultiDiscrete.

from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from tft_rl.env.action_encoding import grid_to_semantic_action

class GridDiscreteActionWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.n_rows = 55
        self.n_cols = 38
        self.action_space = spaces.Discrete(self.n_rows * self.n_cols)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self._transform_obs(obs)
        return obs, info

    def step(self, action: int):
        row = int(action) // self.n_cols
        col = int(action) % self.n_cols
        semantic = np.asarray(grid_to_semantic_action(row, col), dtype=np.int64)

        obs, r, term, trunc, info = self.env.step(semantic)
        obs = self._transform_obs(obs)
        return obs, r, term, trunc, info

    def _transform_obs(self, obs: dict):
        # Flatten action_mask from (55,38) -> (2090,)
        mask2d = obs["action_mask"]
        obs = dict(obs)  # shallow copy
        obs["action_mask"] = mask2d.reshape(-1)
        return obs
