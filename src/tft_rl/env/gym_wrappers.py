# Goal: provide Gymnasium wrappers to normalize rewards/terminations for learning.
# Why: upstream env returns cumulative reward; RL algorithms expect per-step reward.

from __future__ import annotations
import gymnasium as gym

class DeltaRewardWrapper(gym.Wrapper):
    """
    Converts cumulative rewards into per-step delta rewards:
      r_delta[t] = r_total[t] - r_total[t-1]
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._prev_total = 0

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        self._prev_total = 0
        return out

    def step(self, action):
        obs, total_r, terminated, truncated, info = self.env.step(action)
        r = total_r - self._prev_total
        self._prev_total = total_r
        return obs, r, terminated, truncated, info
