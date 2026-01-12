# Goal: reduce training noise from upstream mask imperfections by penalizing likely-invalid actions.
# Why: the simulator sometimes marks actions legal that still fail at runtime.

from __future__ import annotations
import gymnasium as gym
import numpy as np
import traceback
from gymnasium import spaces

class InvalidActionPenaltyWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, penalty: float = -0.01):
        super().__init__(env)
        self.penalty = penalty
        self._prev_round = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_round = None
        return obs, info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)

        # Heuristic: if nothing progressed and no reward event happened, apply tiny penalty.
        # Uses game_round if present.
        cur_round = info.get("game_round", None)
        if self._prev_round is None:
            self._prev_round = cur_round

        likely_no_progress = (cur_round == self._prev_round) and (r == 0)
        if likely_no_progress:
            r = r + self.penalty

        self._prev_round = cur_round
        return obs, r, term, trunc, info


# Goal: prevent rare simulator crashes from killing training runs.
# Why: third-party env has occasional None-handling bugs.

class CrashShieldWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, crash_reward: float = -1.0):
        super().__init__(env)
        self.crash_reward = crash_reward
        self._last_obs = None

        # Build a safe fallback obs from reset to use if step crashes
        obs, _ = self.env.reset()
        self._fallback_obs = obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        return obs, info

    def step(self, action):
        try:
            obs, r, term, trunc, info = self.env.step(action)
            self._last_obs = obs
            return obs, r, term, trunc, info
        except Exception as e:
            # Print crash info so we can later fix root cause
            print("=== ENV CRASH ===")
            print(repr(e))
            traceback.print_exc()

            # Return a terminal transition so the rollout can continue
            obs = self._last_obs if self._last_obs is not None else self._fallback_obs
            info = {"env_crash": True, "exception": repr(e)}
            return obs, float(self.crash_reward), True, False, info
