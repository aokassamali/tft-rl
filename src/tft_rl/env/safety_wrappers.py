# Goal: reduce training noise from upstream mask imperfections by penalizing likely-invalid actions.
# Why: the simulator sometimes marks actions legal that still fail at runtime.

from __future__ import annotations
import gymnasium as gym

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
