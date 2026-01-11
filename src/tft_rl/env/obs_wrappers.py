# Goal: flatten nested observation dict into a single 1D float32 vector for PPO.
# Why: CleanRL policies expect fixed-size tensors.

from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces

def _flatten_any(x) -> np.ndarray:
    if isinstance(x, dict):
        parts = []
        for k in sorted(x.keys()):
            parts.append(_flatten_any(x[k]))
        return np.concatenate(parts, axis=0) if parts else np.zeros((0,), dtype=np.float32)
    if isinstance(x, (list, tuple)):
        parts = [_flatten_any(v) for v in x]
        return np.concatenate(parts, axis=0) if parts else np.zeros((0,), dtype=np.float32)
    if isinstance(x, np.ndarray):
        return x.astype(np.float32).ravel()
    if np.isscalar(x):
        return np.asarray([x], dtype=np.float32)
    # Unknown type: ignore
    return np.zeros((0,), dtype=np.float32)

class FlattenObsWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        # Build a sample obs to determine flat dim
        obs, _ = env.reset()
        flat = _flatten_any(obs["observations"])
        self._flat_dim = int(flat.shape[0])

        # Observation will become dict: {"obs": flat_vec, "action_mask": mask_vec}
        self.observation_space = spaces.Dict({
            "obs": spaces.Box(low=-np.inf, high=np.inf, shape=(self._flat_dim,), dtype=np.float32),
            "action_mask": env.observation_space["action_mask"] if isinstance(env.observation_space, spaces.Dict) else spaces.Box(0, 1, shape=(env.action_space.n,), dtype=np.float32),
        })

    def observation(self, obs):
        flat = _flatten_any(obs["observations"])
        return {
            "obs": flat,
            "action_mask": obs["action_mask"],
        }
