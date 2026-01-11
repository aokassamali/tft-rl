# Goal: smoke-test the GridDiscreteActionWrapper version of the env.
# Why: action_mask is now flattened to shape (2090,) and action_space is Discrete(2090).

import numpy as np
from tft_rl.env.wrappers import make_position_single_step_env

def sample_legal_action_index(mask_1d: np.ndarray) -> int:
    legal = np.flatnonzero(mask_1d > 0)
    if legal.size == 0:
        return 0
    return int(np.random.choice(legal))

def main():
    env = make_position_single_step_env()

    out = env.reset(seed=0) if "seed" in env.reset.__code__.co_varnames else env.reset()
    obs, info = out if isinstance(out, tuple) and len(out) == 2 else (out, {})

    print("=== RESET ===")
    print("obs keys:", list(obs.keys()))
    print("action_space:", env.action_space)
    print("action_mask shape:", obs["action_mask"].shape, "dtype:", obs["action_mask"].dtype, "obs_shape:", obs["obs"].shape, "lega_action_count:", obs["action_mask"].sum())

    for t in range(30):
        mask = obs["action_mask"]
        a = sample_legal_action_index(mask)

        next_obs, reward, terminated, truncated, step_info = env.step(a)

        print(
            f"t={t} a={a} r={reward} term={terminated} trunc={truncated} "
            f"round={step_info.get('game_round')} empty={step_info.get('state_empty')}"
        )

        obs = next_obs
        if terminated or truncated:
            print("Episode ended.")
            break

if __name__ == "__main__":
    main()

