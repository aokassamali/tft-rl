# Goal: smoke-test the positioning single-step Gym environment.
# Why: confirm obs structure, action mask usage, and the correct action format for env.step().

import numpy as np
from tft_rl.env.wrappers import make_position_single_step_env, adapt_position_action

def describe(x, name="obj"):
    print(f"{name} type:", type(x))
    if isinstance(x, dict):
        print(f"{name} keys:", list(x.keys()))
        for k, v in x.items():
            if hasattr(v, "shape"):
                print(f"  {k}: shape={v.shape}, dtype={getattr(v, 'dtype', None)}")
            else:
                print(f"  {k}: type={type(v)}")
    else:
        if hasattr(x, "shape"):
            print(f"{name} shape:", x.shape, "dtype:", getattr(x, "dtype", None))

def sample_legal_action_from_mask(mask_2d: np.ndarray):
    """
    Goal: sample a legal (i, j) pair from a 2D mask and return a length-3 action list.
    Why: env expects an action list of length 3; mask gives legality over a 2D grid (55 x 38).
    """
    legal = np.argwhere(mask_2d > 0)
    if len(legal) == 0:
        # If mask is empty, fall back to a safe default.
        return [0, 0, 0]
    i = np.random.randint(len(legal))
    a0, a1 = legal[i]
    return [int(a0), int(a1), 0]

def main():
    env = make_position_single_step_env()

    # Reset
    out = env.reset(seed=0) if "seed" in env.reset.__code__.co_varnames else env.reset()
    obs, info = out if isinstance(out, tuple) and len(out) == 2 else (out, {})

    print("=== RESET ===")
    describe(obs, "obs")
    describe(info, "info")

    for t in range(20):
        mask = obs["action_mask"]
        a = sample_legal_action_from_mask(mask)
        a = np.asarray(a, dtype=np.int64)

        next_obs, reward, terminated, truncated, step_info = env.step(a)

        print(f"t={t} reward={reward} terminated={terminated} truncated={truncated} "
            f"state_empty={step_info.get('state_empty')} game_round={step_info.get('game_round')}")

        obs = next_obs

        if terminated or truncated:
            print("Episode ended.")
            break

    print("\n=== ACTION SPACE ===")
    print("action_space:", env.action_space)

    # Option 1: legal sampling via mask (preferred)
    mask = obs["action_mask"]
    a = sample_legal_action_from_mask(mask)

    # Option 2 (fallback): sample from action_space and adapt to length-3
    # a = adapt_position_action(env.action_space.sample())

    print("chosen action (length should be 3):", a)

    print("\n=== STEP ===")
    a = np.asarray(a, dtype=np.int64)
    next_obs, reward, terminated, truncated, step_info = env.step(a)


    describe(next_obs, "next_obs")
    describe(reward, "reward")
    print("terminated:", terminated, "truncated:", truncated)
    describe(step_info, "step_info")

if __name__ == "__main__":
    main()
