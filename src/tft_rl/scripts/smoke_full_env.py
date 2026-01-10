# Goal: smoke-test the full multi-agent PettingZoo AEC environment.
# Why: confirm per-agent obs/action/reward formats and the turn-based stepping API.

import numpy as np
from tft_rl.env.wrappers import make_full_game_env

def describe(x, name="obj"):
    print(f"{name} type:", type(x))
    if isinstance(x, dict):
        print(f"{name} keys:", list(x.keys())[:20])
        for k, v in list(x.items())[:8]:
            if hasattr(v, "shape"):
                print(f"  {k}: shape={v.shape}, dtype={getattr(v, 'dtype', None)}")
            else:
                print(f"  {k}: type={type(v)}")
    else:
        if hasattr(x, "shape"):
            print(f"{name} shape:", x.shape, "dtype:", getattr(x, "dtype", None))

def main():
    env = make_full_game_env()
    env.reset(seed=0)

    print("agents:", env.agents[:10])
    print("possible_agents:", getattr(env, "possible_agents", None))

    # Step through a small number of turns to inspect API.
    n_turns = 20
    for i, agent in enumerate(env.agent_iter(max_iter=n_turns)):
        obs, reward, terminated, truncated, info = env.last()

        print(f"\n=== TURN {i} | agent={agent} ===")
        describe(obs, "obs")
        describe(reward, "reward")
        print("terminated:", terminated, "truncated:", truncated)
        describe(info, "info")

        if terminated or truncated:
            action = None  # PettingZoo convention: pass None if agent is done
        else:
            # Most AEC envs expose action_space(agent)
            space = env.action_space(agent) if callable(getattr(env, "action_space", None)) else None
            print("action_space(agent):", space)

            if space is not None:
                action = space.sample()
            else:
                action = 0

        env.step(action)

    print("\nSmoke test (full env) finished.")

if __name__ == "__main__":
    main()
