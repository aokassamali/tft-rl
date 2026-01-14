# Goal: evaluate a saved policy checkpoint on the env.
# Why: training curves are noisy; eval gives a stable "exam score".

from __future__ import annotations
import argparse
import numpy as np
import torch
import gymnasium as gym

from tft_rl.env.wrappers import make_position_single_step_env
from tft_rl.train.ppo_masked import Agent  # reuse the same network class

def masked_greedy_action(agent: Agent, obs_vec: np.ndarray, mask: np.ndarray, device: torch.device) -> int:
    obs_t = torch.tensor(obs_vec[None, :], dtype=torch.float32, device=device)
    mask_t = torch.tensor(mask[None, :], dtype=torch.float32, device=device)

    with torch.no_grad():
        h = agent.shared(obs_t)
        logits = agent.pi(h)

        # ensure at least one legal action (pass=0)
        if mask_t.sum().item() == 0:
            mask_t[:, 0] = 1.0

        neg_inf = torch.finfo(logits.dtype).min
        masked_logits = torch.where(mask_t > 0, logits, neg_inf)

        a = int(torch.argmax(masked_logits, dim=1).item())
        return a

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="checkpoints/ppo_masked_latest.pt")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    env = make_position_single_step_env()
    obs_dim = env.observation_space["obs"].shape[0]
    n_actions = env.action_space.n

    agent = Agent(obs_dim, n_actions).to(device)
    agent.load_state_dict(torch.load(args.ckpt, map_location=device))
    agent.eval()

    returns = []
    lengths = []
    illegal_count = 0
    total_actions = 0
    crashes = 0

    for ep in range(args.episodes):
        obs, info = env.reset()
        ep_ret = 0.0
        t = 0

        while True:
            obs_vec = obs["obs"]
            mask = obs["action_mask"]

            a = masked_greedy_action(agent, obs_vec, mask, device)

            # legality check at decision time
            total_actions += 1
            if mask[a] < 0.5:
                illegal_count += 1

            obs, r, term, trunc, info = env.step(a)
            ep_ret += float(r)
            t += 1

            if isinstance(info, dict) and info.get("env_crash", False):
                crashes += 1

            if term or trunc:
                break

        returns.append(ep_ret)
        lengths.append(t)

    returns = np.array(returns, dtype=np.float32)
    lengths = np.array(lengths, dtype=np.int32)

    print("=== EVAL RESULTS ===")
    print(f"episodes: {args.episodes}")
    print(f"return mean: {returns.mean():.4f}  median: {np.median(returns):.4f}  std: {returns.std():.4f}")
    print(f"len mean: {lengths.mean():.2f}  median: {np.median(lengths):.2f}")
    print(f"illegal actions: {illegal_count}/{total_actions} ({illegal_count/max(1,total_actions)*100:.3f}%)")
    print(f"env_crash episodes: {crashes}")

if __name__ == "__main__":
    main()
