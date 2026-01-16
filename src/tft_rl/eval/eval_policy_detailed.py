"""
Evaluate a trained masked PPO policy with richer per-episode logging.

Why this exists:
- Our mean return >> median return suggests a heavy tail.
- This script helps diagnose *why* by recording episode-level and step-level aggregates.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import tyro

from tft_rl.env.wrappers import make_position_single_step_env


# --- Agent must match training exactly ---
class Agent(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
        )
        self.pi = nn.Linear(256, n_actions)
        self.v = nn.Linear(256, 1)

    def forward_logits(self, obs: torch.Tensor) -> torch.Tensor:
        h = self.shared(obs)
        return self.pi(h)

    def get_action(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor,
        policy: Literal["trained_sample", "trained_greedy", "random_legal", "always_pass"] = "trained_sample",
    ) -> torch.Tensor:
        logits = self.forward_logits(obs)

        # Fallback: if mask is all zeros, allow pass action (index 0)
        pass_idx = 0
        legal_count = action_mask.sum(dim=1, keepdim=True)
        needs_fallback = (legal_count == 0)
        if needs_fallback.any():
            action_mask = action_mask.clone()
            action_mask[needs_fallback.squeeze(1), pass_idx] = 1.0

        neg_inf = torch.finfo(logits.dtype).min
        masked_logits = torch.where(action_mask > 0, logits, neg_inf)

        if policy == "trained_greedy":
            return torch.argmax(masked_logits, dim=1)

        dist = Categorical(logits=masked_logits)
        return dist.sample()


@dataclass
class Args:
    ckpt: str = "checkpoints/ppo_masked_latest.pt"
    episodes: int = 500
    seed: int = 0
    policy: Literal["trained_sample", "trained_greedy", "random_legal", "always_pass"] = "trained_sample"
    out_dir: str = "results/eval_runs"
    log_events: bool = True
    max_events: int = 200


def percentile_dict(x: np.ndarray, ps=(50, 75, 90, 95, 99)) -> dict:
    return {f"p{p}": float(np.percentile(x, p)) for p in ps}


def main():
    args = tyro.cli(Args)

    os.makedirs(args.out_dir, exist_ok=True)
    run_name = f"eval_{args.policy}_{int(time.time())}"
    out_path = os.path.join(args.out_dir, f"{run_name}.jsonl")

    # Repro
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build env (exactly like training)
    env = make_position_single_step_env()

    # Infer sizes
    obs_space = env.observation_space
    act_space = env.action_space
    obs_dim = obs_space["obs"].shape[0]
    n_actions = act_space.n

    # Load agent
    agent = Agent(obs_dim, n_actions).to(device)
    if args.policy in ["trained_sample", "trained_greedy"]:
        try:
            state = torch.load(args.ckpt, map_location=device)
            agent.load_state_dict(state)
            print(f"Loaded checkpoint from {args.ckpt}")
        except FileNotFoundError:
            print(f"Warning: Checkpoint {args.ckpt} not found! Running with initialized weights.")
    
    agent.eval()

    returns = []
    lengths = []
    final_rounds = []
    legal_means = []
    legal_zeros = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for ep in range(args.episodes):
            obs, info = env.reset(seed=args.seed + ep)

            ep_ret = 0.0
            ep_len = 0
            ep_round = None
            legal_counts = []
            events = []
            nonzero_reward_steps = 0
            round_change_steps = 0
            prev_round = None

            while True:
                # Expect obs is a dict: {"obs": (obs_dim,), "action_mask": (n_actions,)}
                obs_vec = torch.tensor(obs["obs"], device=device, dtype=torch.float32).unsqueeze(0)
                mask = torch.tensor(obs["action_mask"], device=device, dtype=torch.float32).unsqueeze(0)

                lc = float(mask.sum().item())
                legal_counts.append(lc)
                if lc == 0:
                    legal_zeros += 1

                mask_1d = mask.squeeze(0)  # (n_actions,)

                # --- Action Logic ---
                if args.policy == "always_pass":
                    a_np = 0

                elif args.policy == "random_legal":
                    legal = torch.nonzero(mask_1d > 0, as_tuple=False).squeeze(1)
                    if legal.numel() == 0:
                        a_np = 0
                    else:
                        idx = torch.randint(low=0, high=legal.numel(), size=(1,), device=device).item()
                        a_np = int(legal[idx].item())

                else:
                    # Trained policies (sample or greedy)
                    with torch.no_grad():
                        # FIX: pass 'policy' arg, not 'mode', and use args.policy directly
                        a = agent.get_action(obs_vec, mask, policy=args.policy)
                    a_np = int(a.item())

                # --- Step Env (FIX: This is now unindented so it runs for ALL policies) ---
                obs, r, term, trunc, step_info = env.step(a_np)

                ep_ret += float(r)
                ep_len += 1

                done = bool(term or trunc)

                # Pull round (if available) each step
                cur_round = None
                if isinstance(step_info, dict) and "game_round" in step_info:
                    cur_round = step_info["game_round"]

                # Track reward sparsity
                r_f = float(r)
                if r_f != 0.0:
                    nonzero_reward_steps += 1

                # Track round transitions
                if cur_round is not None:
                    if prev_round is None:
                        prev_round = cur_round
                    elif cur_round != prev_round:
                        round_change_steps += 1
                        prev_round = cur_round

                # Log only "interesting" events to keep output small
                if args.log_events:
                    interesting = (r_f != 0.0) or done or (cur_round is not None and ep_round is not None and cur_round != ep_round) or (ep_round is None and cur_round is not None)
                    # The condition above catches: reward events, termination, and first/changed round observations.

                    if interesting:
                        events.append({
                            "t": ep_len,          # 1-indexed step count
                            "a": int(a_np),
                            "r": r_f,
                            "done": done,
                            "round": None if cur_round is None else int(cur_round),
                            "legal_count": lc,
                        })
                        if len(events) > getattr(args, "max_events", 10**9):
                            # cap (optional)
                            events = events[:args.max_events]

                # info sometimes includes "game_round"
                if cur_round is not None:
                    ep_round = cur_round

                if term or trunc:
                    break

            ep_legal_mean = float(np.mean(legal_counts)) if legal_counts else 0.0

            row = {
                "episode": ep,
                "policy": args.policy,
                "return": ep_ret,
                "length": ep_len,
                "final_round": ep_round,
                "legal_mean": ep_legal_mean,

                # --- Probe stats ---
                "nonzero_reward_steps": nonzero_reward_steps,
                "round_change_steps": round_change_steps,
            }

            if args.log_events:
                row["events"] = events
                
            f.write(json.dumps(row) + "\n")

            returns.append(ep_ret)
            lengths.append(ep_len)
            final_rounds.append(-1 if ep_round is None else int(ep_round))
            legal_means.append(ep_legal_mean)

    returns = np.array(returns, dtype=np.float32)
    lengths = np.array(lengths, dtype=np.float32)
    final_rounds = np.array(final_rounds, dtype=np.float32)
    legal_means = np.array(legal_means, dtype=np.float32)

    print("\n=== EVAL (DETAILED) ===")
    print(f"ckpt: {args.ckpt}")
    print(f"policy: {args.policy}")
    print(f"episodes: {args.episodes}")
    print(f"saved: {out_path}\n")

    print(f"return mean={returns.mean():.4f} median={np.median(returns):.4f} std={returns.std():.4f}")
    print("return percentiles:", percentile_dict(returns))
    print(f"len mean={lengths.mean():.2f} median={np.median(lengths):.2f}")
    print("len percentiles:", percentile_dict(lengths))

    # Correlations (quick diagnostic)
    def corr(a, b):
        if np.std(a) < 1e-8 or np.std(b) < 1e-8:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    print(f"\ncorr(return, length)={corr(returns, lengths):.3f}")
    print(f"corr(return, final_round)={corr(returns, final_rounds):.3f}")
    print(f"legal_mean avg={legal_means.mean():.2f} (lower means fewer choices / tighter mask)")
    print(f"legal_count == 0 occurrences: {legal_zeros}")

    env.close()


if __name__ == "__main__":
    main()