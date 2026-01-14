from __future__ import annotations
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
from dataclasses import dataclass
import tyro

# ---------------------------------------------------------
# 1. Define Arguments Class (Top Level)
# ---------------------------------------------------------
@dataclass
class Args:
    # Experiment Settings
    num_envs: int = 1
    rollout_len: int = 256
    total_timesteps: int = 20_000
    
    # Hyperparameters
    learning_rate: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    update_epochs: int = 4
    num_minibatches: int = 4  
    
    # Coefficients & Clipping
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.rollout_len

    @property
    def minibatch_size(self) -> int:
        # Automatically scales!
        # 1 env * 256 steps / 4 = 64
        # 8 envs * 256 steps / 4 = 512
        return self.batch_size // self.num_minibatches

# ---------------------------------------------------------
# 2. Environment & Agent Setup
# ---------------------------------------------------------
from tft_rl.env.wrappers import make_position_single_step_env

def make_vec_env(num_envs: int):
    return gym.vector.SyncVectorEnv([lambda: make_position_single_step_env() for _ in range(num_envs)])

class Agent(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
        )
        self.pi = nn.Linear(256, n_actions)
        self.v  = nn.Linear(256, 1)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        h = self.shared(obs)
        return self.v(h).squeeze(-1)

    def get_action_and_value(self, obs: torch.Tensor, action_mask: torch.Tensor, action: torch.Tensor | None = None):
        h = self.shared(obs)
        logits = self.pi(h)

        # Apply action mask
        pass_idx = 0 
        legal_count = action_mask.sum(dim=1, keepdim=True)
        needs_fallback = (legal_count == 0)
        if needs_fallback.any():
            action_mask = action_mask.clone()
            action_mask[needs_fallback.squeeze(1), pass_idx] = 1.0

        neg_inf = torch.finfo(logits.dtype).min
        masked_logits = torch.where(action_mask > 0, logits, neg_inf)

        dist = Categorical(logits=masked_logits)
        if action is None:
            action = dist.sample()
        logprob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.v(h).squeeze(-1)
        return action, logprob, entropy, value

# ---------------------------------------------------------
# 3. Main Training Loop
# ---------------------------------------------------------
def main():
    # --- Parse Config ---
    args = tyro.cli(Args)
    
    run_name = f"ppo_masked_{int(time.time())}"
    
    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    # Save run config
    with open(os.path.join(ckpt_dir, f"{run_name}_config.txt"), "w", encoding="utf-8") as f:
        f.write(f"run_name={run_name}\n")
        f.write(f"num_envs={args.num_envs}\nrollout_len={args.rollout_len}\n")
        f.write(f"total_timesteps={args.total_timesteps}\n")
        f.write(f"lr={args.learning_rate}\ngamma={args.gamma}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(f"runs/{run_name}")

    envs = make_vec_env(args.num_envs)
    obs_space = envs.single_observation_space
    act_space = envs.single_action_space

    obs_dim = obs_space["obs"].shape[0]
    n_actions = act_space.n

    agent = Agent(obs_dim, n_actions).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # --- Storage ---
    obs_buf = torch.zeros((args.rollout_len, args.num_envs, obs_dim), device=device)
    mask_buf = torch.zeros((args.rollout_len, args.num_envs, n_actions), device=device)
    actions_buf = torch.zeros((args.rollout_len, args.num_envs), device=device, dtype=torch.long)
    logp_buf = torch.zeros((args.rollout_len, args.num_envs), device=device)
    rewards_buf = torch.zeros((args.rollout_len, args.num_envs), device=device)
    dones_buf = torch.zeros((args.rollout_len, args.num_envs), device=device)
    values_buf = torch.zeros((args.rollout_len, args.num_envs), device=device)

    # Reset envs
    o, _ = envs.reset(seed=0)
    next_obs = torch.tensor(o["obs"], device=device, dtype=torch.float32)
    next_mask = torch.tensor(o["action_mask"], device=device, dtype=torch.float32)
    next_done = torch.zeros(args.num_envs, device=device)

    global_step = 0
    start_time = time.time()

    while global_step < args.total_timesteps:
        # --- Rollout collection ---
        for t in range(args.rollout_len):
            global_step += args.num_envs
            obs_buf[t] = next_obs
            mask_buf[t] = next_mask
            dones_buf[t] = next_done

            with torch.no_grad():
                action, logp, entropy, value = agent.get_action_and_value(next_obs, next_mask)
            actions_buf[t] = action
            logp_buf[t] = logp
            values_buf[t] = value

            # Step envs
            a_np = action.cpu().numpy()
            o2, r, term, trunc, info = envs.step(a_np)
            done = np.logical_or(term, trunc)
            
            # Handle Auto-Reset for Vector Envs
            if done.any():
                for i, d in enumerate(done):
                    if d:
                        o_reset, _ = envs.envs[i].reset()
                        o2["obs"][i] = o_reset["obs"]
                        o2["action_mask"][i] = o_reset["action_mask"]
                        done[i] = False

            rewards_buf[t] = torch.tensor(r, device=device, dtype=torch.float32)
            next_obs = torch.tensor(o2["obs"], device=device, dtype=torch.float32)
            next_mask = torch.tensor(o2["action_mask"], device=device, dtype=torch.float32)
            next_done = torch.tensor(done, device=device, dtype=torch.float32)

        # --- Compute GAE advantages ---
        with torch.no_grad():
            next_value = agent.get_value(next_obs)

        advantages = torch.zeros_like(rewards_buf, device=device)
        lastgaelam = torch.zeros(args.num_envs, device=device)
        for t in reversed(range(args.rollout_len)):
            if t == args.rollout_len - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones_buf[t + 1]
                nextvalues = values_buf[t + 1]
            delta = rewards_buf[t] + args.gamma * nextvalues * nextnonterminal - values_buf[t]
            lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            advantages[t] = lastgaelam
        returns = advantages + values_buf

        # Flatten batch
        b_obs = obs_buf.reshape(-1, obs_dim)
        b_mask = mask_buf.reshape(-1, n_actions)
        b_actions = actions_buf.reshape(-1)
        b_logp = logp_buf.reshape(-1)
        b_adv = advantages.reshape(-1)
        b_ret = returns.reshape(-1)
        b_val = values_buf.reshape(-1)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        # --- PPO update ---
        batch_size = args.rollout_len * args.num_envs
        idxs = np.arange(batch_size)

        for epoch in range(args.update_epochs):
            np.random.shuffle(idxs)
            for start in range(0, batch_size, args.minibatch_size):
                mb_idx = idxs[start:start + args.minibatch_size]

                _, newlogp, entropy, newval = agent.get_action_and_value(
                    b_obs[mb_idx], b_mask[mb_idx], b_actions[mb_idx]
                )
                logratio = newlogp - b_logp[mb_idx]
                ratio = logratio.exp()

                pg_loss1 = -b_adv[mb_idx] * ratio
                pg_loss2 = -b_adv[mb_idx] * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * (b_ret[mb_idx] - newval).pow(2).mean()
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        # Logging
        fps = int(global_step / (time.time() - start_time))
        writer.add_scalar("charts/fps", fps, global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("charts/mean_reward_rollout", rewards_buf.mean().item(), global_step)

        if global_step % 20_000 < args.num_envs:
            torch.save(agent.state_dict(), os.path.join(ckpt_dir, "ppo_masked_latest.pt"))

    torch.save(agent.state_dict(), os.path.join(ckpt_dir, "ppo_masked_latest.pt"))
    print(f"Saved checkpoint: {os.path.join(ckpt_dir, 'ppo_masked_latest.pt')}")


    envs.close()
    writer.close()

if __name__ == "__main__":
    main()