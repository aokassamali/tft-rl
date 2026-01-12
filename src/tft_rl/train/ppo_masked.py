# Goal: train PPO on TFT positioning env with action masking.
# Why: mask prevents illegal actions; PPO learns from delayed battle rewards.

from __future__ import annotations
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym

from tft_rl.env.wrappers import make_position_single_step_env

def make_vec_env(num_envs: int):
    # Simple SyncVectorEnv; later we can switch to AsyncVectorEnv if needed.
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

        # Apply action mask: set illegal logits to a large negative number.
        # Mask is 1 for legal actions, 0 for illegal.
        # Also ensure at least one legal action (pass) in pathological cases.
        pass_idx = 0  # grid row=0 col=0 -> pass in our decoder/wrapper
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

def main():
    # --- Config ---
    run_name = f"ppo_masked_{int(time.time())}"
    num_envs = 1
    rollout_len = 256
    total_timesteps = 20_000
    learning_rate = 2.5e-4
    gamma = 0.99
    gae_lambda = 0.95
    update_epochs = 4
    minibatch_size = 1024
    clip_coef = 0.2
    ent_coef = 0.01
    vf_coef = 0.5
    max_grad_norm = 0.5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(f"runs/{run_name}")

    envs = make_vec_env(num_envs)
    obs_space = envs.single_observation_space
    act_space = envs.single_action_space

    obs_dim = obs_space["obs"].shape[0]
    n_actions = act_space.n

    agent = Agent(obs_dim, n_actions).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)

    # --- Storage ---
    obs_buf = torch.zeros((rollout_len, num_envs, obs_dim), device=device)
    mask_buf = torch.zeros((rollout_len, num_envs, n_actions), device=device)
    actions_buf = torch.zeros((rollout_len, num_envs), device=device, dtype=torch.long)
    logp_buf = torch.zeros((rollout_len, num_envs), device=device)
    rewards_buf = torch.zeros((rollout_len, num_envs), device=device)
    dones_buf = torch.zeros((rollout_len, num_envs), device=device)
    values_buf = torch.zeros((rollout_len, num_envs), device=device)

    # Reset envs
    o, _ = envs.reset(seed=0)
    next_obs = torch.tensor(o["obs"], device=device, dtype=torch.float32)
    next_mask = torch.tensor(o["action_mask"], device=device, dtype=torch.float32)
    next_done = torch.zeros(num_envs, device=device)

    global_step = 0
    start_time = time.time()

    while global_step < total_timesteps:
        # --- Rollout collection ---
        for t in range(rollout_len):
            global_step += num_envs
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
            if done.any():
                # reset only the done envs
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
        lastgaelam = torch.zeros(num_envs, device=device)
        for t in reversed(range(rollout_len)):
            if t == rollout_len - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones_buf[t + 1]
                nextvalues = values_buf[t + 1]
            delta = rewards_buf[t] + gamma * nextvalues * nextnonterminal - values_buf[t]
            lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
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
        batch_size = rollout_len * num_envs
        idxs = np.arange(batch_size)

        for epoch in range(update_epochs):
            np.random.shuffle(idxs)
            for start in range(0, batch_size, minibatch_size):
                mb_idx = idxs[start:start + minibatch_size]

                _, newlogp, entropy, newval = agent.get_action_and_value(
                    b_obs[mb_idx], b_mask[mb_idx], b_actions[mb_idx]
                )
                logratio = newlogp - b_logp[mb_idx]
                ratio = logratio.exp()

                # Policy loss
                pg_loss1 = -b_adv[mb_idx] * ratio
                pg_loss2 = -b_adv[mb_idx] * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                v_loss = 0.5 * (b_ret[mb_idx] - newval).pow(2).mean()

                entropy_loss = entropy.mean()

                loss = pg_loss - ent_coef * entropy_loss + vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

        # Logging
        fps = int(global_step / (time.time() - start_time))
        writer.add_scalar("charts/fps", fps, global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("charts/mean_reward_rollout", rewards_buf.mean().item(), global_step)

    envs.close()
    writer.close()

if __name__ == "__main__":
    main()
