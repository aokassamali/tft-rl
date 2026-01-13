# tft-rl
# RL for TFT (WIP)

> **Status:** early WIP / “v0 baseline”  
> **Goal:** train an RL agent to play **Teamfight Tactics** end-to-end (or a simplified slice first), with a focus on *research exploration* and a reproducible training/eval loop.

---

## Why this project exists

TFT is a nice stress-test for modern RL because it’s:
- **Long-horizon** (many decisions before payoff)
- **Partially observed** + stochastic (shop RNG, combat variance)
- **Hybrid action space** (discrete choices, structured constraints, action masking)
- **Non-stationary** across patches / metas

The aim is to build a clean “RL substrate” (env → policy → eval) that makes progress measurable and debuggable.

---

## What’s implemented right now (v0)

- [ ] Environment runs end-to-end without crashing for N episodes
- [x] Basic training loop (PPO baseline) producing TensorBoard curves
- [x] Basic wrappers/action masking plumbing (still rough)
- [ ] Deterministic seeds + config logging (in progress)
- [ ] A minimal eval script that compares to simple heuristics (in progress)

**What “v0” means:** the pipeline runs, logs, and produces *some* learning signal—even if the policy is bad.

---