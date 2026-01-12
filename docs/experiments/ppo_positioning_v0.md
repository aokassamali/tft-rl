# PPO Positioning v0

Date: 2026-01-11  
Env: TFT_Single_Player_Simulator  
Action Space: Discrete(2090) with mask  
Obs: Flattened (348-dim) + action_mask  

## Training
- Algo: PPO (masked)
- num_envs: 1
- Steps: ~20k
- Reward: sparse, round-based
- No reward shaping
- No invalid action penalty

## Observations
- Training stable (no crashes / NaNs)
- Entropy decreases over time
- Policy and value losses stable
- Mean reward noisy but non-zero
- Episodes terminate correctly

TensorBoard not committed (gitignored).
