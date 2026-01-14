# Project Status — TFT RL

## Current Version
- **Version:** v0.1-position-only
- **Date:** 2026-01-13
- **Git Commit:** <PASTE_COMMIT_HASH_HERE>

## Environment
- TFT single-player simulator (Set 4)
- Position-only action space
- Discrete masked PPO
- No illegal actions permitted

## Training Setup
- Algorithm: PPO (masked actions)
- num_envs: 8
- rollout_len: 256
- total_timesteps: ~200k
- Action masking enforced at policy level
- Observation flattened + action_mask provided

## Training Stability
- ✅ 0 environment crashes
- ✅ 0 illegal actions during training
- FPS stable after disabling simulator debug prints

## Evaluation Results (Latest)
episodes: 500
return mean: 5.516
return median: 2.500
return std: 4.742
episode length mean: 98.77
episode length median: 53.50
illegal actions: 0%
env crashes: 0


## Observations
- Return distribution is heavy-tailed (mean >> median)
- Policy occasionally achieves strong outcomes but is inconsistent
- Entropy decreasing smoothly (expected PPO behavior)
- Policy loss oscillatory but stable
- Value loss trending down

## Known Simulator Issues
- Occasional debug prints from simulator (e.g. Yuumi ability edge case)
- Some logs reference “shop slot empty” / no-op moves — believed benign
- Simulator may report internal “levels” > TFT max player level (likely battle difficulty)

## Current Assessment
- Environment + wrappers are stable
- PPO learning signal is real but early
- This version is suitable as a **baseline checkpoint**

## Next Steps (Planned)
1. Diagnose heavy-tailed returns (variance reduction or curriculum)
2. Add richer evaluation metrics (placement proxy, survival curve)
3. Consider extending action space (items or limited shop actions)
4. Scale training (longer runs, more seeds)
