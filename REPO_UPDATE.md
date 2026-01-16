# Repository Update — TFT RL Project (v0 → v1 Pivot)

**Date:** 2026-01-xx  
**Status:** v0 frozen, new simulator planned

---

## Summary

This update documents the current state of the project, what has been successfully achieved so far, and a deliberate pivot in direction based on technical and research findings.

The short version: **the RL pipeline works; the simulator does not scale**. Rather than continuing to wrap and patch a brittle simulator, I will focus on building a new, RL-native TFT simulator from first principles.

---

## What Has Been Accomplished (v0)

Over the past phase, I built and validated a complete end-to-end RL training loop on top of an existing TFT simulator originally designed for MuZero-style tree search.

Key achievements:

- Implemented a **fully masked discrete action space** (~2,000 actions)
- Achieved **0 illegal actions** during training and evaluation
- Stable PPO training with:
  - GAE
  - entropy regularization
  - masked categorical policies
- Robust evaluation pipeline with:
  - multiple baseline policies (random legal, always pass, greedy)
  - detailed per-episode logs
  - no environment crashes across tens of thousands of steps
- Demonstrated that the trained policy **outperforms random and trivial baselines**

This establishes that:
- the PPO implementation is correct
- action masking works as intended
- the training + evaluation harness is sound

---

## Key Finding

Despite stable training and improved returns, evaluation revealed a critical issue:

corr(return, episode_length) = 1.0
corr(return, final_round) = 1.0


This indicates the policy is primarily learning **how to survive longer**, not which specific actions improve outcomes *within* a round.

In other words, the current reward signal is **too coarse** to distinguish:
- good positioning vs
- harmless stalling behavior

This limits the agent’s ability to learn meaningful tactical decisions.

---

## Simulator Diagnosis

While debugging and extending the environment, a deeper architectural issue became clear.

The current simulator:

- Uses deeply nested, mutable Python objects  
  (`Player → Board → Champion → Items`)
- Relies heavily on **in-place mutation**
- Was designed for **tree search (MuZero)**, not rollout-based RL
- Makes clean resets difficult without expensive deep copies
- Runs battles via Python loops, making it too slow for scalable PPO

This leads to **state entanglement**:
- state leaks across steps and resets
- action legality depends on fragile internal object references
- most development effort goes into defensive wrappers and bug isolation

Effectively, the simulator is a **stateful object engine**, while PPO expects a **functional input/output box**.

---

## Decision: Freeze v0 and Pivot

The current environment is being **frozen as a v0 baseline**.

Rather than continuing to layer wrappers on top of a simulator with incompatible assumptions, the next phase will focus on:

> Building a new TFT simulator explicitly designed for RL.

This preserves all progress made so far while avoiding long-term technical debt.

---

## Next Direction (v1)

The new simulator will be:

- RL-native and rollout-friendly
- Stateless or functionally structured where possible
- Fast enough for large-scale PPO rollouts
- Built bottom-up, starting with **positioning as an isolated subtask**
- Driven by interpretable rewards (e.g. health delta per round)

The long-term goal is a modular pipeline:
1. Positioning
2. Economy
3. Itemization
4. Full end-to-end TFT gameplay

Each component should be learnable and testable in isolation.

---

## Repository Plan

- This repository remains as a **v0 reference implementation**
  - Demonstrates PPO + masking + evaluation
  - Tagged as a stable baseline
  - A **new repository** will be created for the v1 simulator
  - Clean architecture
  - Clean commit history
  - Research-first design

---

## Closing Note

It's unfortunate that the current existing env doesn't work for my goals and needs, but I'm definitely taking away from it how to approach building a logic based sim for TFT