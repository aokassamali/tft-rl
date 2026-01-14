## Patch: Silence simulator debug prints

Reason:
The TFTMuZeroAgent simulator prints messages such as
"Shop slot is empty" and "No champion to move" during training.
These are informational no-op messages and cause severe stdout spam,
reducing FPS and obscuring real errors during RL training.

Change:
- Commented out print statements in:
  Simulator/player.py

Logic impact:
- None (prints only)
- Action legality, state transitions unchanged

Status:
- Safe
- Reversible
