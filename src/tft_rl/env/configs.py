# Goal: central place to construct TFTConfig objects reproducibly.
# Why: configs define the "rules of the world" and must be explicit for research.

from __future__ import annotations
from pathlib import Path
import sys

def _ensure_submodule_on_path():
    repo_root = Path(__file__).resolve().parents[3]
    submodule_root = repo_root / "third_party" / "tftmuzeroagent"
    if str(submodule_root) not in sys.path:
        sys.path.insert(0, str(submodule_root))

def make_default_tft_config():
    """
    Create the default TFTConfig for the regular (Set 4-style) simulator.

    This config defines:
    - action space
    - max actions per round
    - observation/action tokenization
    - core game rules
    """
    _ensure_submodule_on_path()
    from Simulator.tft_simulator import TFTConfig

    # TFTConfig uses internal defaults if no args are passed
    config = TFTConfig()
    return config
