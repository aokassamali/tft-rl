# Goal: construct TFT environments behind a stable interface, even if the upstream signatures differ.
# Why: third-party research repos often have inconsistent APIs; we normalize them here.

from __future__ import annotations
from pathlib import Path
import sys
import inspect

from tft_rl.env.configs import make_default_tft_config
from tft_rl.env.gym_wrappers import DeltaRewardWrapper
from tft_rl.env.action_wrappers import GridDiscreteActionWrapper
from tft_rl.env.safety_wrappers import InvalidActionPenaltyWrapper, CrashShieldWrapper
from tft_rl.env.obs_wrappers import FlattenObsWrapper

def _ensure_submodule_on_path():
    repo_root = Path(__file__).resolve().parents[3]
    submodule_root = repo_root / "third_party" / "tftmuzeroagent"
    if str(submodule_root) not in sys.path:
        sys.path.insert(0, str(submodule_root))

def make_position_single_step_env():
    _ensure_submodule_on_path()
    from Simulator.tft_single_player_simulator import TFT_Single_Player_Simulator
    tft_config = make_default_tft_config()
    
    env = TFT_Single_Player_Simulator(tft_config=tft_config)
    env = CrashShieldWrapper(env, crash_reward=-1.0)
    env = DeltaRewardWrapper(env)
    env = GridDiscreteActionWrapper(env)
    #env = InvalidActionPenaltyWrapper(env, penalty=-0.01)
    env = FlattenObsWrapper(env)
    return env

def adapt_position_action(action):
    """
    TFTMuZeroAgent's positioning single-player simulator currently expects an action list of length 3,
    even though action_space advertises MultiDiscrete([55, 38]) (length 2).
    We adapt sampled actions into the expected format.

    NOTE: We are setting the 3rd component to 0 until we confirm its semantic meaning.
    """
    # action may be np.ndarray shape (2,)
    try:
        a0 = int(action[0])
        a1 = int(action[1])
        return [a0, a1, 0]
    except Exception:
        # already a list?
        if isinstance(action, list) and len(action) == 3:
            return action
        raise

def make_full_game_env():
    _ensure_submodule_on_path()
    from Simulator.tft_simulator import TFT_Simulator
    tft_config = make_default_tft_config()

    # Robust construction: try positional config, then no-arg.
    sig = inspect.signature(TFT_Simulator)
    params = sig.parameters

    if len(params) >= 1:
        # Many classes show (self, config) in signature; inspect.signature includes 'self' only for functions,
        # but for classes it shows __init__ params. We'll just try positional.
        try:
            return TFT_Simulator(tft_config)
        except TypeError:
            pass

    return TFT_Simulator()
