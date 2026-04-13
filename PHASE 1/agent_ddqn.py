"""
agent_template.py  –  OBELIX Submission Agent
==============================================
Submission file.  Must define:  policy(obs, rng) -> str

This file packages the trained Dueling DDQN weights and all inference
logic so that evaluate.py can call policy() with zero external dependencies
beyond numpy and torch.

Architecture recap (mirrors train_ddqn.py):
  Input  : 4 stacked frames (18×4=72) + 5 last-action one-hot + 1 rotation-streak
           = 78-dimensional state vector
  Network: Dueling DDQN (body→value+advantage)
  Output : argmax Q(s,a) over 5 discrete actions

The agent additionally runs a lightweight FSM guard:
  • If the model has been choosing rotations for MAX_PURE_ROT consecutive
    steps with NO sensor bit active → inject a forward step to break the loop.
    This is a safety net on top of the trained policy and handles rare
    out-of-distribution states.

Usage:
  python evaluate.py --agent_file agent_template.py --runs 10 --difficulty 0
"""

import os
import numpy as np
import torch
import torch.nn as nn

# ─────────────────────────────────────────────────────────────────────────────
# Constants (must match train_ddqn.py exactly)
# ─────────────────────────────────────────────────────────────────────────────
ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS = len(ACTIONS)
ROTATION_ACTIONS = {0, 1, 3, 4}
FRAME_STACK = 4
OBS_DIM = 18
STACK_DIM = FRAME_STACK * OBS_DIM     # 72
AUG_DIM   = STACK_DIM + N_ACTIONS + 1  # 78

MAX_PURE_ROT = 12  # safety FSM: max consecutive rotations w/o sensor activity

# ─────────────────────────────────────────────────────────────────────────────
# Network (identical to train_ddqn.py)
# ─────────────────────────────────────────────────────────────────────────────
class DuelingDDQN(nn.Module):
    def __init__(self, state_dim: int, n_actions: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256),       nn.ReLU(),
        )
        self.value = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.advantage = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.body(x)
        v = self.value(h)
        a = self.advantage(h)
        return v + (a - a.mean(dim=1, keepdim=True))


# ─────────────────────────────────────────────────────────────────────────────
# Lazy model loading  (loads once on first call)
# ─────────────────────────────────────────────────────────────────────────────
_model: DuelingDDQN = None

def _load_model():
    global _model
    if _model is not None:
        return _model

    # Look for weights.pth next to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(script_dir, "weights_withwall.pth")

    if not os.path.exists(weights_path):
        print(f"Error: weights_ddqn.pth not found at {weights_path}")
        raise FileNotFoundError(
            f"weights.pth not found at {weights_path}\n"
            "Run train_ddqn.py first to generate the weights file."
        )
    else:    print(f"Loading model weights from {weights_path}...")

    ckpt = torch.load(weights_path, map_location="cpu")
    state_dim = ckpt.get("aug_dim", AUG_DIM)
    n_act     = ckpt.get("n_actions", N_ACTIONS)

    _model = DuelingDDQN(state_dim, n_act)
    _model.load_state_dict(ckpt["online_state_dict"])
    _model.eval()
    return _model


# ─────────────────────────────────────────────────────────────────────────────
# Per-episode state  (persists across steps within an episode)
# ─────────────────────────────────────────────────────────────────────────────
from collections import deque

_frames         = deque(maxlen=FRAME_STACK)
_last_action_idx = 2      # FW
_rot_streak      = 0      # consecutive rotations
_no_sensor_streak = 0     # steps with all sensors silent
_pure_rot_streak  = 0     # rotations without any sensor bit (loop detector)
_in_push         = False  # did we attach to the box?
_prev_was_push   = False  # detect episode boundary


def _reset_state(obs: np.ndarray):
    """Re-initialise episode state from the first observation."""
    global _frames, _last_action_idx, _rot_streak
    global _no_sensor_streak, _pure_rot_streak, _in_push, _prev_was_push
    _frames.clear()
    for _ in range(FRAME_STACK):
        _frames.append(obs.copy())
    _last_action_idx  = 2
    _rot_streak       = 0
    _no_sensor_streak = 0
    _pure_rot_streak  = 0
    _in_push          = False
    _prev_was_push    = False


def _build_state() -> np.ndarray:
    stacked = np.concatenate(list(_frames)).astype(np.float32)
    act_oh  = np.zeros(N_ACTIONS, dtype=np.float32)
    act_oh[_last_action_idx] = 1.0
    rot_feat = np.array([min(_rot_streak, 20) / 20.0], dtype=np.float32)
    return np.concatenate([stacked, act_oh, rot_feat])


# ─────────────────────────────────────────────────────────────────────────────
# Episode boundary detection
# ─────────────────────────────────────────────────────────────────────────────
_call_count = 0  # global call counter across all policy() invocations
_prev_obs   = None

def _detect_new_episode(obs: np.ndarray) -> bool:
    """
    Heuristic: a new episode started if:
      - First ever call, OR
      - We were in push mode AND all sensors just went silent for 3+ steps
        (post-success or max_steps reset)
    """
    global _in_push, _no_sensor_streak
    if _prev_obs is None:
        return True
    any_now  = bool(np.any(obs[:17]))
    any_prev = bool(np.any(_prev_obs[:17]))
    if _in_push and not any_now and not any_prev:
        _no_sensor_streak += 1
        if _no_sensor_streak >= 3:
            return True
    else:
        _no_sensor_streak = 0
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────
def policy(obs, rng=None):
    """
    Required entry point called by evaluate.py.

    Parameters
    ----------
    obs : array-like, shape (18,)
    rng : np.random.Generator | None

    Returns
    -------
    str  –  one of "L45", "L22", "FW", "R22", "R45"
    """
    global _last_action_idx, _rot_streak, _pure_rot_streak
    global _in_push, _prev_was_push, _prev_obs, _call_count

    obs = np.asarray(obs, dtype=np.float32)
    model = _load_model()

    # ── Episode boundary detection ─────────────────────────────────────────
    _call_count += 1
    if _detect_new_episode(obs):
        _reset_state(obs)
    _prev_obs = obs.copy()

    # ── Update frame stack ─────────────────────────────────────────────────
    _frames.append(obs.copy())

    # ── Detect push state (attached to box) ────────────────────────────────
    ir_active = bool(obs[16])
    if ir_active:
        _in_push = True

    # ── Build augmented state ──────────────────────────────────────────────
    state = _build_state()

    # ── Safety FSM: rotation loop guard ───────────────────────────────────
    # If the trained model has been spinning with ZERO sensor activity for
    # MAX_PURE_ROT steps → force one forward step to break the deadlock.
    any_sensor = bool(np.any(obs[:17]))
    if _last_action_idx in ROTATION_ACTIONS and not any_sensor:
        _pure_rot_streak += 1
    else:
        _pure_rot_streak = 0

    if _pure_rot_streak >= MAX_PURE_ROT:
        _pure_rot_streak = 0
        action_idx = 2  # FW – probe whether the robot is facing open space
    else:
        # ── Neural network decision ────────────────────────────────────────
        with torch.no_grad():
            s_t = torch.FloatTensor(state).unsqueeze(0)
            q   = model(s_t)
            action_idx = int(q.argmax(dim=1).item())

    # ── Update rotation streak ─────────────────────────────────────────────
    if action_idx in ROTATION_ACTIONS:
        _rot_streak = min(_rot_streak + 1, 20)
    else:
        _rot_streak = 0

    _last_action_idx = action_idx
    return ACTIONS[action_idx]