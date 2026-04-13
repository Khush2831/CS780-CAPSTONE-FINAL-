"""
================================================================
agent_phase3_v2.py  –  OBELIX Phase-3 Submission Policy (v2)
================================================================
Compatible with weights trained by train_phase3_resume.py.
STATE_DIM=190 — same as v1, weight-compatible.

Key fixes over agent_phase3.py:
  1. Hard FW mask when obs[17]=1 — never press into wall
  2. Anti-oscillation override CONDITIONED on not-stuck
     (old code forced FW even when stuck against wall — catastrophic)
  3. consec_stuck passed to build_state for feature[5]
  4. Gap-seek override: when stuck ≥ 5 steps near wall, use
     DR-based gap navigation to slide toward gap at y=250

Place next to weights_phase3_v2_best.pth when submitting.
================================================================
"""

from __future__ import annotations
import math, os
from typing import Optional
import numpy as np

# ── Constants (must match training exactly) ──────────────────
ACTIONS     = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS   = 5
OBS_DIM     = 18
ACT_DIM     = 5
EXTRA_DIM   = 6
FRAME_STACK = 8
PER_STEP    = OBS_DIM + ACT_DIM    # 23
STACK_DIM   = FRAME_STACK * PER_STEP  # 184
STATE_DIM   = STACK_DIM + EXTRA_DIM   # 190

ROT_DELTA   = {"L45": 45.0, "L22": 22.5, "FW": 0.0,
               "R22": -22.5, "R45": -45.0}
FW_STEP_PX  = 5.0
ARENA_SIZE  = 500
RESET_EVERY = 1510   # max_steps=1500 + small buffer

WALL_X      = 250.0
GAP_Y       = 250.0
GAP_HALF    = 65.0
NEAR_WALL   = 60.0   # x within WALL_X ± this → near wall


# ─────────────────────────────────────────────────────────────
# DEAD RECKONING
# ─────────────────────────────────────────────────────────────
class DeadReckoning:
    def __init__(self, x=250., y=250., angle=0., arena=ARENA_SIZE):
        self.x = x; self.y = y; self.angle = angle
        self.arena = arena; self._lo = 40.; self._hi = float(arena) - 40.

    def reset(self, x=250., y=250., angle=0.):
        self.x, self.y, self.angle = float(x), float(y), float(angle)

    def update(self, action_str: str):
        self.angle += ROT_DELTA[action_str]
        if action_str == "FW":
            rad    = math.radians(self.angle)
            self.x = float(np.clip(self.x + FW_STEP_PX * math.cos(rad), self._lo, self._hi))
            self.y = float(np.clip(self.y + FW_STEP_PX * math.sin(rad), self._lo, self._hi))

    def features(self, dark_steps: int, push_inferred: bool,
                 consec_stuck: int = 0) -> np.ndarray:
        rad       = math.radians(self.angle)
        lo, hi    = 10., float(self.arena) - 10.
        min_bound = min(self.x - lo, self.y - lo, hi - self.x, hi - self.y)
        min_bound = float(np.clip(min_bound / (self.arena * 0.5), 0., 1.))
        stuck_sig = float(np.clip(consec_stuck / 20.0, 0., 1.))
        return np.array([
            math.sin(rad), math.cos(rad),
            float(np.clip(dark_steps / FRAME_STACK, 0., 1.)),
            float(push_inferred), min_bound, stuck_sig,
        ], dtype=np.float32)

    def near_wall(self) -> bool:
        return abs(self.x - WALL_X) < NEAR_WALL

    def gap_seek_action(self) -> str:
        """Navigate toward gap at y=GAP_Y when stuck at wall."""
        dy = GAP_Y - self.y
        if abs(dy) < GAP_HALF:
            return "FW"   # In gap zone — try to go through
        target_angle = 270.0 if dy > 0 else 90.0
        current      = self.angle % 360.0
        diff         = (target_angle - current + 360.0) % 360.0
        if diff <= 22.5 or diff >= 337.5: return "FW"
        elif diff <= 67.5:  return "L22"
        elif diff <= 112.5: return "L45"
        elif diff <= 247.5: return "R45"
        elif diff <= 292.5: return "R45"
        else:               return "R22"


# ─────────────────────────────────────────────────────────────
# FRAME STACK
# ─────────────────────────────────────────────────────────────
class FrameStack:
    def __init__(self, k=FRAME_STACK):
        self.k = k; self.buf = np.zeros((k, PER_STEP), dtype=np.float32)

    def reset(self, obs: np.ndarray, action_idx: int = 2):
        oh = np.zeros(ACT_DIM, dtype=np.float32); oh[action_idx] = 1.
        f  = np.concatenate([obs.astype(np.float32), oh])
        for i in range(self.k): self.buf[i] = f

    def push(self, obs: np.ndarray, action_idx: int):
        oh = np.zeros(ACT_DIM, dtype=np.float32); oh[action_idx] = 1.
        f  = np.concatenate([obs.astype(np.float32), oh])
        self.buf[:-1] = self.buf[1:]; self.buf[-1] = f

    def flat(self) -> np.ndarray: return self.buf.flatten()


def build_state(fs, dr, dark_steps, push_inferred, consec_stuck=0) -> np.ndarray:
    return np.concatenate([fs.flat(), dr.features(dark_steps, push_inferred, consec_stuck)])


# ─────────────────────────────────────────────────────────────
# DUELING DQN
# ─────────────────────────────────────────────────────────────
def _make_model():
    import torch.nn as nn

    class DuelingDQN(nn.Module):
        def __init__(self, in_dim=STATE_DIM, n_actions=N_ACTIONS):
            super().__init__()
            self.feature   = nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(),
                                           nn.Linear(256, 128),    nn.ReLU())
            self.value     = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
            self.advantage = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, n_actions))

        def forward(self, x):
            f = self.feature(x); v = self.value(f); a = self.advantage(f)
            return v + (a - a.mean(dim=1, keepdim=True))

    return DuelingDQN()


# ─────────────────────────────────────────────────────────────
# GLOBAL STATE
# ─────────────────────────────────────────────────────────────
_MODEL:         Optional[object]        = None
_FS:            Optional[FrameStack]    = None
_DR:            Optional[DeadReckoning] = None
_DARK_STEPS:    int   = 0
_PUSH_INFERRED: bool  = False
_LAST_ACTION:   int   = 2
_LAST_OBS:      Optional[np.ndarray]    = None
_EP_STEP:       int   = 0
_CONSEC_ROT:    int   = 0
_CONSEC_STUCK:  int   = 0


def _load_model():
    global _MODEL
    if _MODEL is not None: return
    import torch
    model = _make_model()
    d = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(d, "weights_phase3_v2_best.pth"),
        os.path.join(d, "weights_phase3_v2_2nd.pth"),
        os.path.join(d, "weights_phase3_v2.pth"),
        os.path.join(d, "weights_phase3_best.pth"),
        os.path.join(d, "weights_phase3.pth"),
        os.path.join(d, "weights.pth"),
    ]
    for wpath in candidates:
        if os.path.exists(wpath):
            model.load_state_dict(torch.load(wpath, map_location="cpu", weights_only=True))
            print(f"[agent_phase3_v2] Loaded: {wpath}")
            break
    else:
        print("[agent_phase3_v2] WARNING: No weights found.")
    model.eval(); _MODEL = model


def _detect_new_episode(obs: np.ndarray) -> bool:
    global _EP_STEP, _LAST_OBS
    if _EP_STEP >= RESET_EVERY: return True
    if _EP_STEP >= 5 and _LAST_OBS is not None:
        if np.all(obs[:17] == 0) and np.any(_LAST_OBS[:17] != 0):
            if int(np.sum(obs[:17] != _LAST_OBS[:17])) >= 6: return True
    return False


def _reset_episode(obs: np.ndarray):
    global _FS, _DR, _DARK_STEPS, _PUSH_INFERRED, _LAST_ACTION
    global _LAST_OBS, _EP_STEP, _CONSEC_ROT, _CONSEC_STUCK
    _FS            = FrameStack(FRAME_STACK); _FS.reset(obs, 2)
    _DR            = DeadReckoning()
    _DARK_STEPS    = 0; _PUSH_INFERRED = False; _LAST_ACTION = 2
    _LAST_OBS      = obs.copy(); _EP_STEP = 0
    _CONSEC_ROT    = 0; _CONSEC_STUCK = 0


# ─────────────────────────────────────────────────────────────
# POLICY
# ─────────────────────────────────────────────────────────────
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    import torch

    global _MODEL, _FS, _DR, _DARK_STEPS, _PUSH_INFERRED
    global _LAST_ACTION, _LAST_OBS, _EP_STEP, _CONSEC_ROT, _CONSEC_STUCK

    _load_model()

    # ── Episode boundary ──────────────────────────────────────
    if _FS is None or _detect_new_episode(obs):
        _reset_episode(obs)
    else:
        _FS.push(_LAST_OBS, _LAST_ACTION)

    # ── Update tracking ───────────────────────────────────────
    any_sensor     = bool(np.any(obs[:17]))
    _DARK_STEPS    = 0 if any_sensor else _DARK_STEPS + 1
    if obs[16]:    _PUSH_INFERRED = True
    _CONSEC_STUCK  = _CONSEC_STUCK + 1 if obs[17] else 0

    # ── Build state ───────────────────────────────────────────
    state = build_state(_FS, _DR, _DARK_STEPS, _PUSH_INFERRED, _CONSEC_STUCK)

    # ── HARD OVERRIDE 1: Gap-seek when stuck at wall ──────────
    # Image 1,2,4 fix: instead of spinning randomly, navigate toward gap
    if _CONSEC_STUCK >= 5 and _DR.near_wall():
        action_str = _DR.gap_seek_action()
        # Still don't allow FW if obs says stuck right now
        if action_str == "FW" and obs[17]:
            action_str = "L22"   # slight rotation if FW would hit wall
        a = ACTIONS.index(action_str)
        _DR.update(action_str)
        _LAST_ACTION = a; _LAST_OBS = obs.copy(); _EP_STEP += 1
        _CONSEC_ROT = 0 if action_str == "FW" else _CONSEC_ROT + 1
        return action_str

    # ── Greedy Q-action with hard FW mask when stuck ──────────
    s_t = torch.from_numpy(state).unsqueeze(0)
    with torch.no_grad():
        qs = _MODEL(s_t).squeeze(0).numpy().copy()

    # HARD MASK: FW = -∞ when stuck (images 1,2,4 root fix)
    if obs[17]:
        qs[2] = -1e9

    a          = int(np.argmax(qs))
    action_str = ACTIONS[a]

    # ── HARD OVERRIDE 2: Anti-oscillation (ONLY when NOT stuck) ──
    # Old code forced FW even when stuck — that was catastrophic.
    # Now only override when NOT stuck.
    if not obs[17]:
        if action_str != "FW":
            _CONSEC_ROT += 1
            if _CONSEC_ROT >= 12:   # 12 consecutive rotations → force FW
                action_str  = "FW"
                a           = 2
                _CONSEC_ROT = 0
        else:
            _CONSEC_ROT = 0
    else:
        _CONSEC_ROT = 0   # reset when stuck (rotations are intentional)

    # ── Update dead reckoning ─────────────────────────────────
    _DR.update(action_str)

    # ── Bookkeeping ───────────────────────────────────────────
    _LAST_ACTION = a; _LAST_OBS = obs.copy(); _EP_STEP += 1
    return action_str