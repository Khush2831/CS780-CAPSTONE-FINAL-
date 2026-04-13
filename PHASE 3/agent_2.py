

from __future__ import annotations
import math
import os
from typing import Optional

import numpy as np

# ── constants (must match train_phase3.py exactly) ──────────
ACTIONS     = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS   = 5
OBS_DIM     = 18
ACT_DIM     = 5
EXTRA_DIM   = 6
FRAME_STACK = 8
PER_STEP    = OBS_DIM + ACT_DIM          # 23
STACK_DIM   = FRAME_STACK * PER_STEP    # 184
STATE_DIM   = STACK_DIM + EXTRA_DIM     # 190

ROT_DELTA   = {"L45": 45.0, "L22": 22.5, "FW": 0.0,
               "R22": -22.5, "R45": -45.0}
FW_STEP_PX  = 5.0
ARENA_SIZE  = 500

RESET_EVERY = 1005   # max_steps=1000 + small buffer


class DeadReckoning:
    def __init__(self, x=250.0, y=250.0, angle=0.0, arena=ARENA_SIZE):
        self.x     = x
        self.y     = y
        self.angle = angle
        self.arena = arena
        self._lo   = 40.0
        self._hi   = float(arena) - 40.0

    def reset(self, x=250.0, y=250.0, angle=0.0):
        self.x, self.y, self.angle = x, y, angle

    def update(self, action_str: str):
        self.angle += ROT_DELTA[action_str]
        if action_str == "FW":
            rad    = math.radians(self.angle)
            self.x = float(np.clip(self.x + FW_STEP_PX * math.cos(rad), self._lo, self._hi))
            self.y = float(np.clip(self.y + FW_STEP_PX * math.sin(rad), self._lo, self._hi))

    def features(self, dark_steps: int, push_inferred: bool) -> np.ndarray:
        rad       = math.radians(self.angle)
        lo, hi    = 10.0, float(self.arena) - 10.0
        min_bound = min(self.x - lo, self.y - lo, hi - self.x, hi - self.y)
        min_bound = float(np.clip(min_bound / (self.arena * 0.5), 0.0, 1.0))
        near_wall = float(210.0 < self.x < 290.0)
        return np.array([
            math.sin(rad),
            math.cos(rad),
            float(np.clip(dark_steps / FRAME_STACK, 0.0, 1.0)),
            float(push_inferred),
            min_bound,
            near_wall,
        ], dtype=np.float32)


class FrameStack:
    def __init__(self, k=FRAME_STACK):
        self.k   = k
        self.buf = np.zeros((k, PER_STEP), dtype=np.float32)

    def reset(self, obs: np.ndarray, action_idx: int = 2):   # 2 = "FW"
        oh = np.zeros(ACT_DIM, dtype=np.float32)
        oh[action_idx] = 1.0
        frame = np.concatenate([obs.astype(np.float32), oh])
        for i in range(self.k):
            self.buf[i] = frame

    def push(self, obs: np.ndarray, action_idx: int):
        oh = np.zeros(ACT_DIM, dtype=np.float32)
        oh[action_idx] = 1.0
        frame = np.concatenate([obs.astype(np.float32), oh])
        self.buf[:-1] = self.buf[1:]
        self.buf[-1]  = frame

    def flat(self) -> np.ndarray:
        return self.buf.flatten()


# ─────────────────────────────────────────────────────────────
# DUELING DQN  (identical to training)
# ─────────────────────────────────────────────────────────────
def _make_model():
    import torch.nn as nn
    import torch

    class DuelingDQN(nn.Module):
        def __init__(self, in_dim=STATE_DIM, n_actions=N_ACTIONS):
            super().__init__()
            self.feature = nn.Sequential(
                nn.Linear(in_dim, 256), nn.ReLU(),
                nn.Linear(256, 128),   nn.ReLU(),
            )
            self.value = nn.Sequential(
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, 1),
            )
            self.advantage = nn.Sequential(
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, n_actions),
            )

        def forward(self, x):
            f = self.feature(x)
            v = self.value(f)
            a = self.advantage(f)
            return v + (a - a.mean(dim=1, keepdim=True))

    return DuelingDQN()


_MODEL:         Optional[object]     = None   # DuelingDQN
_FS:            Optional[FrameStack] = None   # frame stack
_DR:            Optional[DeadReckoning] = None  # dead reckoning
_DARK_STEPS:    int   = 0
_PUSH_INFERRED: bool  = False
_LAST_ACTION:   int   = 2       # "FW"
_LAST_OBS:      Optional[np.ndarray] = None
_EP_STEP:       int   = 0
_CONSEC_ROT:    int   = 0       # consecutive non-FW actions (anti-oscillation)


def _load_model():
    """Lazy-load DuelingDQN weights once."""
    global _MODEL
    if _MODEL is not None:
        return
    import torch
    model = _make_model()
    submission_dir = os.path.dirname(os.path.abspath(__file__))

    # Try multiple weight paths in priority order
    candidates = [
        # os.path.join(submission_dir, "weights_phase3_2.pth"),
        os.path.join(submission_dir, "weights_phase3_2nd.pth"),
        # os.path.join(submission_dir, "weights_phase3.pth"),
        # os.path.join(submission_dir, "weights.pth"),
    ]
    loaded = False
    for wpath in candidates:
        if os.path.exists(wpath):
            model.load_state_dict(
                torch.load(wpath, map_location="cpu", weights_only=True))
            print(f"[agent_phase3] Loaded weights from {wpath}")
            loaded = True
            break
    if not loaded:
        print("[agent_phase3] WARNING: No weights found. Running untrained policy.")

    model.eval()
    _MODEL = model


def _detect_new_episode(obs: np.ndarray) -> bool:
    """
    Returns True if we believe a new episode has started.
    Two heuristics (either triggers a reset):
      A) Step counter exceeded RESET_EVERY (covers max-step timeouts).
      B) All sensors suddenly silent after at least 5 active steps
         AND obs differs from last obs by ≥ 6 bits (dramatic change).
    """
    global _EP_STEP, _LAST_OBS
    # A: step cap
    if _EP_STEP >= RESET_EVERY:
        return True
    # B: sensor-silence jump
    if _EP_STEP >= 5 and _LAST_OBS is not None:
        if np.all(obs[:17] == 0) and np.any(_LAST_OBS[:17] != 0):
            n_changed = int(np.sum(obs[:17] != _LAST_OBS[:17]))
            if n_changed >= 6:
                return True
    return False


def _reset_episode(obs: np.ndarray):
    """Reinitialise all global episode state at episode boundary."""
    global _FS, _DR, _DARK_STEPS, _PUSH_INFERRED, _LAST_ACTION
    global _LAST_OBS, _EP_STEP, _CONSEC_ROT

    _FS            = FrameStack(FRAME_STACK)
    _FS.reset(obs, action_idx=2)   # assume FW at start
    _DR            = DeadReckoning()   # start from arena center (unknown real pos)
    _DARK_STEPS    = 0
    _PUSH_INFERRED = False
    _LAST_ACTION   = 2                 # "FW"
    _LAST_OBS      = obs.copy()
    _EP_STEP       = 0
    _CONSEC_ROT    = 0


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    """
    Main entry point.

    obs : np.ndarray, shape (18,), dtype float, values 0 or 1
    rng : np.random.Generator (available but not needed for greedy policy)

    Returns one of: "L45", "L22", "FW", "R22", "R45"
    """
    import torch

    global _MODEL, _FS, _DR, _DARK_STEPS, _PUSH_INFERRED
    global _LAST_ACTION, _LAST_OBS, _EP_STEP, _CONSEC_ROT

    # ── Load model once ───────────────────────────────────────
    _load_model()

    # ── Episode boundary detection ────────────────────────────
    if _FS is None or _detect_new_episode(obs):
        _reset_episode(obs)
    else:
        # Update frame stack with (previous obs, action taken)
        _FS.push(_LAST_OBS, _LAST_ACTION)

    # ── Update tracking ───────────────────────────────────────
    any_sensor      = bool(np.any(obs[:17]))
    _DARK_STEPS     = 0 if any_sensor else _DARK_STEPS + 1
    if obs[16]:           # IR sensor fired → definitely near box
        _PUSH_INFERRED = True

    # ── Build state ───────────────────────────────────────────
    extra = _DR.features(_DARK_STEPS, _PUSH_INFERRED)
    state = np.concatenate([_FS.flat(), extra])

    # ── Greedy action ─────────────────────────────────────────
    s_t = torch.from_numpy(state).unsqueeze(0)
    with torch.no_grad():
        qs = _MODEL(s_t).squeeze(0).numpy()
    a = int(np.argmax(qs))

    # ── Anti-oscillation safety net ───────────────────────────
    # If agent has taken ≥ 10 consecutive rotation actions without
    # moving forward, override with FW to break the loop.
    action_str = ACTIONS[a]
    if action_str != "FW":
        _CONSEC_ROT += 1
        if _CONSEC_ROT >= 10:
            action_str  = "FW"
            a           = 2       # FW index
            _CONSEC_ROT = 0
    else:
        _CONSEC_ROT = 0

    # ── Update dead reckoning (for next step's features) ──────
    _DR.update(action_str)

    # ── Bookkeeping ───────────────────────────────────────────
    _LAST_ACTION = a
    _LAST_OBS    = obs.copy()
    _EP_STEP    += 1

    return action_str