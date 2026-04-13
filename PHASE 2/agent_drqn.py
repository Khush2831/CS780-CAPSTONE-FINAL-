import os
from collections import deque

import numpy as np
import torch
import torch.nn as nn

ACTIONS          = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS        = len(ACTIONS)
ROTATION_ACTIONS = {0, 1, 3, 4}

OBS_DIM    = 18
INPUT_DIM  = OBS_DIM + N_ACTIONS + 3   # 26: raw obs + last-act one-hot + rot/stuck/ir
GRU_HIDDEN = 128

BLINK_CAP    = 30
MAX_PURE_ROT = 12


class DuelingDRQN(nn.Module):
    def __init__(self, input_dim, gru_hidden, n_actions):
        super().__init__()
        self.gru_hidden = gru_hidden
        self.gru = nn.GRU(input_dim, gru_hidden, batch_first=True)
        self.value     = nn.Sequential(
            nn.Linear(gru_hidden, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        self.advantage = nn.Sequential(
            nn.Linear(gru_hidden, 64), nn.ReLU(), nn.Linear(64, n_actions)
        )

    def forward(self, x, h=None):
        out, h_out = self.gru(x, h)
        v = self.value(out)
        a = self.advantage(out)
        Q = v + (a - a.mean(dim=2, keepdim=True))
        return Q, h_out

    def init_hidden(self, batch_size, device):
        return torch.zeros(1, batch_size, self.gru_hidden, device=device)


_model = None

def _load_model():
    global _model
    if _model is not None:
        return _model
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights_drqn_phase2.pth")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Weights not found at {path}")
    print(f"Loading DRQN weights from {path}...")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    m = DuelingDRQN(
        ckpt.get("input_dim", INPUT_DIM),
        ckpt.get("gru_hidden", GRU_HIDDEN),
        ckpt.get("n_actions", N_ACTIONS),
    )
    m.load_state_dict(ckpt["online_state_dict"])
    m.eval()
    _model = m
    return _model


# ── Per-episode state (GRU hidden + input builder) ───────────────────────────
_h            = None   # GRU hidden state: (1, 1, GRU_HIDDEN)
_last_act     = 2
_rot_streak   = 0
_stuck_window = deque(maxlen=10)
_ir_streak    = 0
_blink_steps  = 0
_pure_rot     = 0
_in_push      = False
_no_sens_streak = 0
_prev_obs     = None


def _reset(model):
    global _h, _last_act, _rot_streak, _ir_streak
    global _blink_steps, _pure_rot, _in_push, _no_sens_streak
    _h              = model.init_hidden(1, torch.device("cpu"))
    _last_act       = 2
    _rot_streak     = 0
    _stuck_window.clear()
    _ir_streak      = 0
    _blink_steps    = 0
    _pure_rot       = 0
    _in_push        = False
    _no_sens_streak = 0


def _build_input(obs):
    raw  = obs.astype(np.float32)                                      # 18
    act  = np.zeros(N_ACTIONS, dtype=np.float32)
    act[_last_act] = 1.0                                               #  5
    rot  = np.array([_rot_streak / 20.0], dtype=np.float32)           #  1
    stk  = np.array([sum(_stuck_window) / 10.0 if _stuck_window else 0.0],
                    dtype=np.float32)                                   #  1
    ir_s = np.array([_ir_streak / 10.0], dtype=np.float32)            #  1
    return np.concatenate([raw, act, rot, stk, ir_s])                  # 26


def _detect_new_episode(obs):
    """
    Blink-safe episode boundary detection.
    Threshold 30 avoids false resets during blink silence.
    """
    global _in_push, _no_sens_streak
    if _prev_obs is None:
        return True
    any_now = bool(np.any(obs[:17]))
    if _in_push and not any_now:
        _no_sens_streak += 1
        if _no_sens_streak >= 30:
            return True
    else:
        _no_sens_streak = 0
    return False


def policy(obs, rng=None):
    global _h, _last_act, _rot_streak, _pure_rot, _in_push, _prev_obs
    global _ir_streak, _stuck_window, _blink_steps

    obs   = np.asarray(obs, dtype=np.float32)
    model = _load_model()

    if _detect_new_episode(obs):
        _reset(model)

    _prev_obs = obs.copy()

    # Update features
    _stuck_window.append(float(obs[17]))

    if bool(obs[16]):
        _ir_streak = min(_ir_streak + 1, 10)
        _in_push   = True
    else:
        _ir_streak = 0

    if bool(np.any(obs[:16])):
        _blink_steps = 0
    else:
        _blink_steps = min(_blink_steps + 1, BLINK_CAP)

    # Pure-rotation override: force FW if spinning with no sensors
    any_sensor = bool(np.any(obs[:17]))
    if _last_act in ROTATION_ACTIONS and not any_sensor:
        _pure_rot += 1
    else:
        _pure_rot = 0

    if _pure_rot >= MAX_PURE_ROT:
        _pure_rot  = 0
        action_idx = 2
        # Update streak for forced FW (no GRU step needed — h carries on)
    else:
        inp_vec = _build_input(obs)
        inp_t   = torch.FloatTensor(inp_vec).unsqueeze(0).unsqueeze(0)  # (1,1,26)
        with torch.no_grad():
            Q, _h = model(inp_t, _h)
        action_idx = int(Q[0, 0].argmax().item())

    _rot_streak = min(_rot_streak + 1, 20) if action_idx in ROTATION_ACTIONS else 0
    _last_act   = action_idx
    return ACTIONS[action_idx]