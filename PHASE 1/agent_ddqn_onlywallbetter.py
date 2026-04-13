
import os
from collections import deque

import numpy as np
import torch
import torch.nn as nn

ACTIONS          = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS        = len(ACTIONS)
ROTATION_ACTIONS = {0, 1, 3, 4}

FRAME_STACK = 4
OBS_DIM     = 18
STACK_DIM   = FRAME_STACK * OBS_DIM
AUG_DIM     = STACK_DIM + N_ACTIONS + 1 + 1 + 1   

MAX_PURE_ROT = 12  


class DuelingDDQN(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256),       nn.ReLU(),
        )
        self.value     = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
        self.advantage = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, n_actions))

    def forward(self, x):
        h = self.body(x)
        v = self.value(h)
        a = self.advantage(h)
        return v + (a - a.mean(dim=1, keepdim=True))


_model = None

def _load_model():
    global _model
    if _model is not None:
        return _model
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights_onlywallbetter3000.pth")
    # path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights_best1000.pth")

    if not os.path.exists(path):
        raise FileNotFoundError(f"weights.pth not found at {path}")
    else   : print(f"Loading model weights from {path}...")

    ckpt = torch.load(path, map_location="cpu")
    m = DuelingDDQN(ckpt.get("aug_dim", AUG_DIM), ckpt.get("n_actions", N_ACTIONS))
    m.load_state_dict(ckpt["online_state_dict"])
    m.eval()
    _model = m
    return _model

_frames       = deque(maxlen=FRAME_STACK)
_last_act     = 2
_rot_streak   = 0
_stuck_window = deque(maxlen=10)
_ir_streak    = 0
_pure_rot     = 0      
_in_push      = False
_no_sensor_streak = 0
_prev_obs     = None


def _reset():
    global _last_act, _rot_streak, _stuck_window, _ir_streak
    global _pure_rot, _in_push, _no_sensor_streak
    _last_act    = 2
    _rot_streak  = 0
    _stuck_window.clear()
    _ir_streak   = 0
    _pure_rot    = 0
    _in_push     = False
    _no_sensor_streak = 0


def _build_state():
    stacked = np.concatenate(list(_frames)).astype(np.float32)
    act_oh  = np.zeros(N_ACTIONS, dtype=np.float32)
    act_oh[_last_act] = 1.0
    rot   = np.array([_rot_streak / 20.0],              dtype=np.float32)
    stuck = np.array([sum(_stuck_window) / 10.0],       dtype=np.float32)
    ir_s  = np.array([_ir_streak / 10.0],               dtype=np.float32)
    return np.concatenate([stacked, act_oh, rot, stuck, ir_s])


def _detect_new_episode(obs):
    global _in_push, _no_sensor_streak
    if _prev_obs is None:
        return True
    any_now = bool(np.any(obs[:17]))
    if _in_push and not any_now:
        _no_sensor_streak += 1
        if _no_sensor_streak >= 3:
            return True
    else:
        _no_sensor_streak = 0
    return False


def policy(obs, rng=None):
    global _last_act, _rot_streak, _pure_rot, _in_push, _prev_obs
    global _ir_streak, _stuck_window

    obs   = np.asarray(obs, dtype=np.float32)
    model = _load_model()

    if _detect_new_episode(obs):
        _reset()
        _frames.clear()
        for _ in range(FRAME_STACK):
            _frames.append(obs.copy())
    else:
        _frames.append(obs.copy())

    _prev_obs = obs.copy()

    _stuck_window.append(float(obs[17]))
    if bool(obs[16]):
        _ir_streak = min(_ir_streak + 1, 10)
        _in_push   = True
    else:
        _ir_streak = 0

    any_sensor = bool(np.any(obs[:17]))
    if _last_act in ROTATION_ACTIONS and not any_sensor:
        _pure_rot += 1
    else:
        _pure_rot = 0

    if _pure_rot >= MAX_PURE_ROT:
        _pure_rot  = 0
        action_idx = 2  
    else:
        state = _build_state()
        with torch.no_grad():
            q = model(torch.FloatTensor(state).unsqueeze(0))
            action_idx = int(q.argmax(1).item())

    _rot_streak = min(_rot_streak + 1, 20) if action_idx in ROTATION_ACTIONS else 0
    _last_act   = action_idx

    return ACTIONS[action_idx]