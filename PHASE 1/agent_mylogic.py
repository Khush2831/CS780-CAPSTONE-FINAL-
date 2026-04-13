# import os
# import numpy as np
# from collections import deque

# ACTIONS      = ("L45", "L22", "FW", "R22", "R45")
# OBS_DIM      = 18
# STACK        = 4
# INPUT_DIM    = OBS_DIM * STACK
# MAX_STEPS    = 1000
# PUSH_RESERVE = 120

# _model       = None
# _frames      = deque(maxlen=STACK)
# _rng         = np.random.default_rng(0)

# _step        = 0
# _state       = "FIND"
# _ir_count    = 0
# _attach_fw   = 0
# _spin_dir    = 1
# _no_sensor   = 0


# def _build_model():
#     import torch.nn as nn
#     class DuelingQNetwork(nn.Module):
#         def __init__(self, input_dim=INPUT_DIM, n_actions=5):
#             super().__init__()
#             self.shared = nn.Sequential(
#                 nn.Linear(input_dim, 128), nn.ReLU(),
#                 nn.Linear(128, 128), nn.ReLU(),
#             )
#             self.value     = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
#             self.advantage = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, n_actions))
#         def forward(self, x):
#             h = self.shared(x)
#             v = self.value(h)
#             a = self.advantage(h)
#             return v + (a - a.mean(dim=1, keepdim=True))
#     return DuelingQNetwork()


# def _load_model():
#     global _model
#     if _model is not None:
#         return
#     import torch
#     path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights.pth")
#     m = _build_model()
#     m.load_state_dict(torch.load(path, map_location="cpu"))
#     m.eval()
#     _model = m


# def _reset(obs):
#     global _step, _state, _ir_count, _attach_fw, _spin_dir, _no_sensor
#     _step      = 0
#     _state     = "FIND"
#     _ir_count  = 0
#     _attach_fw = 0
#     _spin_dir  = 1
#     _no_sensor = 0
#     _frames.clear()
#     for _ in range(STACK):
#         _frames.append(obs.copy())


# def _find_action(obs):
#     global _spin_dir, _no_sensor

#     if obs[16] == 1:
#         return "FW"

#     left  = float(obs[0] + obs[1]*2 + obs[2] + obs[3]*2)
#     front = float(obs[4] + obs[5]*2 + obs[6]*2 + obs[7]*3 +
#                   obs[8]*2 + obs[9]*3 + obs[10] + obs[11]*2)
#     right = float(obs[12] + obs[13]*2 + obs[14] + obs[15]*2)

#     if left + front + right == 0:
#         _no_sensor += 1
#         if _no_sensor % 10 < 7:
#             return "L45" if _spin_dir > 0 else "R45"
#         return "FW"

#     _no_sensor = 0
#     if front >= left and front >= right:
#         return "FW"
#     if left > right:
#         return "L22" if left < 4 else "L45"
#     return "R22" if right < 4 else "R45"


# def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
#     global _step, _state, _ir_count, _attach_fw

#     # ── reliable episode reset: step counter hits MAX_STEPS ──────────────
#     # evaluator runs episodes back-to-back; _step==0 means fresh start
#     if _step == 0 or _step >= MAX_STEPS:
#         _reset(obs)

#     _frames.append(obs.copy())
#     _step += 1
#     steps_left = MAX_STEPS - _step

#     # ── FIND: navigate toward box ─────────────────────────────────────────
#     if _state == "FIND":
#         if obs[16] == 1:
#             _ir_count += 1
#             if _ir_count >= 2:
#                 _state = "ATTACH"
#                 _attach_fw = 0
#                 return "FW"
#         else:
#             _ir_count = max(0, _ir_count - 1)
#         return _find_action(obs)

#     # ── ATTACH: drive forward until overlap established ───────────────────
#     if _state == "ATTACH":
#         _attach_fw += 1
#         if _attach_fw >= 8:
#             _state = "SPIN"
#         return "FW"

#     # ── time to push: switch to PUSH when PUSH_RESERVE steps remain ───────
#     if steps_left <= PUSH_RESERVE:
#         _state = "PUSH"

#     # ── SPIN: rotate in place farming +100/step overlap bonus ────────────
#     if _state == "SPIN":
#         return "L45"

#     # ── PUSH: go straight to boundary ────────────────────────────────────
#     if _state == "PUSH":
#         if obs[17] == 1:
#             return "R22"
#         return "FW"

#     return "FW"




# import os
# import numpy as np
# from collections import deque

# ACTIONS      = ("L45", "L22", "FW", "R22", "R45")
# OBS_DIM      = 18
# STACK        = 4
# INPUT_DIM    = OBS_DIM * STACK
# MAX_STEPS    = 1000
# PUSH_RESERVE = 120

# _model     = None
# _frames    = deque(maxlen=STACK)
# _rng       = np.random.default_rng(0)
# _step      = 0
# _state     = "FIND"
# _ir_count  = 0
# _attach_fw = 0
# _spin_dir  = 1
# _no_sensor = 0


# def _build_model():
#     import torch.nn as nn
#     class DuelingQNetwork(nn.Module):
#         def __init__(self, input_dim=INPUT_DIM, n_actions=5):
#             super().__init__()
#             self.shared = nn.Sequential(
#                 nn.Linear(input_dim, 128), nn.ReLU(),
#                 nn.Linear(128, 128), nn.ReLU(),
#             )
#             self.value     = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
#             self.advantage = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, n_actions))
#         def forward(self, x):
#             h = self.shared(x)
#             v = self.value(h)
#             a = self.advantage(h)
#             return v + (a - a.mean(dim=1, keepdim=True))
#     return DuelingQNetwork()


# def _load_model():
#     global _model
#     if _model is not None:
#         return
#     import torch
#     path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights_best.pth")
#     m = _build_model()
#     m.load_state_dict(torch.load(path, map_location="cpu"))
#     m.eval()
#     _model = m


# def _reset(obs):
#     global _step, _state, _ir_count, _attach_fw, _spin_dir, _no_sensor
#     _step      = 0
#     _state     = "FIND"
#     _ir_count  = 0
#     _attach_fw = 0
#     _spin_dir  = 1
#     _no_sensor = 0
#     _frames.clear()
#     for _ in range(STACK):
#         _frames.append(obs.copy())


# def _is_new_episode(obs):
#     """
#     Three independent signals — any one triggers a reset:
#     1. Very first call ever (_step == 0)
#     2. Step counter hit max (_step >= MAX_STEPS) — episode ran full length
#     3. We think we're attached (SPIN/PUSH) but ALL sensors are dark
#        → box is far away → must be a new episode after early success
#     """
#     if _step == 0:
#         return True
#     if _step >= MAX_STEPS:
#         return True
#     if _state in ("SPIN", "PUSH") and np.sum(obs[:17]) == 0:
#         return True
#     return False


# def _find_action(obs):
#     global _spin_dir, _no_sensor

#     if obs[16] == 1:
#         return "FW"

#     left  = float(obs[0] + obs[1]*2 + obs[2] + obs[3]*2)
#     front = float(obs[4] + obs[5]*2 + obs[6]*2 + obs[7]*3 +
#                   obs[8]*2 + obs[9]*3 + obs[10] + obs[11]*2)
#     right = float(obs[12] + obs[13]*2 + obs[14] + obs[15]*2)

#     if left + front + right == 0:
#         _no_sensor += 1
#         if _no_sensor % 10 < 7:
#             return "L45" if _spin_dir > 0 else "R45"
#         return "FW"

#     _no_sensor = 0
#     if front >= left and front >= right:
#         return "FW"
#     if left > right:
#         return "L22" if left < 4 else "L45"
#     return "R22" if right < 4 else "R45"


# def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
#     global _step, _state, _ir_count, _attach_fw

#     if _is_new_episode(obs):
#         _reset(obs)

#     _frames.append(obs.copy())
#     _step += 1
#     steps_left = MAX_STEPS - _step

#     # ── FIND ──────────────────────────────────────────────────────────────
#     if _state == "FIND":
#         if obs[16] == 1:
#             _ir_count += 1
#             if _ir_count >= 2:
#                 _state = "ATTACH"
#                 _attach_fw = 0
#                 return "FW"
#         else:
#             _ir_count = max(0, _ir_count - 1)
#         return _find_action(obs)

#     # ── ATTACH: FW until overlap guaranteed ───────────────────────────────
#     if _state == "ATTACH":
#         _attach_fw += 1
#         if _attach_fw >= 8:
#             _state = "SPIN"
#         return "FW"

#     # ── switch to PUSH when budget runs low ───────────────────────────────
#     if steps_left <= PUSH_RESERVE:
#         _state = "PUSH"

#     # ── SPIN: harvest +100/step overlap bonus ─────────────────────────────
#     if _state == "SPIN":
#         return "L45"

#     # ── PUSH: drive straight to boundary ─────────────────────────────────
#     if _state == "PUSH":
#         if obs[17] == 1:
#             return "R22"
#         return "FW"

#     return "FW"



import os
import numpy as np
from collections import deque

ACTIONS      = ("L45", "L22", "FW", "R22", "R45")
OBS_DIM      = 18
STACK        = 4
INPUT_DIM    = OBS_DIM * STACK
MAX_STEPS    = 1000
PUSH_RESERVE = 120

_model     = None
_frames    = deque(maxlen=STACK)
_rng       = np.random.default_rng(0)
_step      = 0
_state     = "FIND"
_ir_count  = 0
_attach_fw = 0


def _build_model():
    import torch.nn as nn
    class DuelingQNetwork(nn.Module):
        def __init__(self, input_dim=INPUT_DIM, n_actions=5):
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(input_dim, 128), nn.ReLU(),
                nn.Linear(128, 128), nn.ReLU(),
            )
            self.value     = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
            self.advantage = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, n_actions))
        def forward(self, x):
            h = self.shared(x)
            v = self.value(h)
            a = self.advantage(h)
            return v + (a - a.mean(dim=1, keepdim=True))
    return DuelingQNetwork()


def _load_model():
    global _model
    if _model is not None:
        return
    import torch
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights_best.pth")
    m = _build_model()
    m.load_state_dict(torch.load(path, map_location="cpu"))
    m.eval()
    _model = m


def _reset(obs):
    global _step, _state, _ir_count, _attach_fw
    _step      = 0
    _state     = "FIND"
    _ir_count  = 0
    _attach_fw = 0
    _frames.clear()
    for _ in range(STACK):
        _frames.append(obs.copy())


def _is_new_episode(obs):
    if _step == 0:
        return True
    if _step >= MAX_STEPS:
        return True
    if _state in ("SPIN", "PUSH") and np.sum(obs[:17]) == 0:
        return True
    return False


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _step, _state, _ir_count, _attach_fw

    if _is_new_episode(obs):
        _reset(obs)

    _frames.append(obs.copy())
    _step += 1
    steps_left = MAX_STEPS - _step

    # ── ATTACH ────────────────────────────────────────────────────────────
    if _state == "ATTACH":
        _attach_fw += 1
        if _attach_fw >= 8:
            _state = "SPIN"
        return "FW"

    # ── switch to PUSH when budget low ────────────────────────────────────
    if _state in ("SPIN", "PUSH") and steps_left <= PUSH_RESERVE:
        _state = "PUSH"

    # ── SPIN ──────────────────────────────────────────────────────────────
    if _state == "SPIN":
        return "L45"

    # ── PUSH ──────────────────────────────────────────────────────────────
    if _state == "PUSH":
        return "R22" if obs[17] == 1 else "FW"

    # ── FIND: IR override then neural network ─────────────────────────────
    if obs[16] == 1:
        _ir_count += 1
        if _ir_count >= 2:
            _state     = "ATTACH"
            _attach_fw = 0
        return "FW"
    else:
        _ir_count = max(0, _ir_count - 1)

    if obs[17] == 1:
        return "L22" if _rng.random() < 0.5 else "R22"

    # network handles ALL of find phase — both near and far
    _load_model()
    import torch
    stacked = np.concatenate(list(_frames)).astype(np.float32)
    x = torch.from_numpy(stacked).unsqueeze(0)
    with torch.no_grad():
        q = _model(x).squeeze(0).numpy()
    return ACTIONS[int(np.argmax(q))]