
import math
import os
from collections import deque

import numpy as np
import torch
import torch.nn as nn

# WEIGHTS_FILE = "weights_p2v2.pth"
# WEIGHTS_FILE = "weights_p2v2_second.pth"
# WEIGHTS_FILE = "weights_p2v2_ep2000.pth"
WEIGHTS_FILE = "weights_p2v2_max2000_ep300.pth"


ACTIONS          = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS        = len(ACTIONS)
ROTATION_ACTIONS = {0, 1, 3, 4}
ROT_DIR          = {"L45": 1, "L22": 1, "FW": 0, "R22": -1, "R45": -1}

FRAME_STACK = 4
OBS_DIM     = 18
STACK_DIM   = FRAME_STACK * OBS_DIM
COORD_DIM   = 5
AUG_DIM     = STACK_DIM + N_ACTIONS + 1 + 1 + 1 + COORD_DIM  # 85

SONAR_FACE_ANGLES = [-90, -90, 0, 0, 0, 0, 90, 90]
NEAR_DIST_PX      = 67
FAR_DIST_PX       = 120
IR_DIST_PX        = 10
FORWARD_STEP_PX   = 5
MAX_BOX_CONF      = 30
MAX_ARENA_DIST    = 350.0
MAX_PURE_ROT      = 12


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


class CoordinateTracker:
    ANGLE_MAP = {"L45": 45.0, "L22": 22.5, "FW": 0.0, "R22": -22.5, "R45": -45.0}

    def __init__(self):
        self.robot_x         = 0.0
        self.robot_y         = 0.0
        self.robot_theta     = 0.0
        self.box_est_x       = 0.0
        self.box_est_y       = 0.0
        self.box_steps_since = 0
        self.has_estimate    = False
        self.in_push         = False

    def reset(self):
        self.robot_x         = 0.0
        self.robot_y         = 0.0
        self.robot_theta     = 0.0
        self.box_est_x       = 0.0
        self.box_est_y       = 0.0
        self.box_steps_since = 0
        self.has_estimate    = False
        self.in_push         = False

    def update_action(self, action, stuck_flag):
        delta = self.ANGLE_MAP[action]
        self.robot_theta = (self.robot_theta + delta) % 360.0
        if action == "FW" and not stuck_flag:
            t = math.radians(self.robot_theta)
            self.robot_x += FORWARD_STEP_PX * math.cos(t)
            self.robot_y += FORWARD_STEP_PX * math.sin(t)
            if self.in_push and self.has_estimate:
                self.box_est_x += FORWARD_STEP_PX * math.cos(t)
                self.box_est_y += FORWARD_STEP_PX * math.sin(t)

    def update_obs(self, obs):
        if bool(obs[16]):
            self.in_push = True

        any_sonar = bool(np.any(obs[:16]))
        ir_fired  = bool(obs[16])

        if not any_sonar and not ir_fired:
            if self.has_estimate:
                self.box_steps_since += 1
            return

        est_xs, est_ys, weights = [], [], []
        t = math.radians(self.robot_theta)

        if ir_fired:
            est_xs.append(self.robot_x + IR_DIST_PX * math.cos(t))
            est_ys.append(self.robot_y + IR_DIST_PX * math.sin(t))
            weights.append(4.0)

        for i in range(8):
            face_rad = math.radians(self.robot_theta + SONAR_FACE_ANGLES[i])
            if bool(obs[2 * i]):
                est_xs.append(self.robot_x + NEAR_DIST_PX * math.cos(face_rad))
                est_ys.append(self.robot_y + NEAR_DIST_PX * math.sin(face_rad))
                weights.append(2.0)
            if bool(obs[2 * i + 1]):
                est_xs.append(self.robot_x + FAR_DIST_PX * math.cos(face_rad))
                est_ys.append(self.robot_y + FAR_DIST_PX * math.sin(face_rad))
                weights.append(1.0)

        if est_xs:
            total_w          = sum(weights)
            self.box_est_x   = sum(w * x for w, x in zip(weights, est_xs)) / total_w
            self.box_est_y   = sum(w * y for w, y in zip(weights, est_ys)) / total_w
            self.box_steps_since = 0
            self.has_estimate    = True

    def get_features(self):
        if self.has_estimate:
            dx = self.box_est_x - self.robot_x
            dy = self.box_est_y - self.robot_y
            dist = math.sqrt(dx * dx + dy * dy)
            angle_to_box = math.degrees(math.atan2(dy, dx))
            rel_angle = angle_to_box - self.robot_theta
            while rel_angle >  180: rel_angle -= 360
            while rel_angle < -180: rel_angle += 360
            rel_rad = math.radians(rel_angle)
            return np.array([
                math.sin(rel_rad),
                math.cos(rel_rad),
                min(dist / MAX_ARENA_DIST, 1.0),
                min(self.box_steps_since / MAX_BOX_CONF, 1.0),
                1.0
            ], dtype=np.float32)
        return np.array([0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)

    def get_distance_to_box(self):
        if not self.has_estimate:
            return 0.0
        dx = self.box_est_x - self.robot_x
        dy = self.box_est_y - self.robot_y
        return math.sqrt(dx * dx + dy * dy)

    def bearing_to_box_deg(self):
        if not self.has_estimate:
            return 0.0
        dx = self.box_est_x - self.robot_x
        dy = self.box_est_y - self.robot_y
        rel = math.degrees(math.atan2(dy, dx)) - self.robot_theta
        while rel >  180: rel -= 360
        while rel < -180: rel += 360
        return rel


_model = None

def _load_model():
    global _model
    if _model is not None:
        return _model
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), WEIGHTS_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Weights not found: {path}")
    print(f"Loading weights from {path}...")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
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
_rot_dir_hist = deque(maxlen=8)

_tracker      = CoordinateTracker()

_in_push          = False
_no_sens_streak   = 0
_prev_obs         = None


def _reset():
    global _last_act, _rot_streak, _ir_streak, _pure_rot
    global _in_push, _no_sens_streak, _rot_dir_hist
    _last_act       = 2
    _rot_streak     = 0
    _stuck_window.clear()
    _ir_streak      = 0
    _pure_rot       = 0
    _rot_dir_hist   = deque(maxlen=8)
    _in_push        = False
    _no_sens_streak = 0
    _tracker.reset()


def _build_state():
    stacked = np.concatenate(list(_frames)).astype(np.float32)
    act_oh  = np.zeros(N_ACTIONS, dtype=np.float32)
    act_oh[_last_act] = 1.0
    rot   = np.array([_rot_streak / 20.0],         dtype=np.float32)
    stuck = np.array([sum(_stuck_window) / 10.0 if _stuck_window else 0.0],
                     dtype=np.float32)
    ir_s  = np.array([_ir_streak / 10.0],          dtype=np.float32)
    coord = _tracker.get_features()
    return np.concatenate([stacked, act_oh, rot, stuck, ir_s, coord])


def _detect_new_episode(obs):
    """
    Blink-safe episode boundary detection.
    Box blinks out for max 30 steps, so threshold = 35 to avoid false resets.
    """
    global _in_push, _no_sens_streak
    if _prev_obs is None:
        return True
    any_now = bool(np.any(obs[:17]))
    if _in_push and not any_now:
        _no_sens_streak += 1
        if _no_sens_streak >= 35:
            return True
    else:
        _no_sens_streak = 0
    return False


def policy(obs, rng=None):
    global _last_act, _rot_streak, _pure_rot, _in_push, _prev_obs
    global _ir_streak, _rot_dir_hist

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

   
    _tracker.update_obs(obs)

    _stuck_window.append(float(obs[17]))
    if bool(obs[16]):
        _ir_streak = min(_ir_streak + 1, 10)
        _in_push   = True
    else:
        _ir_streak = 0

    state = _build_state()
    with torch.no_grad():
        q = model(torch.FloatTensor(state).unsqueeze(0))
        action_idx = int(q.argmax(1).item())

    if action_idx == 2:  # FW
        _rot_dir_hist.clear()
    else:
        _rot_dir_hist.append(ROT_DIR[ACTIONS[action_idx]])

    any_sensor = bool(np.any(obs[:17]))

    if action_idx in ROTATION_ACTIONS and not any_sensor:
        _pure_rot += 1
    else:
        _pure_rot = 0

    if _pure_rot >= MAX_PURE_ROT:
        _pure_rot  = 0
        action_idx = 2  # FW

    elif (not any_sensor
          and _tracker.has_estimate
          and not _tracker.in_push
          and _tracker.box_steps_since < MAX_BOX_CONF):
        bearing = _tracker.bearing_to_box_deg()
        if abs(bearing) < 45.0:
            _pure_rot  = 0
            action_idx = 2  # FW

   
    _tracker.update_action(ACTIONS[action_idx], stuck_flag=bool(obs[17]))

    # ── Update rot_streak ─────────────────────────────────────────────────────
    _rot_streak = min(_rot_streak + 1, 20) if action_idx in ROTATION_ACTIONS else 0
    _last_act   = action_idx

    return ACTIONS[action_idx]