"""
train_phase2_v2.py  –  Dueling DDQN  |  Phase 2: Blinking Box (no wall)
=========================================================================
Run (fresh):
    python train_phase2_v2.py --episodes 4000 --save weights_p2v2.pth

Run (resume):
    python train_phase2_v2.py --episodes 4000 --resume weights_p2v2.pth --save weights_p2v2.pth

WHAT'S NEW vs PHASE 2 v1
--------------------------
1. CoordinateTracker: dead-reckoning (x,y,theta) for robot + box estimate
   - FW moves exactly 5px in heading direction (if not stuck)
   - Rotations: ±22.5°, ±45° tracked exactly
   - Stuck_flag (obs[17]) prevents false position updates on blocked FW
   - Box position estimated from sensor geometry when sensors fire
   - During blink: box position held from last sighting (it's stationary in Level 2)
   - 5 geometric features fed to NN: sin(rel_angle), cos(rel_angle),
     dist_norm, confidence_norm, has_estimate

2. Circular spin penalty (-200):
   - 8 consecutive same-direction rotations (all-left or all-right) while sensors dark
   - Directly penalises the "full circle spin" failure mode the user observed
   - Resets tracking after penalty fires

3. Bearing improvement bonus (+3.0):
   - FW reduces distance to estimated box while sensors are DARK
   - Teaches agent to navigate toward last-known box position during blink
   - Only fires when box estimate exists and sensors are silent

4. Terminal timeout penalty (-2 * max_steps):
   - Applied at terminal step when episode times out WITHOUT success
   - Strongly distinguishes "timeout failure" from "in-progress"
   - Motivates the agent to solve the problem rather than drift

5. Multiple checkpoint saving:
   - weights_best.pth      : highest mean100 ever seen
   - weights_second.pth    : second-highest distinct checkpoint
   - weights_ep{N}.pth     : periodic save every 500 episodes

6. No wall obstacles (wall_obstacles=False, user confirmed only no-wall matters)
   - WALL_CONFUSION_PENALTY removed
   - Faster convergence without wall disambiguation problem

AUG_DIM = 72 (stack) + 5 (last-act) + 1 (rot_streak) + 1 (recent_stuck)
                      + 1 (ir_streak) + 5 (coord_features) = 85
"""

import argparse
import math
import os
import random
import sys
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from obelix import OBELIX

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
ACTIONS          = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS        = len(ACTIONS)
ROTATION_ACTIONS = {0, 1, 3, 4}          # indices in ACTIONS
ROT_DIR          = {"L45": 1, "L22": 1, "FW": 0, "R22": -1, "R45": -1}

FRAME_STACK = 4
OBS_DIM     = 18
STACK_DIM   = FRAME_STACK * OBS_DIM      # 72
COORD_DIM   = 5                           # from CoordinateTracker
#                       stacked  last-act  rot_str  stuck  ir_str  coord
AUG_DIM     = STACK_DIM + N_ACTIONS + 1 + 1 + 1 + COORD_DIM       # = 85

# Sensor geometry (from obelix.py)
SONAR_FACE_ANGLES = [-90, -90, 0, 0, 0, 0, 90, 90]  # degrees, per sonar sensor
NEAR_DIST_PX      = 67    # midpoint of near zone (45–90 px)
FAR_DIST_PX       = 120   # midpoint of far zone  (90–150 px)
IR_DIST_PX        = 10    # midpoint of IR zone   (0–20 px)
FORWARD_STEP_PX   = 5     # hardcoded in obelix.py
MAX_BOX_CONF      = 30    # normalisation cap for box confidence (blink off lasts 10-30 steps)
MAX_ARENA_DIST    = 350.0 # rough max distance across the arena (px)

GAMMA           = 0.97
LR              = 3e-4
BATCH_SIZE      = 128
REPLAY_SIZE     = 100_000
MIN_REPLAY      = 2_000
TARGET_UPDATE   = 500
CLIP_GRAD       = 1.0

EPS_START        = 1.0
EPS_END          = 0.05
EPS_DECAY_STEPS  = 120_000

PERIODIC_SAVE_EVERY = 100   # save weights every N episodes

# ── Reward shaping coefficients ──────────────────────────────────────────────
ESCAPE_BONUS              =  15.0   # rotation freed robot from stuck
SENSOR_PROGRESS_BONUS     =   2.0   # new sensor bit lit after rotation
FORWARD_MOMENTUM_BONUS    =   1.0   # FW succeeded while sensors active
ROTATION_LOOP_K           =   8
ROTATION_LOOP_PENALTY     =  -8.0   # K rotations, sensors dark, no recent sighting
IR_BONUS                  =  10.0   # first IR activation this episode
BEARING_IMPROVEMENT_BONUS =   3.0   # FW reduces dist to estimated box while sensors dark
CIRCULAR_SPIN_PENALTY     = -200.0  # 8 same-dir rotations while sensors dark
# Applied once at terminal step:
TIMEOUT_PENALTY_FACTOR    =  -2.0   # multiplied by max_steps if episode times out


# ─────────────────────────────────────────────────────────────────────────────
# CoordinateTracker
# ─────────────────────────────────────────────────────────────────────────────
class CoordinateTracker:
    """
    Dead-reckoning tracker for robot position and box position estimate.

    Robot position is tracked relative to episode start (starts at 0,0,theta=0).
    Box position is estimated from sensor geometry whenever sensors fire.
    During blink (sensors dark), last-known box position is held — this is
    correct for Level 2 because the box is STATIONARY (not moving).

    The 5 output features are episode-invariant (relative direction/distance)
    and bounded, making them clean NN inputs.
    """

    ANGLE_MAP = {"L45": 45.0, "L22": 22.5, "FW": 0.0, "R22": -22.5, "R45": -45.0}

    def __init__(self):
        self.robot_x            = 0.0
        self.robot_y            = 0.0
        self.robot_theta        = 0.0   # degrees
        self.box_est_x          = 0.0
        self.box_est_y          = 0.0
        self.box_steps_since    = 0     # steps since box was last observed
        self.has_estimate       = False
        self.in_push            = False

    def reset(self):
        self.robot_x        = 0.0
        self.robot_y        = 0.0
        self.robot_theta    = 0.0
        self.box_est_x      = 0.0
        self.box_est_y      = 0.0
        self.box_steps_since= 0
        self.has_estimate   = False
        self.in_push        = False

    def update_action(self, action, stuck_flag):
        """
        Update robot position from the action just taken.
        stuck_flag = obs[17] from the RETURNED observation after the step.
        Rotations always succeed; FW only updates position if not stuck.
        """
        delta = self.ANGLE_MAP[action]
        self.robot_theta = (self.robot_theta + delta) % 360.0

        if action == "FW" and not stuck_flag:
            t = math.radians(self.robot_theta)
            self.robot_x += FORWARD_STEP_PX * math.cos(t)
            self.robot_y += FORWARD_STEP_PX * math.sin(t)
            # Box moves with robot when attached
            if self.in_push and self.has_estimate:
                self.box_est_x += FORWARD_STEP_PX * math.cos(t)
                self.box_est_y += FORWARD_STEP_PX * math.sin(t)

    def update_obs(self, obs):
        """
        Update box position estimate from current sensor observation.
        Weighted average across all firing sensors; higher weight for
        closer/more-direct sensors.
        """
        # Detect push state via IR proxy
        if bool(obs[16]):
            self.in_push = True

        any_sonar = bool(np.any(obs[:16]))
        ir_fired  = bool(obs[16])

        if not any_sonar and not ir_fired:
            # Sensors dark — box is either blinked out or out of range
            if self.has_estimate:
                self.box_steps_since += 1
            return

        # Gather position estimates from all firing sensors
        est_xs, est_ys, weights = [], [], []
        t = math.radians(self.robot_theta)

        if ir_fired:
            ex = self.robot_x + IR_DIST_PX * math.cos(t)
            ey = self.robot_y + IR_DIST_PX * math.sin(t)
            est_xs.append(ex); est_ys.append(ey); weights.append(4.0)

        for i in range(8):
            face_rad = math.radians(self.robot_theta + SONAR_FACE_ANGLES[i])
            near_bit = bool(obs[2 * i])
            far_bit  = bool(obs[2 * i + 1])
            if near_bit:
                ex = self.robot_x + NEAR_DIST_PX * math.cos(face_rad)
                ey = self.robot_y + NEAR_DIST_PX * math.sin(face_rad)
                est_xs.append(ex); est_ys.append(ey); weights.append(2.0)
            if far_bit:
                ex = self.robot_x + FAR_DIST_PX * math.cos(face_rad)
                ey = self.robot_y + FAR_DIST_PX * math.sin(face_rad)
                est_xs.append(ex); est_ys.append(ey); weights.append(1.0)

        if est_xs:
            total_w       = sum(weights)
            self.box_est_x = sum(w * x for w, x in zip(weights, est_xs)) / total_w
            self.box_est_y = sum(w * y for w, y in zip(weights, est_ys)) / total_w
            self.box_steps_since = 0
            self.has_estimate    = True

    def get_features(self):
        """
        Returns 5 normalised geometric features for NN input:
          [sin(rel_angle), cos(rel_angle), dist_norm, confidence_norm, has_estimate]
        All values in [-1, 1] or [0, 1]. Episode-invariant.
        """
        if self.has_estimate:
            dx = self.box_est_x - self.robot_x
            dy = self.box_est_y - self.robot_y
            dist = math.sqrt(dx * dx + dy * dy)

            angle_to_box = math.degrees(math.atan2(dy, dx))
            rel_angle    = angle_to_box - self.robot_theta
            # Normalise to [-180, 180]
            while rel_angle >  180: rel_angle -= 360
            while rel_angle < -180: rel_angle += 360

            rel_rad  = math.radians(rel_angle)
            sin_a    = math.sin(rel_rad)
            cos_a    = math.cos(rel_rad)
            d_norm   = min(dist / MAX_ARENA_DIST, 1.0)
            conf     = min(self.box_steps_since / MAX_BOX_CONF, 1.0)
            has_est  = 1.0
        else:
            sin_a, cos_a, d_norm, conf, has_est = 0.0, 0.0, 0.0, 1.0, 0.0

        return np.array([sin_a, cos_a, d_norm, conf, has_est], dtype=np.float32)

    def get_distance_to_box(self):
        if not self.has_estimate:
            return 0.0
        dx = self.box_est_x - self.robot_x
        dy = self.box_est_y - self.robot_y
        return math.sqrt(dx * dx + dy * dy)

    def bearing_to_box_deg(self):
        """Relative bearing to estimated box in degrees (in [-180, 180])."""
        if not self.has_estimate:
            return 0.0
        dx = self.box_est_x - self.robot_x
        dy = self.box_est_y - self.robot_y
        angle_to_box = math.degrees(math.atan2(dy, dx))
        rel = angle_to_box - self.robot_theta
        while rel >  180: rel -= 360
        while rel < -180: rel += 360
        return rel


# ─────────────────────────────────────────────────────────────────────────────
# Dueling DDQN
# ─────────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# Prioritized Replay Buffer
# ─────────────────────────────────────────────────────────────────────────────
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity   = capacity
        self.alpha      = alpha
        self.buffer     = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos        = 0
        self.max_prio   = 1.0

    def push(self, t):
        if len(self.buffer) < self.capacity:
            self.buffer.append(t)
        else:
            self.buffer[self.pos] = t
        self.priorities[self.pos] = self.max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        n     = len(self.buffer)
        prios = self.priorities[:n] ** self.alpha
        probs = prios / prios.sum()
        idxs  = np.random.choice(n, batch_size, replace=False, p=probs)
        w     = (n * probs[idxs]) ** (-beta)
        w    /= w.max()
        return [self.buffer[i] for i in idxs], idxs, torch.FloatTensor(w)

    def update_priorities(self, idxs, td_errors):
        for idx, err in zip(idxs, td_errors):
            self.priorities[idx] = float(abs(err)) + 1e-6
            self.max_prio = max(self.max_prio, self.priorities[idx])

    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────────────────────────────────────────
# Reward Shaper
# ─────────────────────────────────────────────────────────────────────────────
class RewardShaper:
    def __init__(self):
        self.prev_stuck    = False
        self.prev_sensor   = np.zeros(17, dtype=bool)
        self.action_hist   = deque(maxlen=ROTATION_LOOP_K)
        self.ir_given      = False
        # Circular spin tracking (8 same-dir rotations)
        self.rot_dir_hist  = deque(maxlen=8)

    def reset(self):
        self.prev_stuck  = False
        self.prev_sensor = np.zeros(17, dtype=bool)
        self.action_hist.clear()
        self.ir_given    = False
        self.rot_dir_hist.clear()

    def shape(self, raw, action_idx, obs, next_obs, tracker,
              prev_dist_to_box, done, won, max_steps):
        shaped     = raw
        action     = ACTIONS[action_idx]
        next_stuck = bool(next_obs[17])
        curr_bits  = obs[:17].astype(bool)
        next_bits  = next_obs[:17].astype(bool)
        ir_now     = bool(next_obs[16])
        any_sonar  = bool(np.any(next_obs[:16]))
        any_sensor = bool(np.any(next_obs[:17]))

        # ── Circular spin tracking ────────────────────────────────────────────
        if action == "FW":
            self.rot_dir_hist.clear()
        else:
            self.rot_dir_hist.append(ROT_DIR[action])

        # ── 1. Escape bonus ───────────────────────────────────────────────────
        if self.prev_stuck and action_idx in ROTATION_ACTIONS and not next_stuck:
            shaped += ESCAPE_BONUS

        # ── 2. Sensor progress after rotation ────────────────────────────────
        if action_idx in ROTATION_ACTIONS:
            n_new = int((next_bits & ~curr_bits).sum())
            if n_new > 0:
                shaped += SENSOR_PROGRESS_BONUS * n_new

        # ── 3. Forward momentum ───────────────────────────────────────────────
        if action_idx == 2 and not next_stuck and np.any(curr_bits):
            shaped += FORWARD_MOMENTUM_BONUS

        # ── 4. Rotation loop penalty ──────────────────────────────────────────
        #    Gate: only penalise if box estimate is stale (not a blink we're riding out)
        self.action_hist.append(action_idx)
        box_recently_seen = (tracker.has_estimate and tracker.box_steps_since < 15)
        if (len(self.action_hist) == ROTATION_LOOP_K
                and all(a in ROTATION_ACTIONS for a in self.action_hist)
                and not np.any(next_bits)
                and not box_recently_seen):
            shaped += ROTATION_LOOP_PENALTY

        # ── 5. IR bonus ───────────────────────────────────────────────────────
        if ir_now and not self.ir_given:
            shaped += IR_BONUS
            self.ir_given = True

        # ── 6. Bearing improvement bonus ─────────────────────────────────────
        #    FW reduced distance to estimated box while sensors were DARK
        #    (if sensors are active, forward_momentum_bonus already fires)
        if (action_idx == 2
                and not next_stuck
                and not np.any(curr_bits)        # sensors were dark before action
                and tracker.has_estimate
                and not tracker.in_push
                and prev_dist_to_box > 0):
            curr_dist = tracker.get_distance_to_box()
            delta = prev_dist_to_box - curr_dist
            if delta > 0:
                shaped += BEARING_IMPROVEMENT_BONUS * (delta / FORWARD_STEP_PX)

        # ── 7. Circular spin penalty ──────────────────────────────────────────
        if (len(self.rot_dir_hist) == 8
                and not np.any(next_obs[:17])
                and (all(d ==  1 for d in self.rot_dir_hist)
                     or all(d == -1 for d in self.rot_dir_hist))):
            shaped += CIRCULAR_SPIN_PENALTY
            self.rot_dir_hist.clear()

        # ── 8. Terminal timeout penalty ───────────────────────────────────────
        if done and not won:
            shaped += TIMEOUT_PENALTY_FACTOR * max_steps

        self.prev_stuck  = next_stuck
        self.prev_sensor = next_bits.copy()
        return shaped


# ─────────────────────────────────────────────────────────────────────────────
# Observation builder
# ─────────────────────────────────────────────────────────────────────────────
class ObsBuilder:
    """
    Builds the augmented state vector fed to the NN.
    Replaces blink_memory with CoordinateTracker geometric features.
    """

    def __init__(self):
        self.frames       = deque(maxlen=FRAME_STACK)
        self.last_act     = 2
        self.rot_streak   = 0
        self.stuck_window = deque(maxlen=10)
        self.ir_streak    = 0

    def reset(self, obs, tracker):
        self.frames.clear()
        for _ in range(FRAME_STACK):
            self.frames.append(obs.copy())
        self.last_act    = 2
        self.rot_streak  = 0
        self.stuck_window.clear()
        self.ir_streak   = 0
        return self._build(tracker)

    def step(self, obs, action_idx, tracker):
        self.frames.append(obs.copy())
        self.last_act = action_idx

        if action_idx in ROTATION_ACTIONS:
            self.rot_streak = min(self.rot_streak + 1, 20)
        else:
            self.rot_streak = 0

        self.stuck_window.append(float(obs[17]))

        if bool(obs[16]):
            self.ir_streak = min(self.ir_streak + 1, 10)
        else:
            self.ir_streak = 0

        return self._build(tracker)

    def _build(self, tracker):
        stacked = np.concatenate(list(self.frames)).astype(np.float32)  # 72

        act_oh = np.zeros(N_ACTIONS, dtype=np.float32)
        act_oh[self.last_act] = 1.0                                      #  5

        rot   = np.array([self.rot_streak / 20.0],         dtype=np.float32)  # 1
        stuck = np.array([sum(self.stuck_window) / 10.0 if self.stuck_window else 0.0],
                         dtype=np.float32)                                     # 1
        ir_s  = np.array([self.ir_streak / 10.0],          dtype=np.float32)  # 1

        coord = tracker.get_features()                                   #  5

        return np.concatenate([stacked, act_oh, rot, stuck, ir_s, coord])  # 85


# ─────────────────────────────────────────────────────────────────────────────
# Heuristic override
# ─────────────────────────────────────────────────────────────────────────────
def heuristic_override(action_idx, obs, tracker, pure_rot_count):
    """
    Two safety overrides that wrap the RL policy:

    1. Bearing override: if sensors are dark, we know where the box is,
       and it is roughly ahead → force FW to navigate toward it.

    2. Pure-rotation override: if agent has been rotating for MAX_PURE_ROT
       consecutive steps with all sensors dark → force FW to break the loop.

    Returns (final_action_idx, new_pure_rot_count).
    """
    MAX_PURE_ROT = 12

    any_sensor = bool(np.any(obs[:17]))

    # Update pure_rot counter
    if action_idx in ROTATION_ACTIONS and not any_sensor:
        pure_rot_count += 1
    else:
        pure_rot_count = 0

    # Override 2: pure-rotation loop breaker
    if pure_rot_count >= MAX_PURE_ROT:
        pure_rot_count = 0
        return 2, pure_rot_count   # FW

    # Override 1: bearing-toward-box during blink
    if (not any_sensor
            and tracker.has_estimate
            and not tracker.in_push
            and tracker.box_steps_since < MAX_BOX_CONF):
        bearing = tracker.bearing_to_box_deg()
        if abs(bearing) < 45.0:   # box is roughly ahead → go forward
            pure_rot_count = 0
            return 2, pure_rot_count  # FW

    return action_idx, pure_rot_count


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────
def make_ckpt(online_net, total_steps, best_mean):
    return {
        "online_state_dict": online_net.state_dict(),
        "frame_stack":       FRAME_STACK,
        "obs_dim":           OBS_DIM,
        "n_actions":         N_ACTIONS,
        "aug_dim":           AUG_DIM,
        "total_steps":       total_steps,
        "best_mean":         best_mean,
    }

def save_ckpt(ckpt, path):
    torch.save(ckpt, path)

def periodic_path(base_path, ep):
    stem = base_path.replace(".pth", "")
    return f"{stem}_ep{ep}.pth"

def second_path(base_path):
    return base_path.replace(".pth", "_second.pth")


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────
def train(args):
    device = torch.device("cpu")

    online_net = DuelingDDQN(AUG_DIM, N_ACTIONS).to(device)
    target_net = DuelingDDQN(AUG_DIM, N_ACTIONS).to(device)
    target_net.eval()

    optimizer = optim.Adam(online_net.parameters(), lr=LR)
    replay    = PrioritizedReplayBuffer(REPLAY_SIZE)
    shaper    = RewardShaper()
    obs_bld   = ObsBuilder()
    tracker   = CoordinateTracker()

    # ── Resume ───────────────────────────────────────────────────────────────
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        online_net.load_state_dict(ckpt["online_state_dict"])
        target_net.load_state_dict(ckpt["online_state_dict"])
        total_steps = ckpt.get("total_steps", 0)
        best_mean   = ckpt.get("best_mean", -np.inf)
        print(f"  Resumed from '{args.resume}'")
        print(f"  total_steps={total_steps:,}  best_mean={best_mean:.1f}")
        eps_now = max(EPS_END, EPS_START - (EPS_START - EPS_END) * total_steps / EPS_DECAY_STEPS)
        print(f"  Epsilon continues from ~{eps_now:.3f}")
    else:
        if args.resume:
            print(f"  Resume file '{args.resume}' not found — starting fresh.")
        total_steps = 0
        best_mean   = -np.inf
        target_net.load_state_dict(online_net.state_dict())

    second_best_mean = -np.inf

    env = OBELIX(
        scaling_factor=5, arena_size=500, max_steps=args.max_steps,
        wall_obstacles=True, difficulty=1, box_speed=2, seed=args.seed,
    )

    raw_history = []

    print("=" * 76)
    print(f"  Dueling DDQN v2  |  Phase 2  difficulty=1  wall=True")
    print(f"  CoordTracker + CircSpinPenalty + BearingBonus + TimeoutPenalty")
    print(f"  AUG_DIM={AUG_DIM}  Episodes={args.episodes}  max_steps={args.max_steps}  seed={args.seed}")
    print("=" * 76)
    print(f"{'Ep':>6}  {'RawRew':>9}  {'Mean100':>9}  {'Eps':>6}  {'Steps':>10}  Status")
    print("-" * 76)

    for ep in range(1, args.episodes + 1):
        raw_obs = env.reset(seed=args.seed + ep)
        tracker.reset()
        state   = obs_bld.reset(raw_obs, tracker)
        shaper.reset()

        ep_raw       = 0.0
        done         = False
        won          = False
        pure_rot_cnt = 0

        ep_steps = 0


        while not done:
            ep_steps += 1
            eps = max(EPS_END,
                      EPS_START - (EPS_START - EPS_END) * total_steps / EPS_DECAY_STEPS)

            if random.random() < eps:
                action_idx = random.randrange(N_ACTIONS)
            else:
                with torch.no_grad():
                    q = online_net(torch.FloatTensor(state).unsqueeze(0))
                    action_idx = int(q.argmax(1).item())

            # Heuristic override (navigation assist)
            action_idx, pure_rot_cnt = heuristic_override(
                action_idx, raw_obs, tracker, pure_rot_cnt)

            # Record distance to box BEFORE action (for bearing improvement)
            prev_dist = tracker.get_distance_to_box()

            next_raw_obs, raw_reward, done = env.step(ACTIONS[action_idx], render=False)

            # Update tracker: action first (moves robot), then obs (estimates box)
            tracker.update_action(ACTIONS[action_idx], stuck_flag=bool(next_raw_obs[17]))
            tracker.update_obs(next_raw_obs)

            next_state = obs_bld.step(next_raw_obs, action_idx, tracker)

            if done and ep_steps < args.max_steps:
                won = True

            shaped_r = shaper.shape(
                raw_reward, action_idx, raw_obs, next_raw_obs,
                tracker, prev_dist, done, won, args.max_steps
            )

            raw_obs = next_raw_obs

            replay.push((state, action_idx, shaped_r, next_state, float(done)))
            state    = next_state
            ep_raw  += raw_reward
            total_steps += 1

            # ── Train ─────────────────────────────────────────────────────────
            if len(replay) >= MIN_REPLAY:
                beta    = min(1.0, 0.4 + 0.6 * total_steps / (args.episodes * args.max_steps))
                samples, idxs, weights = replay.sample(BATCH_SIZE, beta)
                weights = weights.to(device)

                s_b, a_b, r_b, ns_b, d_b = zip(*samples)
                s_t  = torch.FloatTensor(np.array(s_b)).to(device)
                ns_t = torch.FloatTensor(np.array(ns_b)).to(device)
                a_t  = torch.LongTensor(a_b).to(device)
                r_t  = torch.FloatTensor(r_b).to(device)
                d_t  = torch.FloatTensor(d_b).to(device)

                with torch.no_grad():
                    best_a = online_net(ns_t).argmax(1)
                    tgt_q  = target_net(ns_t).gather(1, best_a.unsqueeze(1)).squeeze(1)
                    y      = r_t + GAMMA * tgt_q * (1 - d_t)

                pred_q    = online_net(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)
                td_errors = (y - pred_q).detach().cpu().numpy()
                loss      = (weights * (y - pred_q).pow(2)).mean()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(online_net.parameters(), CLIP_GRAD)
                optimizer.step()
                replay.update_priorities(idxs, td_errors)

            if total_steps % TARGET_UPDATE == 0:
                target_net.load_state_dict(online_net.state_dict())

        raw_history.append(ep_raw)
        mean100 = float(np.mean(raw_history[-100:]))
        eps_now = max(EPS_END, EPS_START - (EPS_START - EPS_END) * total_steps / EPS_DECAY_STEPS)
        status  = "WIN" if won else "   "

        print(f"{ep:>6}  {ep_raw:>9.1f}  {mean100:>9.1f}  {eps_now:>6.3f}  {total_steps:>10,}  {'WIN' if won else 'LOSS'}")

        # ── Checkpoint saving ─────────────────────────────────────────────────

        # Best checkpoint
        if mean100 > best_mean and ep >= 10:
            # Previous best becomes second-best candidate
            if best_mean > second_best_mean and best_mean > -np.inf:
                second_best_mean = best_mean
                save_ckpt(make_ckpt(online_net, total_steps, second_best_mean),
                          second_path(args.save))
                print(f"         └─ 2nd-best updated mean={second_best_mean:.1f}  → {second_path(args.save)}")
            best_mean = mean100
            save_ckpt(make_ckpt(online_net, total_steps, best_mean), args.save)
            print(f"         └─ Best mean={best_mean:.1f}  saved → {args.save}")

        # Second-best checkpoint (if distinct improvement below current best)
        elif mean100 > second_best_mean and mean100 < best_mean and ep >= 10:
            second_best_mean = mean100
            save_ckpt(make_ckpt(online_net, total_steps, second_best_mean),
                      second_path(args.save))
            print(f"         └─ 2nd-best mean={second_best_mean:.1f}  → {second_path(args.save)}")

        # Periodic checkpoint
        if ep % PERIODIC_SAVE_EVERY == 0:
            ppath = periodic_path(args.save, ep)
            save_ckpt(make_ckpt(online_net, total_steps, mean100), ppath)
            print(f"         └─ Periodic ep={ep}  mean={mean100:.1f}  → {ppath}")

    print("=" * 76)
    print(f"  Done.  Best 100-ep mean (raw): {best_mean:.1f}")
    print("=" * 76)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes",  type=int,  default=4000)
    parser.add_argument("--max_steps", type=int,  default=2000)
    parser.add_argument("--seed",      type=int,  default=0)
    parser.add_argument("--save",      type=str,  default="weights_p2v2.pth")
    parser.add_argument("--resume",    type=str,  default=None,
                        help="Path to checkpoint .pth to resume from")
    args = parser.parse_args()
    train(args)