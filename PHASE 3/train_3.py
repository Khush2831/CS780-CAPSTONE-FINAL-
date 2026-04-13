#!/usr/bin/env python3
"""
================================================================
train_combined.py  –  OBELIX  |  Dueling DDQN + PER + Curriculum
================================================================

Combines the best of train_2.py with targeted fixes for the
wall-stuck problem.

WHAT IS KEPT FROM train_2.py (unchanged)
─────────────────────────────────────────
• Frame stack  (8 frames × 23 dims = 184 dims temporal window)
• Dead reckoning  (sin/cos heading, dark_steps, push_inferred,
                   wall_dist, near_wall)
• Prioritized Experience Replay  (SumTree + PER, alpha=0.6)
• Curriculum  (diff 0→2→3 with probabilistic mixing)
• State-dependent biased exploration  (explore_probs)
• Anti-spin reward  (−200 for 360° without FW)
• Ghost-box reward shaping  (track box during blink)
• Boundary progress reward  (post-attach, weight 3.0)
• Distance progress reward  (pre-attach, weight 2.0)
• Wall crossing bonus  (+1000)
• IR+FW synergy reward  (+5)
• Terminal shaping  (+1500+speed on win, −2×max_steps on timeout)
• Trajectory prioritization  (golden replay for wins)
• Per-episode seed variation
• Micro-loop detection  (−10 if position spread < 60px over 32 steps)

THREE KEY ADDITIONS (fix for wall-stuck issue)
──────────────────────────────────────────────
1. ACTION MASKING in greedy policy:
   When obs[17]=1 (stuck), FW is forcibly masked out of Q-values
   → network CANNOT pick FW when already stuck, ever.
   Applied identically in training and inference.

2. STUCK-AFTER-FW PENALTY in shaped reward:
   When the agent picks FW AND obs2[17]=1 (motion failed):
     custom_r -= 400   (stacks with env's -200 = -600 total signal)
   This makes the wall-collision signal 3× stronger, so the network
   learns wall avoidance faster.

3. prev_was_fw AS EXTRA FEATURE (7th dead-reckoning dim):
   Tells the network explicitly whether the last action was FW.
   Combined with obs[17], the network can recognise the exact
   "stuck-after-FW" state and learn to avoid it.
   STATE_DIM is now 191 (was 190).

Usage
─────
    python train_combined.py --episodes 5000 --wall_prob 0.75

Resume from checkpoint:
    python train_combined.py --episodes 5000 --resume --resume_from weights_best.pth
================================================================
"""

from __future__ import annotations
import argparse
import importlib.util
import math
import os
import random
from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

torch.set_num_threads(os.cpu_count() or 4)

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
ACTIONS      = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS    = 5
OBS_DIM      = 18          # raw sensor bits
ACT_DIM      = 5           # one-hot encoded previous action
EXTRA_DIM    = 7           # dead-reckoning + prev_was_fw  (was 6)
FRAME_STACK  = 8           # temporal window
PER_STEP     = OBS_DIM + ACT_DIM   # 23 dims per stack frame
STACK_DIM    = FRAME_STACK * PER_STEP  # 184
STATE_DIM    = STACK_DIM + EXTRA_DIM   # 191

ROT_DELTA    = {"L45": 45.0, "L22": 22.5, "FW": 0.0,
                "R22": -22.5, "R45": -45.0}
FW_STEP_PX   = 5.0
TRAIN_FREQ   = 4

FW_IDX       = ACTIONS.index("FW")   # = 2


# ─────────────────────────────────────────────────────────────
# DEAD RECKONING  (7 features, used identically train/inference)
# ─────────────────────────────────────────────────────────────
class DeadReckoning:
    """
    Tracks estimated bot position / heading from actions alone.
    Feature 6 (new): prev_was_fw — was the last action FW?
    This lets the network identify the stuck-after-FW danger state.
    """
    def __init__(self, x: float = 250.0, y: float = 250.0,
                 angle: float = 0.0, arena: int = 500):
        self.x     = x
        self.y     = y
        self.angle = angle
        self.arena = arena
        self._lo   = 40.0
        self._hi   = float(arena) - 40.0
        self._prev_was_fw = False

    def reset(self, x: float, y: float, angle: float):
        self.x, self.y, self.angle = x, y, angle
        self._prev_was_fw = False

    def update(self, action_str: str):
        self._prev_was_fw = (action_str == "FW")
        self.angle += ROT_DELTA[action_str]
        if action_str == "FW":
            rad    = math.radians(self.angle)
            self.x = float(np.clip(self.x + FW_STEP_PX * math.cos(rad), self._lo, self._hi))
            self.y = float(np.clip(self.y + FW_STEP_PX * math.sin(rad), self._lo, self._hi))

    def features(self, dark_steps: int, push_inferred: bool) -> np.ndarray:
        """
        7-dim feature vector (all in [0,1] or [-1,1]):
          0: sin(facing_angle)
          1: cos(facing_angle)
          2: dark_steps / FRAME_STACK
          3: push_inferred flag
          4: distance to nearest boundary / (arena/2)
          5: is_near_wall  (estimated x in [210,290])
          6: prev_was_fw   (NEW — enables stuck-after-FW recognition)
        """
        rad       = math.radians(self.angle)
        lo        = 10.0
        hi        = self.arena - 10.0
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
            float(self._prev_was_fw),   # NEW
        ], dtype=np.float32)


# ─────────────────────────────────────────────────────────────
# FRAME STACK
# ─────────────────────────────────────────────────────────────
class FrameStack:
    def __init__(self, k: int = FRAME_STACK):
        self.k   = k
        self.buf = np.zeros((k, PER_STEP), dtype=np.float32)

    def reset(self, obs: np.ndarray, action_idx: int = FW_IDX):
        oh       = np.zeros(ACT_DIM, dtype=np.float32)
        oh[action_idx] = 1.0
        frame    = np.concatenate([obs.astype(np.float32), oh])
        for i in range(self.k):
            self.buf[i] = frame

    def push(self, obs: np.ndarray, action_idx: int):
        oh       = np.zeros(ACT_DIM, dtype=np.float32)
        oh[action_idx] = 1.0
        frame    = np.concatenate([obs.astype(np.float32), oh])
        self.buf[:-1] = self.buf[1:]
        self.buf[-1]  = frame

    def flat(self) -> np.ndarray:
        return self.buf.flatten()


def build_state(fs: FrameStack, dr: DeadReckoning,
                dark_steps: int, push_inferred: bool) -> np.ndarray:
    return np.concatenate([fs.flat(), dr.features(dark_steps, push_inferred)])


# ─────────────────────────────────────────────────────────────
# ACTION MASKING  — THE KEY FIX
# ─────────────────────────────────────────────────────────────
def mask_q_values(q_vals: torch.Tensor, obs: np.ndarray,
                  prev_was_fw: bool) -> torch.Tensor:
    """
    When obs[17]=1 (stuck_flag) AND the previous action was FW,
    mask out FW from Q-values by setting Q[FW] = -inf.

    This mathematically prevents the network from ever selecting FW
    in the stuck-after-FW state — the most common cause of runaway
    -200/step chains.

    Called identically during training (greedy selection) and inference.
    """
    if bool(obs[17]) and prev_was_fw:
        q_vals = q_vals.clone()
        q_vals[:, FW_IDX] = float('-inf')
    return q_vals


# ─────────────────────────────────────────────────────────────
# NETWORK  (Dueling Double DQN)
# ─────────────────────────────────────────────────────────────
class DuelingDQN(nn.Module):
    def __init__(self, in_dim: int = STATE_DIM, n_actions: int = N_ACTIONS):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.feature(x)
        v = self.value(f)
        a = self.advantage(f)
        return v + (a - a.mean(dim=1, keepdim=True))


# ─────────────────────────────────────────────────────────────
# PRIORITISED REPLAY  (SumTree + PER)
# ─────────────────────────────────────────────────────────────
class SumTree:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree     = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.n        = 0
        self.write    = 0

    def _propagate(self, idx: int, change: float):
        while idx:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def update(self, idx: int, p: float):
        change         = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def add(self, p: float) -> int:
        leaf_idx = self.write + self.capacity - 1
        self.update(leaf_idx, p)
        self.write = (self.write + 1) % self.capacity
        self.n     = min(self.n + 1, self.capacity)
        return leaf_idx

    def _retrieve(self, s: float) -> int:
        idx = 0
        while idx < self.capacity - 1:
            l = 2 * idx + 1
            r = l + 1
            if s <= self.tree[l]:
                idx = l
            else:
                s  -= self.tree[l]
                idx = r
        return idx

    def sample(self, n: int):
        total   = self.tree[0]
        seg     = total / n
        pts     = np.random.uniform(seg * np.arange(n), seg * (np.arange(n) + 1))
        idxs    = np.array([self._retrieve(p) for p in pts], dtype=np.int32)
        prios   = self.tree[idxs]
        data_idxs = idxs - self.capacity + 1
        return idxs, prios, data_idxs

    @property
    def total_p(self): return float(self.tree[0])


class PrioritizedReplay:
    def __init__(self, cap: int = 300_000, alpha: float = 0.6):
        self.cap          = cap
        self.alpha        = alpha
        self.tree         = SumTree(cap)
        self.max_priority = 1.0

        self.s_buf  = np.zeros((cap, STATE_DIM), dtype=np.float32)
        self.a_buf  = np.zeros(cap,              dtype=np.int64)
        self.r_buf  = np.zeros(cap,              dtype=np.float32)
        self.s2_buf = np.zeros((cap, STATE_DIM), dtype=np.float32)
        self.d_buf  = np.zeros(cap,              dtype=np.float32)

    def add(self, s, a, r, s2, done) -> int:
        idx              = self.tree.write
        self.s_buf[idx]  = s
        self.a_buf[idx]  = a
        self.r_buf[idx]  = r
        self.s2_buf[idx] = s2
        self.d_buf[idx]  = done
        return self.tree.add(self.max_priority ** self.alpha)

    def sample(self, batch: int, beta: float = 0.4):
        idxs, prios, data_idxs = self.tree.sample(batch)
        probs   = prios / self.tree.total_p
        total   = self.tree.n
        weights = (total * probs) ** (-beta)
        weights = (weights / weights.max()).astype(np.float32)
        return (self.s_buf[data_idxs], self.a_buf[data_idxs],
                self.r_buf[data_idxs], self.s2_buf[data_idxs],
                self.d_buf[data_idxs], idxs, weights)

    def update_priorities(self, idxs, td_errors):
        for i, e in zip(idxs, td_errors):
            p = float(e) + 1e-5
            self.max_priority = max(self.max_priority, p)
            self.tree.update(i, p ** self.alpha)

    def __len__(self): return self.tree.n


# ─────────────────────────────────────────────────────────────
# CURRICULUM  &  SCHEDULING
# ─────────────────────────────────────────────────────────────
def curriculum_difficulty(ep: int, total: int) -> int:
    frac = ep / max(1, total)
    r    = random.random()
    if frac < 0.25:
        return 0
    elif frac < 0.50:
        return 0 if r < 0.60 else 2
    elif frac < 0.75:
        return 0 if r < 0.30 else (2 if r < 0.60 else 3)
    else:
        return 0 if r < 0.10 else (2 if r < 0.30 else 3)


def epsilon_schedule(ep: int, total: int,
                     eps_start: float = 1.0,
                     eps_end:   float = 0.05,
                     decay_frac: float = 0.70) -> float:
    decay_eps = int(total * decay_frac)
    if ep >= decay_eps:
        return eps_end
    return eps_start + (eps_end - eps_start) * ep / decay_eps


def explore_probs(obs: np.ndarray, prev_was_fw: bool) -> np.ndarray:
    """
    State-dependent biased exploration.
    FW is never chosen during exploration when stuck (obs[17]=1),
    matching the action masking applied during greedy exploitation.
    """
    if obs[17]:                          # stuck: no FW
        return np.array([0.25, 0.25, 0.00, 0.25, 0.25])
    any_sensor = bool(np.any(obs[:17]))
    if any_sensor:
        return np.array([0.05, 0.10, 0.70, 0.10, 0.05])
    return    np.array([0.10, 0.15, 0.50, 0.15, 0.10])


# ─────────────────────────────────────────────────────────────
# ENV LOADER
# ─────────────────────────────────────────────────────────────
def load_obelix(path: str):
    spec = importlib.util.spec_from_file_location("obelix_env", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="OBELIX Combined Trainer")
    ap.add_argument("--obelix_py",      default="obelix.py")
    ap.add_argument("--out",            default="weights_combined.pth")
    ap.add_argument("--episodes",       type=int,   default=5000)
    ap.add_argument("--max_steps",      type=int,   default=1500)
    ap.add_argument("--scaling_factor", type=int,   default=5)
    ap.add_argument("--arena_size",     type=int,   default=500)
    ap.add_argument("--box_speed",      type=int,   default=2)
    ap.add_argument("--gamma",          type=float, default=0.99)
    ap.add_argument("--lr",             type=float, default=2e-4)
    ap.add_argument("--batch",          type=int,   default=512)
    ap.add_argument("--replay_cap",     type=int,   default=300_000)
    ap.add_argument("--warmup",         type=int,   default=10_000)
    ap.add_argument("--target_sync",    type=int,   default=3_000)
    ap.add_argument("--eps_start",      type=float, default=1.0)
    ap.add_argument("--eps_end",        type=float, default=0.05)
    ap.add_argument("--seed",           type=int,   default=42)
    ap.add_argument("--wall_prob",      type=float, default=0.75)
    ap.add_argument("--render",         action="store_true")
    ap.add_argument("--resume",         action="store_true")
    ap.add_argument("--resume_from",    default=None)
    ap.add_argument("--eval_every",     type=int,   default=25)
    ap.add_argument("--save_every",     type=int,   default=100)
    # Stuck penalty scale (multiplier on top of env's -200)
    ap.add_argument("--stuck_penalty",  type=float, default=400.0,
                    help="Extra shaped penalty when FW leads to stuck (on top of env -200)")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    OBELIX = load_obelix(args.obelix_py)

    q   = DuelingDQN()
    tgt = DuelingDQN()

    resume_path = args.resume_from or args.out
    if args.resume and os.path.exists(resume_path):
        print(f"[RESUME] Loading from {resume_path}")
        q.load_state_dict(torch.load(resume_path, map_location="cpu", weights_only=True))

    tgt.load_state_dict(q.state_dict())
    tgt.eval()

    opt    = optim.Adam(q.parameters(), lr=args.lr)
    replay = PrioritizedReplay(cap=args.replay_cap, alpha=0.6)

    total_steps = 0
    best_avg    = -float("inf")
    second_avg  = -float("inf")
    recent_ret  = deque(maxlen=50)
    recent_wins = deque(maxlen=50)

    out_base    = args.out.replace(".pth", "")
    best_path   = out_base + "_best.pth"
    second_path = out_base + "_2nd.pth"

    print(f"STATE_DIM={STATE_DIM}  |  replay_cap={args.replay_cap}")
    print(f"Training {args.episodes} episodes | max_steps={args.max_steps}")
    print(f"Curriculum diff 0→2→3 | wall_prob={args.wall_prob}")
    print(f"Action masking: FW masked when stuck_flag=1 + prev_was_fw=True")
    print(f"Extra stuck penalty: -{args.stuck_penalty} (total -{200+args.stuck_penalty}/event)")

    for ep in range(args.episodes):
        is_eval     = (ep > 0 and ep % args.eval_every == 0)
        current_eps = 0.05 if is_eval else epsilon_schedule(
            ep, args.episodes, args.eps_start, args.eps_end)

        difficulty = curriculum_difficulty(ep, args.episodes)
        use_walls  = (random.random() < args.wall_prob)
        ep_seed    = (args.seed + ep * 7919) % (2 ** 31)

        base_env = OBELIX(
            scaling_factor  = args.scaling_factor,
            arena_size      = args.arena_size,
            max_steps       = args.max_steps,
            wall_obstacles  = use_walls,
            difficulty      = difficulty,
            box_speed       = args.box_speed,
            seed            = ep_seed,
        )
        obs = base_env.sensor_feedback.copy()

        dr = DeadReckoning(
            x     = float(base_env.bot_center_x),
            y     = float(base_env.bot_center_y),
            angle = float(base_env.facing_angle),
            arena = args.arena_size,
        )
        fs = FrameStack(FRAME_STACK)
        fs.reset(obs)

        dark_steps    = 0
        push_inferred = False
        rot_accum     = 0.0
        prev_was_fw   = False          # tracks last action for masking + feature

        ghost_x = float(base_env.box_center_x)
        ghost_y = float(base_env.box_center_y)
        prev_ghost_dist = math.hypot(
            base_env.bot_center_x - ghost_x,
            base_env.bot_center_y - ghost_y)

        prev_dist = math.hypot(
            float(base_env.bot_center_x) - float(base_env.box_center_x),
            float(base_env.bot_center_y) - float(base_env.box_center_y))
        prev_bound_dist: Optional[float] = None

        start_x       = float(base_env.bot_center_x)
        started_opp   = ((start_x < 250 and base_env.box_center_x > 250) or
                         (start_x > 250 and base_env.box_center_x < 250))
        crossed_wall  = False
        claimed_cross = False

        hist_pos  = np.zeros((32, 2), dtype=np.float32)
        hist_idx  = 0
        ep_addrs  = []

        s       = build_state(fs, dr, dark_steps, push_inferred)
        ep_ret  = 0.0
        ep_step = 0
        is_success = False

        for _ in range(args.max_steps):
            # ── Action selection ──────────────────────────────────
            if random.random() < current_eps:
                # Biased exploration — never FW when stuck
                probs = explore_probs(obs, prev_was_fw)
                a     = int(np.random.choice(N_ACTIONS, p=probs))
            else:
                # Greedy with action masking — never FW when stuck+prev_FW
                with torch.no_grad():
                    s_t    = torch.as_tensor(s, dtype=torch.float32).unsqueeze(0)
                    q_vals = q(s_t)
                    q_vals = mask_q_values(q_vals, obs, prev_was_fw)
                    a      = int(q_vals.argmax(dim=1).item())

            action_str  = ACTIONS[a]

            # ── Update dead reckoning (before env step) ───────────
            dr.update(action_str)

            # ── Anti-spin accumulator ─────────────────────────────
            if action_str == "FW":
                rot_accum = 0.0
            else:
                rot_accum += ROT_DELTA[action_str]

            # ── Env step ──────────────────────────────────────────
            obs2, env_r, done = base_env.step(action_str, render=args.render)

            # ── Tracking ──────────────────────────────────────────
            any_sensor2 = bool(np.any(obs2[:17]))
            dark_steps  = 0 if any_sensor2 else dark_steps + 1

            if obs2[16]:
                push_inferred = True

            bot_x  = float(base_env.bot_center_x)
            bot_y  = float(base_env.bot_center_y)
            box_x  = float(base_env.box_center_x)
            box_y  = float(base_env.box_center_y)
            curr_d = math.hypot(bot_x - box_x, bot_y - box_y)

            if base_env.box_visible:
                ghost_x, ghost_y = box_x, box_y
            elif base_env.box_move_enabled:
                ghost_x = float(np.clip(ghost_x + base_env._box_vx,
                                         40, args.arena_size - 40))
                ghost_y = float(np.clip(ghost_y + base_env._box_vy,
                                         40, args.arena_size - 40))

            # ── Shaped reward ─────────────────────────────────────
            custom_r = float(env_r)

            # 1. Distance progress toward box (pre-attach)
            if not base_env.enable_push:
                if base_env.box_visible:
                    custom_r += (prev_dist - curr_d) * 2.0
                else:
                    ghost_d          = math.hypot(bot_x - ghost_x, bot_y - ghost_y)
                    custom_r        += (prev_ghost_dist - ghost_d) * 0.8
                    prev_ghost_dist  = ghost_d
                prev_dist = curr_d

            # 2. Boundary progress (post-attach)
            if base_env.enable_push:
                lo, hi   = 10.0, args.arena_size - 10.0
                bound_d  = min(box_x - lo, hi - box_x, box_y - lo, hi - box_y)
                if prev_bound_dist is not None:
                    custom_r += (prev_bound_dist - bound_d) * 3.0
                prev_bound_dist = bound_d

            # 3. Anti-spin
            if abs(rot_accum) >= 360.0:
                custom_r  -= 200.0
                rot_accum  = 0.0

            # 4. Micro-loop detection
            hist_pos[hist_idx % 32] = [bot_x, bot_y]
            hist_idx += 1
            if hist_idx >= 32:
                spread = np.ptp(hist_pos, axis=0)
                if spread[0] < 60.0 and spread[1] < 60.0:
                    custom_r -= 10.0

            # 5. Wall crossing bonus (one-time)
            if use_walls and started_opp and not crossed_wall:
                if start_x < 250 and bot_x > 290:
                    crossed_wall = True
                elif start_x > 250 and bot_x < 210:
                    crossed_wall = True
            if crossed_wall and not claimed_cross:
                custom_r     += 1000.0
                claimed_cross = True

            # 6. IR + FW synergy
            if obs2[16] and action_str == "FW":
                custom_r += 5.0

            # ── KEY FIX 2: STUCK-AFTER-FW EXTRA PENALTY ──────────
            # If we picked FW and the environment says we're stuck,
            # add a large extra penalty so the network strongly learns
            # that FW into walls is catastrophic.
            # This stacks with env's -200: total signal = -(200 + stuck_penalty)
            if action_str == "FW" and bool(obs2[17]):
                custom_r -= args.stuck_penalty

            # 7. Terminal shaping
            if done:
                ep_step_final = ep_step + 1
                if ep_step_final < args.max_steps:
                    is_success  = True
                    speed_bonus = max(0.0, 500.0 * (1.0 - ep_step_final / args.max_steps))
                    custom_r   += 1500.0 + speed_bonus
                else:
                    custom_r   -= 2.0 * args.max_steps

            # ── Build next state ──────────────────────────────────
            fs.push(obs2, a)
            s2 = build_state(fs, dr, dark_steps, push_inferred)

            # ── Store transition ──────────────────────────────────
            if not is_eval:
                leaf = replay.add(s, a, custom_r, s2, float(done))
                ep_addrs.append(leaf)
                total_steps += 1

                # ── Learn ──────────────────────────────────────
                if (total_steps % TRAIN_FREQ == 0 and
                        len(replay) >= max(args.warmup, args.batch)):
                    beta = min(1.0, 0.4 + total_steps * 0.6 / 3_000_000)
                    sb, ab, rb, s2b, db, t_idxs, ws = replay.sample(args.batch, beta)

                    sb_t  = torch.as_tensor(sb,  dtype=torch.float32)
                    ab_t  = torch.as_tensor(ab,  dtype=torch.int64)
                    rb_t  = torch.as_tensor(rb,  dtype=torch.float32)
                    s2b_t = torch.as_tensor(s2b, dtype=torch.float32)
                    db_t  = torch.as_tensor(db,  dtype=torch.float32)
                    ws_t  = torch.as_tensor(ws,  dtype=torch.float32)

                    with torch.no_grad():
                        best_a  = q(s2b_t).argmax(dim=1)
                        q_next  = tgt(s2b_t).gather(1, best_a.unsqueeze(1)).squeeze(1)
                        targets = rb_t + args.gamma * (1.0 - db_t) * q_next

                    preds  = q(sb_t).gather(1, ab_t.unsqueeze(1)).squeeze(1)
                    td_err = (preds - targets).abs().detach().cpu().numpy()
                    replay.update_priorities(t_idxs, td_err)

                    loss = (ws_t * nn.functional.smooth_l1_loss(
                        preds, targets, reduction="none")).mean()
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(q.parameters(), 5.0)
                    opt.step()

                    if total_steps % args.target_sync == 0:
                        tgt.load_state_dict(q.state_dict())

            # ── Advance ───────────────────────────────────────────
            prev_was_fw = (action_str == "FW")
            s           = s2
            obs         = obs2
            ep_ret     += custom_r
            ep_step    += 1
            if done:
                break

        # ── POST-EPISODE ──────────────────────────────────────────
        if is_eval:
            lvl = f"diff={difficulty} wall={'ON' if use_walls else 'OFF'}"
            print(f"  >>> [EVAL] [{lvl}] ret={ep_ret:.1f} steps={ep_step} "
                  f"success={is_success} <<<")
        else:
            recent_ret.append(ep_ret)
            recent_wins.append(int(is_success))
            avg50    = float(np.mean(recent_ret))
            win_rate = 100.0 * float(np.mean(recent_wins))

            # Trajectory prioritization (win episodes replayed more)
            multiplier = 1.0; tier = ""
            if is_success:
                speed = max(0.5, min(1.0, 500.0 / max(1, ep_step)))
                if difficulty == 3:
                    multiplier = max(10.0, 50.0 * speed); tier = f"Phase3-WIN|{ep_step}s"
                elif difficulty == 2:
                    multiplier = max(5.0, 30.0 * speed);  tier = f"Phase2-WIN|{ep_step}s"
                else:
                    multiplier = max(3.0, 15.0 * speed);  tier = f"Phase1-WIN|{ep_step}s"
                if use_walls and started_opp and crossed_wall:
                    multiplier *= 2.0
            if multiplier > 1.0 and ep_addrs:
                boosted = (replay.max_priority * multiplier) ** replay.alpha
                for addr in ep_addrs:
                    replay.tree.update(addr, boosted)
                print(f"   --> GOLDEN [{tier}] priority x{multiplier:.1f}")

            lvl = f"diff={difficulty} wall={'ON' if use_walls else 'OFF'}"
            print(f"Ep {ep+1}/{args.episodes} [{lvl}] | "
                  f"ret={ep_ret:.1f} avg={avg50:.1f} win={win_rate:.1f}% | "
                  f"eps={current_eps:.3f} steps={ep_step} | "
                  f"total_steps={total_steps}")

            if ep >= 50:
                if avg50 > best_avg:
                    if os.path.exists(best_path) and avg50 > second_avg:
                        import shutil; shutil.copy2(best_path, second_path)
                        second_avg = best_avg
                    best_avg = avg50
                    torch.save(q.state_dict(), best_path)
                    print(f"   --> NEW BEST {best_avg:.1f}! -> {best_path}")
                elif avg50 > second_avg:
                    second_avg = avg50
                    torch.save(q.state_dict(), second_path)
                    print(f"   --> 2nd BEST {second_avg:.1f}! -> {second_path}")

            if (ep + 1) % args.save_every == 0:
                torch.save(q.state_dict(), args.out)
                print(f"   --> Checkpoint -> {args.out}")

            if (ep + 1) % 200 == 0:
                snap = out_base + f"_ep{ep+1}.pth"
                torch.save(q.state_dict(), snap)
                print(f"   --> Snapshot -> {snap}")

    torch.save(q.state_dict(), args.out)
    print(f"\nDone. Final -> {args.out}")
    print(f"Best avg: {best_avg:.1f}  |  2nd best: {second_avg:.1f}")


if __name__ == "__main__":
    main()