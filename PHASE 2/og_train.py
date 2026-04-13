"""
train_phase2.py  –  Dueling DDQN  |  Phase 2: Blinking Box + Wall
==================================================================
Run (fresh):
    python train_phase2.py --episodes 5000 --save weights_phase2.pth

Run (resume):
    python train_phase2.py --episodes 5000 --resume weights_phase2.pth --save weights_phase2.pth

PHASE 2 KEY CHALLENGE: Blinking Box
-------------------------------------
The box randomly appears and disappears. Sensor silence now has TWO causes:
  (a) Robot is facing the wrong direction / box is far  → should rotate to search
  (b) Box blinked out temporarily                       → should keep moving, do NOT rotate

A single observation cannot distinguish these. Solution:

  blink_memory  – steps since any sonar sensor last fired, normalized over
                  BLINK_CAP steps. When blink_memory is LOW (box was seen
                  recently), silence is likely a blink → persevere forward.
                  When blink_memory is HIGH, silence is genuine → explore.

  This is the only new feature added vs Phase 1 (AUG_DIM: 80 → 81).

Reward shaping changes vs Phase 1:
  - Rotation loop penalty: SUPPRESSED when blink_memory < BLINK_PATIENCE
    (box recently seen → silence is likely a blink, penalising rotation
    would punish the correct "wait and persist" behaviour)
  - BLINK_PERSIST_BONUS: small reward for moving FW while sensors silent
    but box was recently seen (blink_memory < BLINK_PATIENCE)
    → teaches agent to keep heading toward last known box location

All Phase 1 features retained unchanged:
  recent_stuck_count, ir_streak, frame_stack, wall_confusion_penalty, etc.

AUG_DIM = 72 (stacked) + 5 (last-act) + 1 (rot_streak)
                       + 1 (recent_stuck) + 1 (ir_streak) + 1 (blink_memory) = 81
"""

import argparse
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
ROTATION_ACTIONS = {0, 1, 3, 4}

FRAME_STACK = 4
OBS_DIM     = 18
STACK_DIM   = FRAME_STACK * OBS_DIM          # 72
#                       stacked  last-act  rot_streak  recent_stuck  ir_streak  blink_memory
AUG_DIM     = STACK_DIM + N_ACTIONS + 1 + 1 + 1 + 1                            # = 81

BLINK_CAP       = 30    # steps: normalisation cap for blink_memory
BLINK_PATIENCE  = 15    # steps: below this, silence is likely a blink

GAMMA           = 0.97
LR              = 3e-4
BATCH_SIZE      = 128
REPLAY_SIZE     = 100_000
MIN_REPLAY      = 2_000
TARGET_UPDATE   = 500
CLIP_GRAD       = 1.0

EPS_START        = 1.0
EPS_END          = 0.05
EPS_DECAY_STEPS  = 120_000   # longer than Phase 1: blinking makes exploration harder

# ── Reward shaping coefficients ──────────────────────────────────────────────
ESCAPE_BONUS           =  15.0   # rotation that un-stuck robot
SENSOR_PROGRESS_BONUS  =   2.0   # new sensor bit after rotation
FORWARD_MOMENTUM_BONUS =   1.0   # successful FW while sensors active
ROTATION_LOOP_K        =   8
ROTATION_LOOP_PENALTY  =  -8.0   # K rotations in a row, genuinely blind (not blink)
WALL_CONFUSION_PENALTY = -20.0   # FW into wall while sonar active but no IR
IR_BONUS               =  10.0   # first IR activation this episode
BLINK_PERSIST_BONUS    =   1.5   # FW while sensors silent but box recently seen (blink)


# ─────────────────────────────────────────────────────────────────────────────
# Dueling DDQN  (unchanged from Phase 1)
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
# Prioritized Replay Buffer  (unchanged from Phase 1)
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
# Reward Shaper  (Phase 2: blink-aware)
# ─────────────────────────────────────────────────────────────────────────────
class RewardShaper:
    def __init__(self):
        self.prev_stuck    = False
        self.prev_sensor   = np.zeros(17, dtype=bool)
        self.action_hist   = deque(maxlen=ROTATION_LOOP_K)
        self.ir_given      = False
        self.blink_steps   = 0    # steps since last sonar activation

    def reset(self):
        self.prev_stuck  = False
        self.prev_sensor = np.zeros(17, dtype=bool)
        self.action_hist.clear()
        self.ir_given    = False
        self.blink_steps = 0

    def shape(self, raw, action_idx, obs, next_obs):
        shaped     = raw
        next_stuck = bool(next_obs[17])
        curr_bits  = obs[:17].astype(bool)
        next_bits  = next_obs[:17].astype(bool)
        ir_now     = bool(next_obs[16])
        any_sonar  = bool(np.any(next_obs[:16]))

        # Track blink_steps: steps since any sonar last fired
        if any_sonar:
            self.blink_steps = 0
        else:
            self.blink_steps = min(self.blink_steps + 1, BLINK_CAP)

        likely_blink = (self.blink_steps < BLINK_PATIENCE)

        # 1. Escape bonus: rotation that freed the robot from stuck
        if self.prev_stuck and action_idx in ROTATION_ACTIONS and not next_stuck:
            shaped += ESCAPE_BONUS

        # 2. Sensor progress after rotation
        if action_idx in ROTATION_ACTIONS:
            n_new = int((next_bits & ~curr_bits).sum())
            if n_new > 0:
                shaped += SENSOR_PROGRESS_BONUS * n_new

        # 3. Forward momentum: successful FW while sensors active
        if action_idx == 2 and not next_stuck and np.any(curr_bits):
            shaped += FORWARD_MOMENTUM_BONUS

        # 4. Rotation loop penalty — SUPPRESSED during likely blink silence.
        #    If box was recently seen, sensor silence is probably a blink.
        #    Penalising rotation here would punish correct "persist" behaviour.
        self.action_hist.append(action_idx)
        if (len(self.action_hist) == ROTATION_LOOP_K
                and all(a in ROTATION_ACTIONS for a in self.action_hist)
                and not np.any(next_bits)
                and not likely_blink):          # ← Phase 2 gate
            shaped += ROTATION_LOOP_PENALTY

        # 5. Wall confusion penalty (unchanged from Phase 1)
        if action_idx == 2 and next_stuck and any_sonar and not ir_now:
            shaped += WALL_CONFUSION_PENALTY

        # 6. IR bonus: first IR activation this episode
        if ir_now and not self.ir_given:
            shaped += IR_BONUS
            self.ir_given = True

        # 7. Blink persistence bonus: FW while sensors silent but box recently seen.
        #    Teaches agent to keep heading toward last known box location during blink.
        if action_idx == 2 and not next_stuck and not any_sonar and likely_blink:
            shaped += BLINK_PERSIST_BONUS

        self.prev_stuck  = next_stuck
        self.prev_sensor = next_bits.copy()
        return shaped


# ─────────────────────────────────────────────────────────────────────────────
# Observation builder  (Phase 2: adds blink_memory)
# ─────────────────────────────────────────────────────────────────────────────
class ObsBuilder:
    def __init__(self):
        self.frames       = deque(maxlen=FRAME_STACK)
        self.last_act     = 2
        self.rot_streak   = 0
        self.stuck_window = deque(maxlen=10)
        self.ir_streak    = 0
        self.blink_steps  = 0    # steps since last sonar activation

    def reset(self, obs):
        self.frames.clear()
        for _ in range(FRAME_STACK):
            self.frames.append(obs.copy())
        self.last_act    = 2
        self.rot_streak  = 0
        self.stuck_window.clear()
        self.ir_streak   = 0
        self.blink_steps = 0
        return self._build()

    def step(self, obs, action_idx):
        self.frames.append(obs.copy())
        self.last_act = action_idx

        # Rotation streak
        if action_idx in ROTATION_ACTIONS:
            self.rot_streak = min(self.rot_streak + 1, 20)
        else:
            self.rot_streak = 0

        # Recent stuck window
        self.stuck_window.append(float(obs[17]))

        # IR streak
        if bool(obs[16]):
            self.ir_streak = min(self.ir_streak + 1, 10)
        else:
            self.ir_streak = 0

        # Blink memory: steps since any sonar fired
        if bool(np.any(obs[:16])):
            self.blink_steps = 0
        else:
            self.blink_steps = min(self.blink_steps + 1, BLINK_CAP)

        return self._build()

    def _build(self):
        stacked = np.concatenate(list(self.frames)).astype(np.float32)

        act_oh = np.zeros(N_ACTIONS, dtype=np.float32)
        act_oh[self.last_act] = 1.0

        rot   = np.array([self.rot_streak / 20.0],         dtype=np.float32)
        stuck = np.array([sum(self.stuck_window) / 10.0],  dtype=np.float32)
        ir_s  = np.array([self.ir_streak / 10.0],          dtype=np.float32)
        blink = np.array([self.blink_steps / BLINK_CAP],   dtype=np.float32)  # ← new

        return np.concatenate([stacked, act_oh, rot, stuck, ir_s, blink])


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

    # ── Resume from checkpoint ────────────────────────────────────────────────
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        online_net.load_state_dict(ckpt["online_state_dict"])
        target_net.load_state_dict(ckpt["online_state_dict"])
        total_steps = ckpt.get("total_steps", 0)
        best_mean   = ckpt.get("best_mean", -np.inf)
        print(f"  ✔ Resumed from '{args.resume}'")
        print(f"    total_steps={total_steps:,}  |  best_mean={best_mean:.1f}")
        print(f"    Epsilon will continue from ~{max(EPS_END, EPS_START - (EPS_START - EPS_END) * total_steps / EPS_DECAY_STEPS):.3f}")
    else:
        if args.resume:
            print(f"  ⚠ Resume file '{args.resume}' not found — starting fresh.")
        total_steps = 0
        best_mean   = -np.inf
        target_net.load_state_dict(online_net.state_dict())

    env = OBELIX(
        scaling_factor=5, arena_size=500, max_steps=args.max_steps,
        wall_obstacles=True, difficulty=2, box_speed=2, seed=args.seed,
    )

    raw_history = []

    print("=" * 72)
    print(f"  Dueling DDQN  |  Phase 2  difficulty=2 (blinking box)  wall=True")
    print(f"  New feature: blink_memory (distinguishes blink silence from wrong dir)")
    print(f"  state_dim={AUG_DIM}  Episodes={args.episodes}  max_steps={args.max_steps}  seed={args.seed}")
    print("=" * 72)
    print(f"{'Ep':>6}  {'RawRew':>9}  {'Mean100':>9}  {'Eps':>6}  {'Steps':>9}  Status")
    print("-" * 72)

    for ep in range(1, args.episodes + 1):
        raw_obs = env.reset(seed=args.seed + ep)
        state   = obs_bld.reset(raw_obs)
        shaper.reset()

        ep_raw = 0.0
        done   = False
        won    = False

        while not done:
            eps = max(EPS_END,
                      EPS_START - (EPS_START - EPS_END) * total_steps / EPS_DECAY_STEPS)
            if random.random() < eps:
                action_idx = random.randrange(N_ACTIONS)
            else:
                with torch.no_grad():
                    q = online_net(torch.FloatTensor(state).unsqueeze(0))
                    action_idx = int(q.argmax(1).item())

            next_raw_obs, raw_reward, done = env.step(ACTIONS[action_idx], render=False)
            next_state = obs_bld.step(next_raw_obs, action_idx)

            shaped_r = shaper.shape(raw_reward, action_idx, raw_obs, next_raw_obs)
            raw_obs  = next_raw_obs

            replay.push((state, action_idx, shaped_r, next_state, float(done)))
            state    = next_state
            ep_raw  += raw_reward
            total_steps += 1

            if raw_reward >= 1000:
                won = True

            # Train
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
        mean100 = np.mean(raw_history[-100:])
        eps_now = max(EPS_END, EPS_START - (EPS_START - EPS_END) * total_steps / EPS_DECAY_STEPS)
        status  = "WIN" if won else "   "

        print(f"{ep:>6}  {ep_raw:>9.1f}  {mean100:>9.1f}  {eps_now:>6.3f}  {total_steps:>9,}  {status}")

        if mean100 > best_mean and ep >= 10:
            best_mean = mean100
            torch.save({
                "online_state_dict": online_net.state_dict(),
                "frame_stack":       FRAME_STACK,
                "obs_dim":           OBS_DIM,
                "n_actions":         N_ACTIONS,
                "aug_dim":           AUG_DIM,
                "total_steps":       total_steps,
                "best_mean":         best_mean,
            }, args.save)
            print(f"         └─ New best mean={best_mean:.1f}  saved → {args.save}")

    print("=" * 72)
    print(f"  Done.  Best 100-ep mean (raw): {best_mean:.1f}")
    print("=" * 72)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes",  type=int,  default=3000)
    parser.add_argument("--max_steps", type=int,  default=2000)
    parser.add_argument("--seed",      type=int,  default=0)
    parser.add_argument("--save",      type=str,  default="weights_phase2.pth")
    parser.add_argument("--resume",    type=str,  default=None,
                        help="Path to a checkpoint .pth file to resume training from")
    args = parser.parse_args()
    train(args)