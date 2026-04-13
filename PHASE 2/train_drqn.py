"""
train_drqn_phase2.py  –  Dueling DRQN (GRU-based)  |  Phase 2: Blinking Box + Wall
=====================================================================================
Run (fresh):
    python train_drqn_phase2.py --episodes 4000 --save weights_drqn_phase2.pth

Run (resume):
    python train_drqn_phase2.py --episodes 4000 --resume weights_drqn_phase2.pth --save weights_drqn_phase2.pth

WHY DRQN OVER DDQN FOR PHASE 2?
---------------------------------
Phase 2 is a POMDP: the box blinks, so sensor silence has ambiguous meaning.
The DDQN + frame-stack approach patches this with hand-crafted features
(blink_memory scalar). But it cannot encode DIRECTION of last box sighting —
only THAT sensors were recently active.

A GRU hidden state carries information across the entire episode. It naturally
learns to remember: "box was to my left 12 steps ago" — not just "sensors
fired recently". This is the theoretically correct solution for POMDPs and
directly implements the DRQN paper (Hausknecht & Stone, 2015).

ARCHITECTURE
------------
Input per step: 18 (raw obs) + 5 (last-action one-hot) + 3 (rot_streak,
                stuck_ratio, ir_streak) = 26 dims
GRU(26 → 128):  hidden state h_t carries full episode history
Dueling heads:  Value(128→64→1), Advantage(128→64→5)
No frame stack, no blink_memory feature — GRU learns its own memory.

SEQUENCE REPLAY (DRQN training trick)
--------------------------------------
Standard replay buffers break temporal continuity — you can't backprop
through time if transitions are shuffled. Solution:

  1. Store full episodes in an EpisodeBuffer
  2. Sample fixed-length subsequences (SEQ_LEN = 16 steps)
  3. Always initialise GRU hidden state to zero at sequence start
     (zero-start approximation — simplest correct approach)
  4. Backprop through all 16 steps (BPTT)
  5. TD targets computed step-by-step within the sequence

REWARD SHAPING (same logic as Phase 2 DDQN, kept for consistency)
------------------------------------------------------------------
  - ESCAPE_BONUS, SENSOR_PROGRESS, FORWARD_MOMENTUM, WALL_CONFUSION_PENALTY
  - Rotation loop penalty suppressed during likely blink silence
  - BLINK_PERSIST_BONUS: FW while sensors silent but box recently seen
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

OBS_DIM      = 18
EXTRA_DIM    = N_ACTIONS + 3   # last-act one-hot + rot_streak + stuck_ratio + ir_streak
INPUT_DIM    = OBS_DIM + EXTRA_DIM   # = 26
GRU_HIDDEN   = 128
SEQ_LEN      = 16    # subsequence length for BPTT

GAMMA           = 0.97
LR              = 3e-4
BATCH_SIZE      = 32   # sequences, not transitions
MIN_EPISODES    = 100  # fill buffer before training starts
TARGET_UPDATE   = 200  # episodes
CLIP_GRAD       = 1.0
MAX_EP_LEN      = 2001 # safety cap for episode storage

EPS_START        = 1.0
EPS_END          = 0.05
EPS_DECAY_STEPS  = 120_000

BLINK_CAP        = 30
BLINK_PATIENCE   = 15

# ── Reward shaping coefficients ──────────────────────────────────────────────
ESCAPE_BONUS           =  15.0
SENSOR_PROGRESS_BONUS  =   2.0
FORWARD_MOMENTUM_BONUS =   1.0
ROTATION_LOOP_K        =   8
ROTATION_LOOP_PENALTY  =  -8.0
WALL_CONFUSION_PENALTY = -20.0
IR_BONUS               =  10.0
BLINK_PERSIST_BONUS    =   1.5


# ─────────────────────────────────────────────────────────────────────────────
# Dueling DRQN (GRU core)
# ─────────────────────────────────────────────────────────────────────────────
class DuelingDRQN(nn.Module):
    """
    GRU-based Dueling Q-network.
    Input:  (batch, seq_len, INPUT_DIM)
    Output: (batch, seq_len, N_ACTIONS)  Q-values at each timestep
    """
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
        """
        x: (batch, seq_len, input_dim)
        h: (1, batch, gru_hidden) — initial hidden state, zeros if None
        Returns: Q (batch, seq_len, n_actions), h_out (1, batch, gru_hidden)
        """
        out, h_out = self.gru(x, h)          # out: (batch, seq_len, gru_hidden)
        v = self.value(out)                   # (batch, seq_len, 1)
        a = self.advantage(out)               # (batch, seq_len, n_actions)
        Q = v + (a - a.mean(dim=2, keepdim=True))
        return Q, h_out

    def init_hidden(self, batch_size, device):
        return torch.zeros(1, batch_size, self.gru_hidden, device=device)


# ─────────────────────────────────────────────────────────────────────────────
# Episode Buffer  (stores full episodes, samples subsequences)
# ─────────────────────────────────────────────────────────────────────────────
class EpisodeBuffer:
    """
    Stores complete episodes as lists of (input_vec, action, shaped_reward, done).
    Samples random fixed-length subsequences for BPTT.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer   = deque(maxlen=capacity)

    def push(self, episode):
        """episode: list of (inp, action, reward, done) tuples"""
        if len(episode) >= SEQ_LEN:   # only store episodes long enough to sample
            self.buffer.append(episode)

    def sample(self, batch_size):
        """Returns batch of (inp_seq, act_seq, rew_seq, done_seq) tensors."""
        episodes = random.sample(self.buffer, batch_size)
        inp_batch, act_batch, rew_batch, don_batch = [], [], [], []

        for ep in episodes:
            # Random starting point within the episode
            max_start = len(ep) - SEQ_LEN
            start     = random.randint(0, max_start)
            seq       = ep[start : start + SEQ_LEN]

            inp_batch.append([t[0] for t in seq])
            act_batch.append([t[1] for t in seq])
            rew_batch.append([t[2] for t in seq])
            don_batch.append([t[3] for t in seq])

        # Convert to tensors: (batch, seq_len, dim)
        inp_t = torch.FloatTensor(np.array(inp_batch))
        act_t = torch.LongTensor(np.array(act_batch))
        rew_t = torch.FloatTensor(np.array(rew_batch))
        don_t = torch.FloatTensor(np.array(don_batch))
        return inp_t, act_t, rew_t, don_t

    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────────────────────────────────────────
# Input builder  (no frame stack — GRU carries temporal context)
# ─────────────────────────────────────────────────────────────────────────────
class InputBuilder:
    """
    Builds the per-step input vector for the GRU.
    No frame stack needed — the GRU hidden state is the memory.
    """
    def __init__(self):
        self.last_act    = 2
        self.rot_streak  = 0
        self.stuck_win   = deque(maxlen=10)
        self.ir_streak   = 0

    def reset(self, obs):
        self.last_act   = 2
        self.rot_streak = 0
        self.stuck_win.clear()
        self.ir_streak  = 0
        return self._build(obs)

    def step(self, obs, action_idx):
        self.last_act = action_idx
        if action_idx in ROTATION_ACTIONS:
            self.rot_streak = min(self.rot_streak + 1, 20)
        else:
            self.rot_streak = 0
        self.stuck_win.append(float(obs[17]))
        if bool(obs[16]):
            self.ir_streak = min(self.ir_streak + 1, 10)
        else:
            self.ir_streak = 0
        return self._build(obs)

    def _build(self, obs):
        raw   = obs.astype(np.float32)                             # 18
        act   = np.zeros(N_ACTIONS, dtype=np.float32)
        act[self.last_act] = 1.0                                   #  5
        rot   = np.array([self.rot_streak / 20.0], dtype=np.float32)  # 1
        stuck = np.array([sum(self.stuck_win) / 10.0 if self.stuck_win else 0.0],
                         dtype=np.float32)                         #  1
        ir_s  = np.array([self.ir_streak / 10.0], dtype=np.float32)   # 1
        return np.concatenate([raw, act, rot, stuck, ir_s])        # = 26


# ─────────────────────────────────────────────────────────────────────────────
# Reward Shaper  (same logic as Phase 2 DDQN)
# ─────────────────────────────────────────────────────────────────────────────
class RewardShaper:
    def __init__(self):
        self.prev_stuck  = False
        self.prev_sensor = np.zeros(17, dtype=bool)
        self.action_hist = deque(maxlen=ROTATION_LOOP_K)
        self.ir_given    = False
        self.blink_steps = 0

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

        if any_sonar:
            self.blink_steps = 0
        else:
            self.blink_steps = min(self.blink_steps + 1, BLINK_CAP)
        likely_blink = (self.blink_steps < BLINK_PATIENCE)

        if self.prev_stuck and action_idx in ROTATION_ACTIONS and not next_stuck:
            shaped += ESCAPE_BONUS

        if action_idx in ROTATION_ACTIONS:
            n_new = int((next_bits & ~curr_bits).sum())
            if n_new > 0:
                shaped += SENSOR_PROGRESS_BONUS * n_new

        if action_idx == 2 and not next_stuck and np.any(curr_bits):
            shaped += FORWARD_MOMENTUM_BONUS

        self.action_hist.append(action_idx)
        if (len(self.action_hist) == ROTATION_LOOP_K
                and all(a in ROTATION_ACTIONS for a in self.action_hist)
                and not np.any(next_bits)
                and not likely_blink):
            shaped += ROTATION_LOOP_PENALTY

        if action_idx == 2 and next_stuck and any_sonar and not ir_now:
            shaped += WALL_CONFUSION_PENALTY

        if ir_now and not self.ir_given:
            shaped += IR_BONUS
            self.ir_given = True

        if action_idx == 2 and not next_stuck and not any_sonar and likely_blink:
            shaped += BLINK_PERSIST_BONUS

        self.prev_stuck  = next_stuck
        self.prev_sensor = next_bits.copy()
        return shaped


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────
def train(args):
    device = torch.device("cpu")

    online_net = DuelingDRQN(INPUT_DIM, GRU_HIDDEN, N_ACTIONS).to(device)
    target_net = DuelingDRQN(INPUT_DIM, GRU_HIDDEN, N_ACTIONS).to(device)
    target_net.eval()

    optimizer = optim.Adam(online_net.parameters(), lr=LR)
    ep_buffer = EpisodeBuffer(capacity=5000)
    shaper    = RewardShaper()
    inp_bld   = InputBuilder()

    # ── Resume from checkpoint ────────────────────────────────────────────────
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        online_net.load_state_dict(ckpt["online_state_dict"])
        target_net.load_state_dict(ckpt["online_state_dict"])
        total_steps = ckpt.get("total_steps", 0)
        best_mean   = ckpt.get("best_mean", -np.inf)
        print(f"  ✔ Resumed from '{args.resume}'")
        print(f"    total_steps={total_steps:,}  |  best_mean={best_mean:.1f}")
        print(f"    Epsilon ~ {max(EPS_END, EPS_START - (EPS_START - EPS_END) * total_steps / EPS_DECAY_STEPS):.3f}")
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
    print(f"  Dueling DRQN  |  Phase 2  difficulty=2 (blinking box)  wall=True")
    print(f"  GRU hidden={GRU_HIDDEN}  input_dim={INPUT_DIM}  seq_len={SEQ_LEN}")
    print(f"  Episodes={args.episodes}  max_steps={args.max_steps}  seed={args.seed}")
    print("=" * 72)
    print(f"{'Ep':>6}  {'RawRew':>9}  {'Mean100':>9}  {'Eps':>6}  {'Steps':>9}  Status")
    print("-" * 72)

    for ep in range(1, args.episodes + 1):
        raw_obs  = env.reset(seed=args.seed + ep)
        inp_vec  = inp_bld.reset(raw_obs)
        shaper.reset()

        # GRU hidden state carried across steps within the episode
        h = online_net.init_hidden(1, device)

        ep_raw   = 0.0
        done     = False
        won      = False
        ep_data  = []   # list of (inp, action, shaped_reward, done)

        while not done:
            eps = max(EPS_END,
                      EPS_START - (EPS_START - EPS_END) * total_steps / EPS_DECAY_STEPS)

            if random.random() < eps:
                action_idx = random.randrange(N_ACTIONS)
            else:
                with torch.no_grad():
                    inp_t = torch.FloatTensor(inp_vec).unsqueeze(0).unsqueeze(0)  # (1,1,26)
                    Q, h  = online_net(inp_t, h)
                    action_idx = int(Q[0, 0].argmax().item())

            next_raw_obs, raw_reward, done = env.step(ACTIONS[action_idx], render=False)
            next_inp_vec = inp_bld.step(next_raw_obs, action_idx)
            shaped_r     = shaper.shape(raw_reward, action_idx, raw_obs, next_raw_obs)

            ep_data.append((inp_vec, action_idx, shaped_r, float(done)))
            inp_vec  = next_inp_vec
            raw_obs  = next_raw_obs
            ep_raw  += raw_reward
            total_steps += 1

            if raw_reward >= 1000:
                won = True

        # Store full episode
        ep_buffer.push(ep_data)

        # Train on random subsequences once buffer is warm
        if len(ep_buffer) >= MIN_EPISODES:
            inp_b, act_b, rew_b, don_b = ep_buffer.sample(BATCH_SIZE)
            inp_b = inp_b.to(device)   # (B, SEQ_LEN, INPUT_DIM)
            act_b = act_b.to(device)   # (B, SEQ_LEN)
            rew_b = rew_b.to(device)   # (B, SEQ_LEN)
            don_b = don_b.to(device)   # (B, SEQ_LEN)

            h0 = online_net.init_hidden(BATCH_SIZE, device)

            # Online net: Q values for current states
            Q_online, _ = online_net(inp_b, h0)               # (B, SEQ_LEN, N_ACTIONS)

            # DDQN target: online selects action, target evaluates it
            with torch.no_grad():
                h0_tgt           = target_net.init_hidden(BATCH_SIZE, device)
                Q_online_ns, _   = online_net(inp_b, h0)      # for action selection
                Q_target_ns, _   = target_net(inp_b, h0_tgt)  # for value evaluation

                # Shift by 1 to get next-state Q values
                # For the last step in seq we use the last available — acceptable approx
                best_a_ns  = Q_online_ns[:, 1:, :].argmax(dim=2)          # (B, SEQ_LEN-1)
                tgt_q_ns   = Q_target_ns[:, 1:, :].gather(
                    2, best_a_ns.unsqueeze(2)).squeeze(2)                  # (B, SEQ_LEN-1)

                # Pad last step (terminal approximation)
                tgt_q_ns   = torch.cat(
                    [tgt_q_ns, tgt_q_ns[:, -1:]], dim=1)                  # (B, SEQ_LEN)

                y = rew_b + GAMMA * tgt_q_ns * (1.0 - don_b)              # (B, SEQ_LEN)

            # Predicted Q for taken actions
            pred_q = Q_online.gather(2, act_b.unsqueeze(2)).squeeze(2)    # (B, SEQ_LEN)

            loss = (y - pred_q).pow(2).mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(online_net.parameters(), CLIP_GRAD)
            optimizer.step()

        if ep % TARGET_UPDATE == 0:
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
                "input_dim":         INPUT_DIM,
                "gru_hidden":        GRU_HIDDEN,
                "n_actions":         N_ACTIONS,
                "total_steps":       total_steps,
                "best_mean":         best_mean,
            }, args.save)
            print(f"         └─ New best mean={best_mean:.1f}  saved → {args.save}")

    print("=" * 72)
    print(f"  Done.  Best 100-ep mean (raw): {best_mean:.1f}")
    print("=" * 72)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes",  type=int,  default=4000)
    parser.add_argument("--max_steps", type=int,  default=2000)
    parser.add_argument("--seed",      type=int,  default=0)
    parser.add_argument("--save",      type=str,  default="weights_drqn_phase2.pth")
    parser.add_argument("--resume",    type=str,  default=None,
                        help="Path to a checkpoint .pth to resume training from")
    args = parser.parse_args()
    train(args)