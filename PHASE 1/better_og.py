"""
train_ddqn.py  –  Dueling DDQN  |  Phase 1: Static Box + Wall
==============================================================
Run:
    python train_ddqn.py --episodes 5000 --save weights_best.pth --save_eval weights_eval.pth

Fresh start with preloaded weights:
    python train_ddqn.py --load weights_best.pth --episodes 5000 --save weights_best.pth --save_eval weights_eval.pth
"""

import argparse
import os
import random
import shutil
import subprocess
import sys
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from obelix import OBELIX

ACTIONS          = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS        = len(ACTIONS)
ROTATION_ACTIONS = {0, 1, 3, 4}

FRAME_STACK = 4
OBS_DIM     = 18
STACK_DIM   = FRAME_STACK * OBS_DIM
AUG_DIM     = STACK_DIM + N_ACTIONS + 1 + 1 + 1   # = 80

GAMMA           = 0.97
LR              = 3e-4
BATCH_SIZE      = 128
REPLAY_SIZE     = 100_000
MIN_REPLAY      = 2_000
TARGET_UPDATE   = 500
CLIP_GRAD       = 1.0

EPS_START        = 1.0
EPS_END          = 0.05
EPS_DECAY_STEPS  = 80_000

ESCAPE_BONUS           =  15.0
SENSOR_PROGRESS_BONUS  =   2.0
FORWARD_MOMENTUM_BONUS =   1.0
ROTATION_LOOP_K        =   8
ROTATION_LOOP_PENALTY  =  -8.0
WALL_CONFUSION_PENALTY = -20.0
IR_BONUS               =  10.0

EVAL_EVERY = 50   # episodes


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


class RewardShaper:
    def __init__(self):
        self.prev_stuck   = False
        self.prev_sensor  = np.zeros(17, dtype=bool)
        self.action_hist  = deque(maxlen=ROTATION_LOOP_K)
        self.ir_given     = False

    def reset(self):
        self.prev_stuck  = False
        self.prev_sensor = np.zeros(17, dtype=bool)
        self.action_hist.clear()
        self.ir_given    = False

    def shape(self, raw, action_idx, obs, next_obs):
        shaped     = raw
        next_stuck = bool(next_obs[17])
        curr_bits  = obs[:17].astype(bool)
        next_bits  = next_obs[:17].astype(bool)
        ir_now     = bool(next_obs[16])
        any_sonar  = bool(np.any(next_obs[:16]))

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
                and not np.any(next_bits)):
            shaped += ROTATION_LOOP_PENALTY

        if action_idx == 2 and next_stuck and any_sonar and not ir_now:
            shaped += WALL_CONFUSION_PENALTY

        if ir_now and not self.ir_given:
            shaped += IR_BONUS
            self.ir_given = True

        self.prev_stuck  = next_stuck
        self.prev_sensor = next_bits.copy()
        return shaped


class ObsBuilder:
    def __init__(self):
        self.frames       = deque(maxlen=FRAME_STACK)
        self.last_act     = 2
        self.rot_streak   = 0
        self.stuck_window = deque(maxlen=10)
        self.ir_streak    = 0

    def reset(self, obs):
        self.frames.clear()
        for _ in range(FRAME_STACK):
            self.frames.append(obs.copy())
        self.last_act     = 2
        self.rot_streak   = 0
        self.stuck_window.clear()
        self.ir_streak    = 0
        return self._build()

    def step(self, obs, action_idx):
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
        return self._build()

    def _build(self):
        stacked = np.concatenate(list(self.frames)).astype(np.float32)
        act_oh  = np.zeros(N_ACTIONS, dtype=np.float32)
        act_oh[self.last_act] = 1.0
        rot   = np.array([self.rot_streak / 20.0],        dtype=np.float32)
        stuck = np.array([sum(self.stuck_window) / 10.0], dtype=np.float32)
        ir_s  = np.array([self.ir_streak / 10.0],         dtype=np.float32)
        return np.concatenate([stacked, act_oh, rot, stuck, ir_s])


# ─────────────────────────────────────────────────────────────────────────────
# Eval via evaluate.py subprocess
# ─────────────────────────────────────────────────────────────────────────────
def run_eval(weights_path, agent_file):
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    eval_script = os.path.join(script_dir, "evaluate.py")
    if not os.path.exists(eval_script):
        print("  [eval] evaluate.py not found, skipping.")
        return -np.inf

    env = os.environ.copy()
    env["OBELIX_WEIGHTS"] = weights_path

    cmd = [
        sys.executable, eval_script,
        "--agent_file", agent_file,
        "--runs", "10",
        "--wall_obstacles",
        "--max_steps", "2000",
        "--seed", "0",
        "--difficulty", "0",
        "--leaderboard_csv", os.path.join(script_dir, "leaderboard_eval.csv"),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
        for line in (result.stdout + result.stderr).splitlines():
            if "mean=" in line:
                for token in line.split():
                    if token.startswith("mean="):
                        return float(token.split("=")[1])
        print(f"  [eval] Could not parse mean. Output:\n{result.stdout[:300]}")
        return -np.inf
    except Exception as e:
        print(f"  [eval] Error: {e}")
        return -np.inf


def save_ckpt(net, path):
    torch.save({
        "online_state_dict": net.state_dict(),
        "frame_stack": FRAME_STACK,
        "obs_dim": OBS_DIM,
        "n_actions": N_ACTIONS,
        "aug_dim": AUG_DIM,
    }, path)


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────
def train(args):
    device = torch.device("cpu")

    online_net = DuelingDDQN(AUG_DIM, N_ACTIONS).to(device)
    target_net = DuelingDDQN(AUG_DIM, N_ACTIONS).to(device)

    # ── Warm start ────────────────────────────────────────────────────────
    eps_step_offset = 0
    if args.load and os.path.exists(args.load):
        ckpt = torch.load(args.load, map_location="cpu")
        online_net.load_state_dict(ckpt["online_state_dict"])
        eps_step_offset = int(EPS_DECAY_STEPS * 0.4)
        print(f"  Loaded weights from: {args.load}  (eps offset={eps_step_offset:,})")
    elif args.load:
        print(f"  [warn] --load path not found: {args.load}, starting fresh.")

    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(online_net.parameters(), lr=LR)
    replay    = PrioritizedReplayBuffer(REPLAY_SIZE)
    shaper    = RewardShaper()
    obs_bld   = ObsBuilder()

    env = OBELIX(
        scaling_factor=5, arena_size=500, max_steps=args.max_steps,
        wall_obstacles=True, difficulty=0, box_speed=2, seed=args.seed,
    )

    total_steps    = eps_step_offset
    raw_history    = []
    best_mean      = -np.inf   # training Mean100 gate
    best_eval_mean = -np.inf   # evaluate.py gate

    print("=" * 72)
    print(f"  Dueling DDQN  |  Phase 1  wall_obstacles=True  |  state_dim={AUG_DIM}")
    print(f"  save (Mean100) → {args.save}")
    print(f"  save_eval      → {args.save_eval}  (checked every {EVAL_EVERY} eps)")
    print(f"  Episodes={args.episodes}   max_steps={args.max_steps}   seed={args.seed}")
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

        # ── SAVE 1: best rolling Mean100 ──────────────────────────────────
        if mean100 > best_mean and ep >= 10:
            best_mean = mean100
            save_ckpt(online_net, args.save)
            print(f"         └─ [BEST] Mean100={best_mean:.1f}  saved → {args.save}")

        # ── SAVE 2: every 50 eps, run evaluate.py, keep better weights ────
        if ep % EVAL_EVERY == 0:
            print(f"  [eval] ep={ep}: running evaluate.py ...")

            # Write current weights to a temp file
            tmp = args.save_eval + ".tmp.pth"
            save_ckpt(online_net, tmp)

            current_eval = run_eval(tmp, args.agent_file)

            # Also evaluate previously saved eval weights (if they exist)
            prev_eval = -np.inf
            if os.path.exists(args.save_eval):
                prev_eval = run_eval(args.save_eval, args.agent_file)

            print(f"  [eval] current={current_eval:.1f}  previous_best={prev_eval:.1f}  (best so far={best_eval_mean:.1f})")

            if current_eval > prev_eval:
                shutil.copy(tmp, args.save_eval)
                best_eval_mean = current_eval
                print(f"  [eval] ✓ Current weights are better → saved → {args.save_eval}")
            else:
                print(f"  [eval] Previous weights are better → {args.save_eval} unchanged")

            if os.path.exists(tmp):
                os.remove(tmp)

    print("=" * 72)
    print(f"  Done.  Best Mean100 (training): {best_mean:.1f}  → {args.save}")
    print(f"         Best eval score        : {best_eval_mean:.1f}  → {args.save_eval}")
    print("=" * 72)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load",       type=str, default=None,
                        help="Pretrained weights to warm-start from")
    parser.add_argument("--episodes",   type=int, default=2000)
    parser.add_argument("--max_steps",  type=int, default=2000)
    parser.add_argument("--seed",       type=int, default=0)
    parser.add_argument("--save",       type=str, default="weights.pth",
                        help="Saved when rolling Mean100 improves")
    parser.add_argument("--save_eval",  type=str, default="weights_eval.pth",
                        help="Saved every 50 eps if evaluate.py score improves")
    parser.add_argument("--agent_file", type=str, default="agent_template.py",
                        help="Agent file used by evaluate.py subprocess")
    args = parser.parse_args()
    train(args)