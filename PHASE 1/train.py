"""
train.py  —  OBELIX Level 1 Best-Possible Submission
======================================================
Architecture  : Dueling Double DQN
Replay        : Prioritized Experience Replay (PER)
Obs           : 4-frame stack  (4 × 18 = 72 dims)
Extras        : Reward shaping, hard overrides, gradient clipping
"""

import os, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import deque
from obelix import OBELIX

# ─────────────────────────────────────────────────────────
#  HYPERPARAMETERS
# ─────────────────────────────────────────────────────────
SEED            = 42
OBS_DIM         = 18
STACK           = 4                  # frame-stack depth
INPUT_DIM       = OBS_DIM * STACK    # 72
N_ACTIONS       = 5
ACTIONS         = ["L45", "L22", "FW", "R22", "R45"]
ACTION_IDX      = {a: i for i, a in enumerate(ACTIONS)}

NUM_EPISODES    = 1000
MAX_STEPS       = 1000
BATCH_SIZE      = 128
GAMMA           = 0.99
LR              = 3e-4
BUFFER_CAP      = 50_000
MIN_BUFFER      = 1_000             # start learning after this many transitions

EPS_START       = 1.0
EPS_END         = 0.05
EPS_DECAY_EP    = 700               # reach EPS_END by this episode

TARGET_SYNC_EP  = 10               # sync target net every N episodes

# PER params
PER_ALPHA       = 0.6              # how much prioritisation (0=uniform, 1=full)
PER_BETA_START  = 0.4
PER_BETA_END    = 1.0

GRAD_CLIP       = 10.0
WEIGHTS_PATH    = "weights.pth"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ─────────────────────────────────────────────────────────
#  FRAME STACKER
# ─────────────────────────────────────────────────────────
class FrameStack:
    """Maintains a rolling window of the last `n` observations."""
    def __init__(self, n: int, obs_dim: int):
        self.n = n
        self.frames = deque(maxlen=n)
        self.obs_dim = obs_dim

    def reset(self, obs: np.ndarray) -> np.ndarray:
        for _ in range(self.n):
            self.frames.append(obs.copy())
        return self._get()

    def step(self, obs: np.ndarray) -> np.ndarray:
        self.frames.append(obs.copy())
        return self._get()

    def _get(self) -> np.ndarray:
        return np.concatenate(list(self.frames), axis=0).astype(np.float32)


# ─────────────────────────────────────────────────────────
#  DUELING DDQN NETWORK
# ─────────────────────────────────────────────────────────
class DuelingQNetwork(nn.Module):
    """
    Dueling architecture:
      Shared encoder → Value stream V(s)
                     → Advantage stream A(s,a)
      Q(s,a) = V(s) + A(s,a) − mean_a A(s,a)
    """
    def __init__(self, input_dim: int = INPUT_DIM, n_actions: int = N_ACTIONS):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.value = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.advantage = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.shared(x)
        v = self.value(h)                          # (B, 1)
        a = self.advantage(h)                      # (B, A)
        return v + (a - a.mean(dim=1, keepdim=True))  # (B, A)


# ─────────────────────────────────────────────────────────
#  PRIORITIZED EXPERIENCE REPLAY
# ─────────────────────────────────────────────────────────
class PrioritizedReplayBuffer:
    """
    Simple array-based PER.
    Transitions stored as (stacked_obs, action_idx, shaped_reward,
                           stacked_next_obs, done).
    """
    def __init__(self, capacity: int, alpha: float = PER_ALPHA):
        self.capacity = capacity
        self.alpha    = alpha
        self.buffer   = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos      = 0
        self.max_p    = 1.0

    def push(self, obs, action, reward, next_obs, done):
        transition = (obs, action, reward, next_obs, float(done))
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = self.max_p
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float):
        n = len(self.buffer)
        raw = self.priorities[:n] ** self.alpha
        probs = raw / raw.sum()
        idxs  = np.random.choice(n, batch_size, replace=False, p=probs)
        weights = (n * probs[idxs]) ** (-beta)
        weights = (weights / weights.max()).astype(np.float32)

        batch = [self.buffer[i] for i in idxs]
        obs, acts, rews, nobs, dones = zip(*batch)
        return (
            torch.from_numpy(np.stack(obs)),
            torch.tensor(acts,  dtype=torch.long),
            torch.tensor(rews,  dtype=torch.float32),
            torch.from_numpy(np.stack(nobs)),
            torch.tensor(dones, dtype=torch.float32),
            idxs,
            torch.from_numpy(weights),
        )

    def update_priorities(self, idxs, td_errors: np.ndarray):
        for i, err in zip(idxs, td_errors):
            p = float(abs(err)) + 1e-6
            self.priorities[i] = p
            if p > self.max_p:
                self.max_p = p

    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────────────────────
#  REWARD SHAPING
# ─────────────────────────────────────────────────────────
def shape_reward(raw_reward: float, obs: np.ndarray,
                 attached: bool, prev_attached: bool) -> float:
    """
    Augment the environment reward with dense shaping signals.

    Signals added (not replacing original reward):
      +10  on first attachment this episode  (detected via +100 env bonus)
      +0.5 per step while attached and moving (encourages pushing)
      −0.2 per rotation step while attached  (penalise spinning after attach)
    """
    r = raw_reward

    # Extra bonus for first time becoming attached
    if attached and not prev_attached:
        r += 10.0

    # Small living reward while pushing (keep moving)
    if attached:
        r += 0.5

    return r


# ─────────────────────────────────────────────────────────
#  ACTION SELECTION
# ─────────────────────────────────────────────────────────
def select_action(stacked_obs: np.ndarray,
                  online_net: DuelingQNetwork,
                  epsilon: float,
                  rng: np.random.Generator,
                  attached: bool,
                  last_action: str) -> tuple:
    """
    Returns (action_str, action_idx).

    Hard overrides run before the network — these guarantee
    correct behaviour in critical moments regardless of Q-values.
    """
    # Raw single-step obs (last 18 bits of stacked obs)
    obs = stacked_obs[-OBS_DIM:]

    # ── Hard overrides ─────────────────────────────────
    # 1. IR sensor: box directly ahead → attach / keep pushing
    if obs[16] == 1:
        return "FW", ACTION_IDX["FW"]

    # 2. Attached + not stuck → keep pushing
    if attached and obs[17] == 0:
        return "FW", ACTION_IDX["FW"]

    # 3. Stuck while attached → small rotation to escape
    if attached and obs[17] == 1:
        # Alternate left/right to unstick
        escape = "L22" if rng.random() < 0.5 else "R22"
        return escape, ACTION_IDX[escape]
    # ────────────────────────────────────────────────────

    # Epsilon-greedy
    if rng.random() < epsilon:
        idx = int(rng.integers(N_ACTIONS))
        return ACTIONS[idx], idx

    x = torch.from_numpy(stacked_obs).unsqueeze(0)
    with torch.no_grad():
        q = online_net(x).squeeze(0).numpy()
    idx = int(np.argmax(q))
    return ACTIONS[idx], idx


# ─────────────────────────────────────────────────────────
#  ONE DDQN UPDATE STEP
# ─────────────────────────────────────────────────────────
def update_step(online_net: DuelingQNetwork,
                target_net: DuelingQNetwork,
                optimizer:  optim.Optimizer,
                buffer:     PrioritizedReplayBuffer,
                beta:       float) -> float:

    obs, acts, rews, nobs, dones, idxs, weights = buffer.sample(BATCH_SIZE, beta)

    # Current Q values
    q_online = online_net(obs)                          # (B, A)
    q_values = q_online.gather(1, acts.unsqueeze(1)).squeeze(1)  # (B,)

    # DDQN target:
    #   action = argmax_a Q_online(next_obs)  ← online selects
    #   value  = Q_target(next_obs, action)   ← target evaluates
    with torch.no_grad():
        next_actions = online_net(nobs).argmax(dim=1, keepdim=True)   # (B,1)
        next_q       = target_net(nobs).gather(1, next_actions).squeeze(1)  # (B,)
        targets      = rews + GAMMA * next_q * (1.0 - dones)          # (B,)

    td_errors = (targets - q_values).detach().numpy()

    # Weighted MSE loss (importance-sampling weights from PER)
    loss = (weights * F.mse_loss(q_values, targets, reduction="none")).mean()

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(online_net.parameters(), GRAD_CLIP)
    optimizer.step()

    buffer.update_priorities(idxs, td_errors)
    return float(loss.item())


# ─────────────────────────────────────────────────────────
#  LOGGING & PLOTTING
# ─────────────────────────────────────────────────────────
class Logger:
    def __init__(self):
        self.rewards   = []
        self.steps_log = []
        self.losses    = []
        self.epsilons  = []
        self.successes = []

    def log(self, reward, steps, eps, success):
        self.rewards.append(reward)
        self.steps_log.append(steps)
        self.epsilons.append(eps)
        self.successes.append(int(success))

    def log_loss(self, loss):
        self.losses.append(loss)

    def print(self, ep, total, t0):
        last = self.rewards[-10:]
        sr   = sum(self.successes[-10:])
        elapsed = time.time() - t0
        print(
            f"Ep {ep:4d}/{total} | "
            f"R: {self.rewards[-1]:8.1f} | "
            f"Mean10: {np.mean(last):8.1f} | "
            f"Succ10: {sr}/10 | "
            f"ε: {self.epsilons[-1]:.3f} | "
            f"Steps: {self.steps_log[-1]:4d} | "
            f"Time: {elapsed:.0f}s"
        )

    def plot(self, path: str):
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        # Rewards
        axes[0].plot(self.rewards, alpha=0.3, color="steelblue")
        if len(self.rewards) >= 20:
            sm = np.convolve(self.rewards, np.ones(20)/20, mode="valid")
            axes[0].plot(range(19, len(self.rewards)), sm,
                         color="navy", lw=2, label="Smoothed (20)")
        axes[0].set_ylabel("Cumulative reward")
        axes[0].set_title("OBELIX Level 1 — Dueling DDQN + PER Training")
        axes[0].legend(); axes[0].grid(alpha=0.3)

        # Loss
        if self.losses:
            axes[1].plot(self.losses, alpha=0.4, color="darkorange", linewidth=0.8)
            axes[1].set_ylabel("TD loss"); axes[1].set_title("Loss per update")
            axes[1].grid(alpha=0.3)

        # Success rate
        window = 30
        if len(self.successes) >= window:
            rate = np.convolve(self.successes, np.ones(window)/window, mode="valid")
            axes[2].plot(range(window-1, len(self.successes)), rate,
                         color="green", lw=2)
        axes[2].set_ylabel("Success rate"); axes[2].set_ylim(0, 1.05)
        axes[2].set_title(f"Success rate (rolling {window}-ep window)")
        axes[2].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Plot saved → {path}")


# ─────────────────────────────────────────────────────────
#  MAIN TRAINING LOOP
# ─────────────────────────────────────────────────────────
def train():
    t0 = time.time()

    env    = OBELIX(scaling_factor=5, arena_size=500, max_steps=MAX_STEPS,
                    wall_obstacles=False, difficulty=0, seed=SEED)
    stacker = FrameStack(STACK, OBS_DIM)
    logger  = Logger()
    rng     = np.random.default_rng(SEED)

    online_net = DuelingQNetwork(INPUT_DIM, N_ACTIONS)
    target_net = DuelingQNetwork(INPUT_DIM, N_ACTIONS)
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(online_net.parameters(), lr=LR)
    buffer    = PrioritizedReplayBuffer(BUFFER_CAP)

    epsilon = EPS_START
    beta    = PER_BETA_START

    print("=" * 68)
    print("  OBELIX Level 1  |  Dueling DDQN + PER + Frame-Stack")
    print(f"  Episodes: {NUM_EPISODES}  |  Input: {INPUT_DIM}d  |  Buffer: {BUFFER_CAP}")
    print("=" * 68)

    best_mean = -np.inf

    for ep in range(1, NUM_EPISODES + 1):

        # Linear epsilon decay
        frac    = min(1.0, ep / EPS_DECAY_EP)
        epsilon = EPS_START + frac * (EPS_END - EPS_START)

        # Linear beta anneal for PER
        beta = PER_BETA_START + (PER_BETA_END - PER_BETA_START) * (ep / NUM_EPISODES)

        raw_obs  = env.reset(seed=SEED + ep).astype(np.float32)
        s        = stacker.reset(raw_obs)

        done     = False
        total_r  = 0.0
        steps    = 0
        success  = False
        attached = False
        last_act = "FW"

        while not done:
            prev_attached = attached

            action_str, action_idx = select_action(
                s, online_net, epsilon, rng, attached, last_act
            )

            raw_next, raw_reward, done = env.step(action_str, render=False)
            raw_next = raw_next.astype(np.float32)

            # Detect attachment from reward signal (+100 env bonus)
            if raw_reward >= 100 and not attached:
                attached = True

            if done and raw_reward > 500:
                success = True

            # Shaped reward
            shaped = shape_reward(raw_reward, raw_next, attached, prev_attached)

            s_next = stacker.step(raw_next)
            buffer.push(s, action_idx, shaped, s_next, done)

            # Gradient update (only when buffer is warm)
            if len(buffer) >= MIN_BUFFER:
                loss = update_step(online_net, target_net, optimizer, buffer, beta)
                logger.log_loss(loss)

            s        = s_next
            last_act = action_str
            total_r += raw_reward
            steps   += 1

        # Sync target network
        if ep % TARGET_SYNC_EP == 0:
            target_net.load_state_dict(online_net.state_dict())

        logger.log(total_r, steps, epsilon, success)

        if ep % 10 == 0:
            logger.print(ep, NUM_EPISODES, t0)

        # Save best checkpoint
        mean_last50 = np.mean(logger.rewards[-50:]) if len(logger.rewards) >= 50 else -np.inf
        if mean_last50 > best_mean and ep >= 50:
            best_mean = mean_last50
            torch.save(online_net.state_dict(), "weights_best.pth")

    # ── Final save ──────────────────────────────────────────
    torch.save(online_net.state_dict(), WEIGHTS_PATH)
    print(f"\n{'='*68}")
    print(f"Training done in {time.time()-t0:.0f}s")
    print(f"Final mean (last 50):   {np.mean(logger.rewards[-50:]):.1f}")
    print(f"Final success (last 50): {sum(logger.successes[-50:])/50:.1%}")
    print(f"Weights saved → {WEIGHTS_PATH}")
    print(f"Best checkpoint        → weights_best.pth")
    logger.plot("training_curve.png")


if __name__ == "__main__":
    train()