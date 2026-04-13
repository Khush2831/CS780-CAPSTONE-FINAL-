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

SEED          = 42
OBS_DIM       = 18
STACK         = 4
INPUT_DIM     = OBS_DIM * STACK
N_ACTIONS     = 5
ACTIONS       = ["L45", "L22", "FW", "R22", "R45"]
ACTION_IDX    = {a: i for i, a in enumerate(ACTIONS)}

NUM_EPISODES  = 1500
MAX_STEPS     = 1000
BATCH_SIZE    = 64
GAMMA         = 0.99
LR            = 1e-4
BUFFER_CAP    = 50_000
MIN_BUFFER    = 500

EPS_START     = 1.0
EPS_END       = 0.01
EPS_DECAY_EP  = 600

TARGET_SYNC   = 5
GRAD_CLIP     = 5.0
REWARD_SCALE  = 100.0
WEIGHTS_PATH  = "weights.pth"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class FrameStack:
    def __init__(self, n, obs_dim):
        self.n = n
        self.obs_dim = obs_dim
        self.frames = deque(maxlen=n)

    def reset(self, obs):
        for _ in range(self.n):
            self.frames.append(obs.copy())
        return np.concatenate(list(self.frames)).astype(np.float32)

    def step(self, obs):
        self.frames.append(obs.copy())
        return np.concatenate(list(self.frames)).astype(np.float32)


class DuelingQNetwork(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, n_actions=N_ACTIONS):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.value = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        self.advantage = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, n_actions))

    def forward(self, x):
        h = self.shared(x)
        v = self.value(h)
        a = self.advantage(h)
        return v + (a - a.mean(dim=1, keepdim=True))


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.max_p = 1.0

    def push(self, obs, action, reward, next_obs, done):
        t = (obs, action, reward, next_obs, float(done))
        if len(self.buffer) < self.capacity:
            self.buffer.append(t)
        else:
            self.buffer[self.pos] = t
        self.priorities[self.pos] = self.max_p
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta):
        n = len(self.buffer)
        raw = self.priorities[:n] ** self.alpha
        probs = raw / raw.sum()
        idxs = np.random.choice(n, batch_size, replace=False, p=probs)
        weights = (n * probs[idxs]) ** (-beta)
        weights = (weights / weights.max()).astype(np.float32)
        batch = [self.buffer[i] for i in idxs]
        obs, acts, rews, nobs, dones = zip(*batch)
        return (
            torch.from_numpy(np.stack(obs)),
            torch.tensor(acts, dtype=torch.long),
            torch.tensor(rews, dtype=torch.float32),
            torch.from_numpy(np.stack(nobs)),
            torch.tensor(dones, dtype=torch.float32),
            idxs,
            torch.from_numpy(weights),
        )

    def update_priorities(self, idxs, td_errors):
        for i, e in zip(idxs, td_errors):
            p = float(abs(e)) + 1e-6
            self.priorities[i] = p
            if p > self.max_p:
                self.max_p = p

    def __len__(self):
        return len(self.buffer)


def select_action(stacked_obs, online_net, epsilon, rng):
    obs = stacked_obs[-OBS_DIM:]

    if obs[16] == 1 and obs[17] == 0:
        return "FW", ACTION_IDX["FW"]

    if obs[17] == 1:
        escape = "L22" if rng.random() < 0.5 else "R22"
        return escape, ACTION_IDX[escape]

    if rng.random() < epsilon:
        idx = int(rng.integers(N_ACTIONS))
        return ACTIONS[idx], idx

    x = torch.from_numpy(stacked_obs).unsqueeze(0)
    with torch.no_grad():
        q = online_net(x).squeeze(0).numpy()
    idx = int(np.argmax(q))
    return ACTIONS[idx], idx


def update_step(online_net, target_net, optimizer, buffer, beta):
    obs, acts, rews, nobs, dones, idxs, weights = buffer.sample(BATCH_SIZE, beta)

    q_online = online_net(obs)
    q_values = q_online.gather(1, acts.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_acts = online_net(nobs).argmax(dim=1, keepdim=True)
        next_q = target_net(nobs).gather(1, next_acts).squeeze(1)
        targets = rews + GAMMA * next_q * (1.0 - dones)

    td_errors = (targets - q_values).detach().numpy()
    loss = (weights * F.mse_loss(q_values, targets, reduction="none")).mean()

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(online_net.parameters(), GRAD_CLIP)
    optimizer.step()
    buffer.update_priorities(idxs, td_errors)
    return float(loss.item())


def shape_reward(raw_reward, attached, prev_attached):
    r = raw_reward / REWARD_SCALE
    r = np.clip(r, -3.0, 22.0)
    if attached and not prev_attached:
        r += 3.0
    if attached:
        r += 0.1
    return r


class Logger:
    def __init__(self):
        self.rewards = []
        self.steps_log = []
        self.losses = []
        self.epsilons = []
        self.successes = []

    def log(self, reward, steps, eps, success):
        self.rewards.append(reward)
        self.steps_log.append(steps)
        self.epsilons.append(eps)
        self.successes.append(int(success))

    def log_loss(self, loss):
        self.losses.append(loss)

    def print_summary(self, ep, total, t0):
        last = self.rewards[-10:]
        sr = sum(self.successes[-10:])
        print(
            f"Ep {ep:4d}/{total} | R: {self.rewards[-1]:8.1f} | "
            f"Mean10: {np.mean(last):8.1f} | Succ10: {sr}/10 | "
            f"e: {self.epsilons[-1]:.3f} | Steps: {self.steps_log[-1]:4d} | "
            f"Time: {time.time()-t0:.0f}s"
        )

    def save_plot(self, path):
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        axes[0].plot(self.rewards, alpha=0.3, color="steelblue")
        if len(self.rewards) >= 20:
            sm = np.convolve(self.rewards, np.ones(20)/20, mode="valid")
            axes[0].plot(range(19, len(self.rewards)), sm, color="navy", lw=2)
        axes[0].set_title("Reward"); axes[0].grid(alpha=0.3)
        if self.losses:
            axes[1].plot(self.losses, alpha=0.5, color="darkorange", lw=0.8)
            axes[1].set_title("Loss"); axes[1].grid(alpha=0.3)
        window = 30
        if len(self.successes) >= window:
            rate = np.convolve(self.successes, np.ones(window)/window, mode="valid")
            axes[2].plot(range(window-1, len(self.successes)), rate, color="green", lw=2)
        axes[2].set_ylim(0, 1.05); axes[2].set_title(f"Success rate (rolling {window})")
        axes[2].grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
        print(f"Plot saved: {path}")


def train():
    t0 = time.time()
    env = OBELIX(scaling_factor=5, arena_size=500, max_steps=MAX_STEPS,
                 wall_obstacles=False, difficulty=0, seed=SEED)
    stacker = FrameStack(STACK, OBS_DIM)
    logger = Logger()
    rng = np.random.default_rng(SEED)

    online_net = DuelingQNetwork(INPUT_DIM, N_ACTIONS)
    target_net = DuelingQNetwork(INPUT_DIM, N_ACTIONS)
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(online_net.parameters(), lr=LR)
    buffer = PrioritizedReplayBuffer(BUFFER_CAP)

    print("=" * 68)
    print(f"  OBELIX L1 | Dueling DDQN + PER | Episodes: {NUM_EPISODES}")
    print("=" * 68)

    best_mean = -np.inf

    for ep in range(1, NUM_EPISODES + 1):
        frac = min(1.0, ep / EPS_DECAY_EP)
        epsilon = EPS_START + frac * (EPS_END - EPS_START)
        beta = 0.4 + 0.6 * (ep / NUM_EPISODES)

        raw_obs = env.reset(seed=SEED + ep).astype(np.float32)
        s = stacker.reset(raw_obs)

        done = False
        total_r = 0.0
        steps = 0
        success = False
        attached = False

        while not done:
            prev_attached = attached
            action_str, action_idx = select_action(s, online_net, epsilon, rng)
            raw_next, raw_reward, done = env.step(action_str, render=False)
            raw_next = raw_next.astype(np.float32)

            if not attached and raw_reward >= 80:
                attached = True

            if done and raw_reward > 500:
                success = True

            r = shape_reward(raw_reward, attached, prev_attached)
            s_next = stacker.step(raw_next)
            buffer.push(s, action_idx, r, s_next, done)

            if len(buffer) >= MIN_BUFFER:
                loss = update_step(online_net, target_net, optimizer, buffer, beta)
                logger.log_loss(loss)

            s = s_next
            total_r += raw_reward
            steps += 1

        if ep % TARGET_SYNC == 0:
            target_net.load_state_dict(online_net.state_dict())

        logger.log(total_r, steps, epsilon, success)

        if ep % 10 == 0:
            logger.print_summary(ep, NUM_EPISODES, t0)

        if len(logger.rewards) >= 50:
            m50 = np.mean(logger.rewards[-50:])
            if m50 > best_mean:
                best_mean = m50
                torch.save(online_net.state_dict(), "weights_best.pth")

    torch.save(online_net.state_dict(), WEIGHTS_PATH)
    print(f"\n{'='*68}")
    print(f"Done in {time.time()-t0:.0f}s")
    print(f"Mean last 50: {np.mean(logger.rewards[-50:]):.1f}")
    print(f"Success last 50: {sum(logger.successes[-50:])/50:.1%}")
    print(f"Saved: {WEIGHTS_PATH} | Best: weights_best.pth")
    logger.save_plot("training_curve.png")


if __name__ == "__main__":
    train()