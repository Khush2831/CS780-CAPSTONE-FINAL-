import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from obelix import OBELIX

# ── Hyperparameters ───────────────────────────────────────────────────────────
SEED        = 42
ACTIONS     = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS   = 5
OBS_DIM     = 18
NUM_EP      = 2500
MAX_STEPS   = 1000
BATCH       = 64
GAMMA       = 0.99
LR          = 1e-3
BUF_CAP     = 100_000
MIN_BUF     = 1_000
EPS_START   = 1.0
EPS_END     = 0.05
EPS_DECAY   = 0.997
TARGET_SYNC = 10
PER_ALPHA   = 0.6
PER_BETA_S  = 0.4
PER_BETA_E  = 1.0
GRAD_CLIP   = 10.0

np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Network ───────────────────────────────────────────────────────────────────
class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(OBS_DIM, 256), nn.ReLU(),
            nn.Linear(256, 256),     nn.ReLU(),
        )
        self.value = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
        self.adv   = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, N_ACTIONS))

    def forward(self, x):
        h = self.shared(x)
        v = self.value(h)
        a = self.adv(h)
        return v + (a - a.mean(dim=1, keepdim=True))

# ── Prioritized Replay Buffer ─────────────────────────────────────────────────
class PERBuffer:
    def __init__(self, cap, alpha):
        self.cap   = cap
        self.alpha = alpha
        self.buf   = []
        self.prios = np.zeros(cap, dtype=np.float32)
        self.pos   = 0
        self.max_p = 1.0

    def push(self, obs, action, reward, next_obs, done):
        if len(self.buf) < self.cap:
            self.buf.append(None)
        self.buf[self.pos]   = (obs, action, reward, next_obs, float(done))
        self.prios[self.pos] = self.max_p
        self.pos = (self.pos + 1) % self.cap

    def sample(self, batch, beta):
        n     = len(self.buf)
        probs = self.prios[:n] ** self.alpha
        probs /= probs.sum()
        idxs  = np.random.choice(n, batch, replace=False, p=probs)
        w     = (n * probs[idxs]) ** (-beta)
        w     = (w / w.max()).astype(np.float32)
        batch = [self.buf[i] for i in idxs]
        obs, acts, rews, nobs, dones = zip(*batch)
        return (
            torch.FloatTensor(np.stack(obs)),
            torch.LongTensor(acts),
            torch.FloatTensor(rews),
            torch.FloatTensor(np.stack(nobs)),
            torch.FloatTensor(dones),
            idxs,
            torch.FloatTensor(w),
        )

    def update_prios(self, idxs, td_errors):
        for i, e in zip(idxs, td_errors):
            p = float(abs(e)) + 1e-6
            self.prios[i] = p
            self.max_p = max(self.max_p, p)

    def __len__(self):
        return len(self.buf)

# ── Training ──────────────────────────────────────────────────────────────────
def train():
    env    = OBELIX(scaling_factor=5, arena_size=500, max_steps=MAX_STEPS,
                    difficulty=0, seed=SEED)
    online = QNetwork()
    target = QNetwork()
    target.load_state_dict(online.state_dict())
    target.eval()

    opt    = optim.Adam(online.parameters(), lr=LR)
    buf    = PERBuffer(BUF_CAP, PER_ALPHA)
    eps    = EPS_START
    best   = -np.inf

    for ep in range(1, NUM_EP + 1):
        obs  = env.reset(seed=SEED + ep).astype(np.float32)
        done = False
        total_r = 0.0
        beta = PER_BETA_S + (PER_BETA_E - PER_BETA_S) * (ep / NUM_EP)

        while not done:
            # Action selection
            if np.random.rand() < eps:
                act_idx = np.random.randint(N_ACTIONS)
            else:
                with torch.no_grad():
                    act_idx = online(torch.FloatTensor(obs).unsqueeze(0)).argmax().item()

            next_obs, reward, done = env.step(ACTIONS[act_idx], render=False)
            next_obs = next_obs.astype(np.float32)
            buf.push(obs, act_idx, reward, next_obs, done)
            obs      = next_obs
            total_r += reward

            # Update
            if len(buf) >= MIN_BUF:
                s, a, r, ns, d, idxs, w = buf.sample(BATCH, beta)
                q_val = online(s).gather(1, a.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_a  = online(ns).argmax(1, keepdim=True)
                    next_q  = target(ns).gather(1, next_a).squeeze(1)
                    targets = r + GAMMA * next_q * (1 - d)
                td_err = (targets - q_val).detach().numpy()
                loss   = (w * F.smooth_l1_loss(q_val, targets, reduction="none")).mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(online.parameters(), GRAD_CLIP)
                opt.step()
                buf.update_prios(idxs, td_err)

        eps = max(EPS_END, eps * EPS_DECAY)

        if ep % TARGET_SYNC == 0:
            target.load_state_dict(online.state_dict())

        if ep % 100 == 0:
            print(f"Episode {ep:4d} | Reward {total_r:8.1f} | Eps {eps:.3f}")

        if total_r > best:
            best = total_r
            torch.save(online.state_dict(), "weights_best.pth")

    torch.save(online.state_dict(), "weights.pth")
    print(f"Done. Best reward: {best:.1f}")

if __name__ == "__main__":
    train()