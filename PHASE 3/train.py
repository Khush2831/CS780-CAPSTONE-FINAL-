import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from obelix import OBELIX

DEVICE = "cpu"

ACTIONS = ("L45", "L22", "FW", "R22", "R45")
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}

# ==============================
# Dueling DDQN Network
# ==============================
class DuelingDQN(nn.Module):
    def __init__(self, input_dim=23):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.value = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.advantage = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5)
        )

    def forward(self, x):
        x = self.shared(x)
        v = self.value(x)
        a = self.advantage(x)
        return v + (a - a.mean(dim=1, keepdim=True))


# ==============================
# Prioritized Replay Buffer
# ==============================
class PER:
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.buffer = []
        self.priorities = []
        self.pos = 0

    def add(self, experience, priority=1.0):
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(priority)
        else:
            self.buffer[self.pos] = experience
            self.priorities[self.pos] = priority
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size=64, alpha=0.6):
        probs = np.array(self.priorities) ** alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]
        return samples, indices

    def update_priorities(self, indices, td_errors):
        for i, e in zip(indices, td_errors):
            self.priorities[i] = float(abs(e) + 1e-5)


# ==============================
# Feature Engineering
# ==============================
class FeatureBuilder:
    def __init__(self):
        self.reset()

    def reset(self):
        self.last_seen = np.zeros(18)
        self.search_timer = 0
        self.stuck_history = deque(maxlen=5)
        self.last_action = 2
        self.is_pushing = 0
        self.last_direction = 0

    def build(self, obs):
        seen = np.any(obs[:17])

        if seen:
            self.last_seen = obs.copy()
            self.search_timer = 0
        else:
            self.search_timer += 1

        left = np.sum(obs[0:4])
        right = np.sum(obs[12:16])
        if left > right:
            self.last_direction = -1
        elif right > left:
            self.last_direction = 1

        if obs[16] == 1:
            self.is_pushing = 1

        self.stuck_history.append(obs[17])
        stuck_avg = np.mean(self.stuck_history)

        return np.concatenate([
            obs,
            [self.is_pushing],
            [self.search_timer / 50.0],
            [stuck_avg],
            [self.last_direction],
            [self.last_action / 4.0]
        ])


# ==============================
# Search Policy
# ==============================
class SearchPolicy:
    def __init__(self):
        self.pattern = ["FW","FW","FW","FW","L45"]
        self.idx = 0

    def reset(self):
        self.idx = 0

    def step(self):
        action = self.pattern[self.idx]
        self.idx = (self.idx + 1) % len(self.pattern)
        return action


# ==============================
# Agent
# ==============================
class Agent:
    def __init__(self):
        self.net = DuelingDQN().to(DEVICE)
        self.target = DuelingDQN().to(DEVICE)
        self.target.load_state_dict(self.net.state_dict())

        self.opt = optim.Adam(self.net.parameters(), lr=1e-3)
        self.buffer = PER()

        self.feature = FeatureBuilder()
        self.search = SearchPolicy()

        self.gamma = 0.99

    def select_action(self, state, obs, epsilon):
        if random.random() < epsilon:
            return random.randint(0, 4)

        if np.sum(obs[:17]) == 0:
            return ACTION_TO_IDX[self.search.step()]

        x = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            q = self.net(x)
        return int(torch.argmax(q).item())

    def train_step(self, batch_size=64):
        if len(self.buffer.buffer) < batch_size:
            return

        samples, indices = self.buffer.sample(batch_size)

        states, actions, rewards, next_states, dones = zip(*samples)

        # ---- FIXED PART ----
        states = torch.from_numpy(np.array(states, dtype=np.float32)).to(DEVICE)
        next_states = torch.from_numpy(np.array(next_states, dtype=np.float32)).to(DEVICE)

        actions = torch.tensor(actions, dtype=torch.long, device=DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
        dones = torch.tensor(dones, dtype=torch.float32, device=DEVICE)
        # --------------------

        q_vals = self.net(states)
        q_val = q_vals.gather(1, actions.unsqueeze(1)).squeeze()

        next_q = self.net(next_states)
        next_actions = torch.argmax(next_q, dim=1)

        next_q_target = self.target(next_states)
        next_q_val = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze()

        target = rewards + self.gamma * next_q_val * (1 - dones)

        loss = nn.MSELoss()(q_val, target.detach())

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        td_errors = (q_val - target).detach().cpu().numpy()
        self.buffer.update_priorities(indices, td_errors)


# ==============================
# TRAINING LOOP (WITH RENDER)
# ==============================
def train():
    env = OBELIX(
        scaling_factor=5,
        arena_size=500,
        max_steps=1000,
        wall_obstacles=True,
        difficulty=2
    )

    agent = Agent()

    epsilon = 1.0

    for episode in range(500):

        obs = env.reset(seed=episode)
        agent.feature.reset()
        agent.search.reset()

        total_reward = 0
        done = False

        while not done:

            state = agent.feature.build(obs)

            action_idx = agent.select_action(state, obs, epsilon)
            action = ACTIONS[action_idx]

            next_obs, reward, done = env.step(action, render=True)

            next_state = agent.feature.build(next_obs)

            agent.buffer.add(
                (state, action_idx, reward, next_state, done),
                priority=1.0
            )

            agent.train_step()

            obs = next_obs
            agent.feature.last_action = action_idx

            total_reward += reward

        epsilon = max(0.05, epsilon * 0.995)

        if episode % 10 == 0:
            agent.target.load_state_dict(agent.net.state_dict())

        print(f"Episode {episode}, Reward {total_reward}, Epsilon {epsilon:.3f}")

    torch.save(agent.net.state_dict(), "weights.pth")


if __name__ == "__main__":
    train()