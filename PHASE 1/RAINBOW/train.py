import torch
import torch.optim as optim
import numpy as np

from obelix import OBELIX
from model import RainbowC51
from replay_buffer import PER
from nstep import NStep
from utils import shape_reward


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]


def projection(next_dist, rewards, dones, gamma, support):

    batch_size = rewards.shape[0]
    atoms = support.size(0)
    v_min, v_max = support[0], support[-1]
    delta_z = (v_max - v_min) / (atoms - 1)

    proj = torch.zeros_like(next_dist)

    for i in range(atoms):
        tz = rewards + gamma * support[i] * (1 - dones)
        tz = tz.clamp(v_min, v_max)

        b = (tz - v_min) / delta_z
        l = b.floor().long()
        u = b.ceil().long()

        proj.view(-1).index_add_(0, (l + i * batch_size), next_dist[:, i] * (u.float() - b))
        proj.view(-1).index_add_(0, (u + i * batch_size), next_dist[:, i] * (b - l.float()))

    return proj


def train():

    net = RainbowC51().to(device)
    target = RainbowC51().to(device)
    target.load_state_dict(net.state_dict())

    optimizer = optim.Adam(net.parameters(), lr=3e-4)

    buffer = PER()
    nstep = NStep()

    gamma = 0.99
    beta = 0.4

    for episode in range(8000):

        # curriculum
        if episode < 2000:
            diff = 0
        elif episode < 5000:
            diff = 2
        else:
            diff = 3

        env = OBELIX(
                scaling_factor=5,
                arena_size=500,
                max_steps=1000,
                difficulty=diff,
                wall_obstacles=np.random.rand() < 0.5
                    )
        state = env.reset()
        total = 0
        done = False

        while not done:

            net.reset_noise()

            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            probs = net(state_t)
            q = net.get_q(probs)

            action = q.argmax().item()

            next_state, reward, done = env.step(ACTIONS[action], render=False)

            reward = shape_reward(state, next_state, reward)

            n_out = nstep.push((state, action, reward, next_state, done))

            if n_out:
                buffer.push(*n_out)
                nstep.pop()

            state = next_state
            total += reward

            if len(buffer.buffer) > 1000:

                s, a, r, ns, d, idx, w = buffer.sample(64, beta)

                s = torch.FloatTensor(s).to(device)
                ns = torch.FloatTensor(ns).to(device)
                r = torch.FloatTensor(r).to(device)
                d = torch.FloatTensor(d).to(device)

                dist = net(s)
                dist = dist[range(64), a]

                with torch.no_grad():
                    next_dist = target(ns)
                    next_q = target.get_q(next_dist)
                    next_a = next_q.argmax(1)
                    next_dist = next_dist[range(64), next_a]

                proj = projection(next_dist, r, d, gamma, net.support)

                per_sample_loss = -(proj * dist.log()).sum(1)
                loss = per_sample_loss.mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                buffer.update(idx, per_sample_loss.detach().cpu().numpy())

        if episode % 200 == 0:
            target.load_state_dict(net.state_dict())
            torch.save(net.state_dict(), "weights.pth")

        print(f"Episode {episode} Reward {total}")


if __name__ == "__main__":
    train()