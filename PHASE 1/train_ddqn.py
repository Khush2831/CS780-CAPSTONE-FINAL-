import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from obelix import OBELIX
from model import QNetwork
from replay_buffer import ReplayBuffer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

actions = ["L45", "L22", "FW", "R22", "R45"]


def select_action(state, epsilon, qnet):

    if np.random.rand() < epsilon:
        return np.random.randint(5)

    state = torch.FloatTensor(state).unsqueeze(0).to(device)

    with torch.no_grad():
        q_values = qnet(state)

    return q_values.argmax().item()


def train():

    env = OBELIX(
        scaling_factor=5,
        arena_size=500,
        max_steps=1000,
        difficulty=0
    )

    state_dim = 18
    action_dim = 5

    q_net = QNetwork(state_dim, action_dim).to(device)
    target_net = QNetwork(state_dim, action_dim).to(device)

    target_net.load_state_dict(q_net.state_dict())

    optimizer = optim.Adam(q_net.parameters(), lr=1e-3)

    buffer = ReplayBuffer(100000)

    gamma = 0.99
    batch_size = 64

    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995

    target_update = 1000

    steps = 0

    for episode in range(5000):

        state = env.reset()
        done = False

        total_reward = 0

        while not done:

            action = select_action(state, epsilon, q_net)

            next_state, reward, done = env.step(actions[action], render=False)

            buffer.push(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            steps += 1

            if len(buffer) > batch_size:

                states, actions_b, rewards, next_states, dones = buffer.sample(batch_size)

                states = torch.FloatTensor(states).to(device)
                actions_b = torch.LongTensor(actions_b).unsqueeze(1).to(device)
                rewards = torch.FloatTensor(rewards).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                dones = torch.FloatTensor(dones).to(device)

                q_values = q_net(states).gather(1, actions_b).squeeze()

                # Double DQN
                next_actions = q_net(next_states).argmax(1)

                next_q = target_net(next_states).gather(
                    1,
                    next_actions.unsqueeze(1)
                ).squeeze()

                target = rewards + gamma * next_q * (1 - dones)

                loss = nn.MSELoss()(q_values, target.detach())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if steps % target_update == 0:
                target_net.load_state_dict(q_net.state_dict())

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        print(f"Episode {episode} Reward {total_reward}")

        if episode % 500 == 0:
            torch.save(q_net.state_dict(), "weights.pth")


if __name__ == "__main__":
    train()