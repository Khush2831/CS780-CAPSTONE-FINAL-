import torch
import torch.nn as nn
import numpy as np
import os

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
OBS_DIM = 18
N_ACTIONS = 5


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(OBS_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        self.actor = nn.Linear(128, N_ACTIONS)

    def forward(self, x):
        x = self.shared(x)
        logits = self.actor(x)
        return logits


model = ActorCritic()
loaded = False

if os.path.exists("ppo_final.pth"):
    model.load_state_dict(torch.load("ppo_final.pth", map_location="cpu"), strict=False)
    loaded = True

model.eval()


def policy(obs, rng=None):

    # fallback
    if not loaded:
        return np.random.choice(ACTIONS)

    # anti-stuck (important)
    if obs[17] == 1:
        return np.random.choice(["L45", "R45"])

    state = torch.tensor(obs, dtype=torch.float32)

    with torch.no_grad():
        logits = model(state)

    action = torch.argmax(logits).item()

    return ACTIONS[action]