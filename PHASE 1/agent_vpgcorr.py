import torch
import torch.nn as nn
import numpy as np
import os

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
OBS_DIM = 18
N_ACTIONS = 5


class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, N_ACTIONS)
        )

    def forward(self, x):
        logits = self.net(x)
        return torch.softmax(logits, dim=-1)


model = PolicyNet()
loaded = False

_DIR = os.path.dirname(os.path.abspath(__file__))
_WEIGHTS = os.path.join(_DIR, "vpg_weights_final.pth")

if os.path.exists(_WEIGHTS):
    model.load_state_dict(torch.load(_WEIGHTS, map_location="cpu"))
    loaded = True
else:
    print(f"WARNING: weights not found at {_WEIGHTS}")

model.eval()


def policy(obs, rng=None):

    if not loaded:
        return np.random.choice(ACTIONS)

    if obs[17] == 1:
        return np.random.choice(["L45", "R45"])

    state = torch.tensor(obs, dtype=torch.float32)

    with torch.no_grad():
        probs = model(state)

    action = torch.argmax(probs).item()

    return ACTIONS[action]