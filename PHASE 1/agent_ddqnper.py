import numpy as np
import torch
import torch.nn as nn
import os

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
OBS_DIM = 18
N_ACTIONS = 5


class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature = nn.Sequential(
            nn.Linear(OBS_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, N_ACTIONS)
        )

    def forward(self, x):
        f = self.feature(x)
        v = self.value_stream(f)
        a = self.advantage_stream(f)
        return v + (a - a.mean(dim=1, keepdim=True))


model = QNetwork()
model_loaded = False

weight_paths = [
    # "weights_per.pth",
    "weights.pth",
    # "weights_best.pth"
]

for path in weight_paths:
    if os.path.exists(path):
        try:
            state_dict = torch.load(path, map_location="cpu")
            model.load_state_dict(state_dict)
            model_loaded = True
            print(f"[INFO] Loaded weights from {path}")
            break
        except Exception as e:
            print(f"[WARNING] Failed loading {path}: {e}")

model.eval()



def policy(obs, rng=None):

    if not model_loaded:
        return np.random.choice(ACTIONS)

   
    if obs[17] == 1:
        return np.random.choice(["L45", "L22", "R22", "R45"])

    state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        q_values = model(state)

    action_idx = int(torch.argmax(q_values))

    return ACTIONS[action_idx]