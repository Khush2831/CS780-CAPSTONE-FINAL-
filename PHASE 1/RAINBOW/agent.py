import torch
from model import RainbowC51

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

model = RainbowC51()
model.load_state_dict(torch.load("weights.pth", map_location="cpu"))
model.eval()


def policy(obs, rng=None):

    state = torch.FloatTensor(obs).unsqueeze(0)

    with torch.no_grad():
        probs = model(state)
        q = model.get_q(probs)

    return ACTIONS[q.argmax().item()]