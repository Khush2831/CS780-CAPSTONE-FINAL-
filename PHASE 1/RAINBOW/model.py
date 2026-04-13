import torch
import torch.nn as nn
import torch.nn.functional as F
from noisy import NoisyLinear


class RainbowC51(nn.Module): 
    def __init__(self, state_dim=18, action_dim=5, atoms=51, v_min=-200, v_max=2000):
        super().__init__()

        self.action_dim = action_dim
        self.atoms = atoms

        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = NoisyLinear(128, 128)

        self.value = NoisyLinear(128, atoms)
        self.advantage = NoisyLinear(128, action_dim * atoms)

        self.register_buffer("support", torch.linspace(v_min, v_max, atoms))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        value = self.value(x).view(-1, 1, self.atoms)
        adv = self.advantage(x).view(-1, self.action_dim, self.atoms)

        q_atoms = value + adv - adv.mean(dim=1, keepdim=True)
        probs = F.softmax(q_atoms, dim=2)

        return probs

    def reset_noise(self):
        self.fc2.reset_noise()
        self.value.reset_noise()
        self.advantage.reset_noise()

    def get_q(self, probs):
        return torch.sum(probs * self.support, dim=2)