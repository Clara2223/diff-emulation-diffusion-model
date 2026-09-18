import torch.nn as nn

from ..model import STATE_DIM


class PhaseSpaceDynamicsNet(nn.Module):
    """Future model: predict the next phase-space state from time slices i.e. time conditioning?."""

    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, STATE_DIM),
        )

    def forward(self, state):
        return self.net(state)
