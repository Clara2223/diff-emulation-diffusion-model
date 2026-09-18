import numpy as np
import torch
import torch.nn as nn


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
STATE_DIM = 5
MODEL_VERSION = "equilibrium_score_v1_small"


class PhaseSpaceScoreNet(nn.Module):
    """Small score network for (x, y, vx, vy, energy)."""

    def __init__(self, hidden_dim=64, embedding_dim=16):
        super().__init__()
        frequencies = torch.exp(
            torch.linspace(np.log(1.0), np.log(1000.0), embedding_dim // 2)
        )
        self.register_buffer("frequencies", frequencies)
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM + embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, STATE_DIM),
        )

    def forward(self, state, sigma):
        log_sigma = torch.log(sigma.clamp_min(1e-6)).unsqueeze(1)
        angles = log_sigma * self.frequencies.unsqueeze(0)
        noise_embedding = torch.cat((torch.sin(angles), torch.cos(angles)), dim=1)
        return self.net(torch.cat((state, noise_embedding), dim=1))
