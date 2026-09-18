import numpy as np
import torch

from .model import DEVICE


def train_score(model, loader, mean, std, steps=2_000, sigma_min=0.02, sigma_max=1.0):
    """Train with denoising score matching at several noise levels."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    iterator = iter(loader)
    model.train()
    for _ in range(steps):
        try:
            (clean,) = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            (clean,) = next(iterator)
        clean = (clean.to(DEVICE) - mean) / std
        sigma = torch.exp(
            torch.empty(clean.shape[0], device=DEVICE).uniform_(
                np.log(sigma_min), np.log(sigma_max)
            )
        )
        noise = torch.randn_like(clean)
        prediction = model(clean + sigma[:, None] * noise, sigma)
        target = -noise / sigma[:, None]
        loss = ((prediction - target) ** 2).sum(dim=1).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model
