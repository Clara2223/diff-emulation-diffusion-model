from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STATE_DIM = 5  # x, y, vx, vy, potential energy


def potential(x, y):
    A1, x1, y1, s1 = 3.0, -1.0, 0.0, 0.5
    A2, x2, y2, s2 = 3.0, 1.0, 0.0, 0.5
    k_confine = 0.2
    g1 = A1 * np.exp(-((x - x1) ** 2 + (y - y1) ** 2) / (2 * s1**2))
    g2 = A2 * np.exp(-((x - x2) ** 2 + (y - y2) ** 2) / (2 * s2**2))
    return -(g1 + g2) + 0.5 * k_confine * (x**2 + y**2)


def load_phase_space(path, max_samples=200_000, seed=7):
    """Load aligned position, velocity, and energy observations."""
    with np.load(path) as data:
        positions = data["positions"]
        velocities = data["velocities"]
        energies = data["energy"][:, None]
        rng = np.random.default_rng(seed)
        n_samples = min(max_samples, len(positions))
        indices = rng.choice(len(positions), size=n_samples, replace=False)
        state = np.concatenate(
            (positions[indices], velocities[indices], energies[indices]), axis=1
        )
    return torch.from_numpy(state.astype(np.float32, copy=False))


def load_phase_space_transitions(path, max_samples=200_000, seed=7):
    """Load aligned consecutive phase-space states for dynamics learning."""
    with np.load(path) as data:
        positions = data["positions"]
        velocities = data["velocities"]
        energies = data["energy"][:, None]
        n_transitions = len(positions) - 1
        rng = np.random.default_rng(seed)
        count = min(max_samples, n_transitions)
        indices = rng.choice(n_transitions, size=count, replace=False)
        current = np.concatenate(
            (positions[indices], velocities[indices], energies[indices]), axis=1
        )
        following = np.concatenate(
            (
                positions[indices + 1],
                velocities[indices + 1],
                energies[indices + 1],
            ),
            axis=1,
        )
    return (
        torch.from_numpy(current.astype(np.float32, copy=False)),
        torch.from_numpy(following.astype(np.float32, copy=False)),
    )


class PhaseSpaceScoreNet(nn.Module):
    """Noise-conditioned score network for (x, y, vx, vy, energy)."""

    def __init__(self, hidden_dim=128, embedding_dim=32):
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
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, STATE_DIM),
        )

    def forward(self, state, sigma):
        log_sigma = torch.log(sigma.clamp_min(1e-6)).unsqueeze(1)
        angles = log_sigma * self.frequencies.unsqueeze(0)
        embedding = torch.cat((torch.sin(angles), torch.cos(angles)), dim=1)
        return self.net(torch.cat((state, embedding), dim=1))


class PhaseSpaceDynamicsNet(nn.Module):
    """Predict the next normalized phase-space state from the current one."""

    def __init__(self, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, STATE_DIM),
        )

    def forward(self, state):
        return self.net(state)


def train_score_model(
    model,
    loader,
    mean,
    std,
    steps=10_000,
    sigma_min=0.02,
    sigma_max=1.0,
    lr=1e-3,
):
    """Train with denoising score matching over a range of noise levels."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    data_iterator = iter(loader)
    log_sigma_min = np.log(sigma_min)
    log_sigma_max = np.log(sigma_max)

    for _ in range(steps):
        try:
            (clean_state,) = next(data_iterator)
        except StopIteration:
            data_iterator = iter(loader)
            (clean_state,) = next(data_iterator)

        clean_state = ((clean_state.to(DEVICE) - mean) / std).float()
        sigma = torch.exp(
            torch.empty(clean_state.shape[0], device=DEVICE).uniform_(
                log_sigma_min, log_sigma_max
            )
        )
        noise = torch.randn_like(clean_state)
        noisy_state = clean_state + sigma[:, None] * noise
        target_score = -noise / sigma[:, None]

        prediction = model(noisy_state, sigma)
        loss = ((prediction - target_score) ** 2).sum(dim=1).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


def train_dynamics_model(model, loader, mean, std, steps=10_000, lr=1e-3):
    """Fit one-step MD transitions in standardized phase space."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    data_iterator = iter(loader)
    for _ in range(steps):
        try:
            current, following = next(data_iterator)
        except StopIteration:
            data_iterator = iter(loader)
            current, following = next(data_iterator)
        current = (current.to(DEVICE) - mean) / std
        following = (following.to(DEVICE) - mean) / std
        prediction = model(current)
        loss = ((prediction - following) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


def rollout_dynamics(model, initial_state, mean, std, n_steps=500):
    """Roll out learned one-step dynamics; energy is kept physically consistent."""
    model.eval()
    state = ((initial_state.to(DEVICE) - mean) / std).unsqueeze(0)
    trajectory = []
    with torch.no_grad():
        for _ in range(n_steps):
            state = model(state)
            physical_state = state * std + mean
            physical_state[:, 4] = torch.as_tensor(
                potential(physical_state[:, 0].cpu(), physical_state[:, 1].cpu()),
                device=DEVICE,
                dtype=physical_state.dtype,
            )
            state = (physical_state - mean) / std
            trajectory.append(physical_state.squeeze(0).cpu().numpy())
    return np.asarray(trajectory)


def sample_annealed_langevin(
    model,
    mean,
    std,
    n_samples=5000,
    sigma_min=0.02,
    sigma_max=1.0,
    n_levels=24,
    steps_per_level=12,
    step_scale=0.02,
):
    """Generate phase-space samples by descending through noise levels."""
    model.eval()
    state = torch.randn(n_samples, STATE_DIM, device=DEVICE) * sigma_max
    sigmas = torch.exp(
        torch.linspace(np.log(sigma_max), np.log(sigma_min), n_levels, device=DEVICE)
    )

    with torch.no_grad():
        for sigma in sigmas:
            sigma_batch = torch.full((n_samples,), sigma, device=DEVICE)
            step_size = step_scale * sigma**2
            for _ in range(steps_per_level):
                score = model(state, sigma_batch)
                state = state + step_size * score
                state = state + torch.sqrt(2 * step_size) * torch.randn_like(state)

    samples = state.cpu().numpy() * std.cpu().numpy() + mean.cpu().numpy()
    samples[:, 4] = potential(samples[:, 0], samples[:, 1])
    return samples


def free_energy_surface(samples, kBT=0.4, limits=(-2.5, 2.5), bins=60):
    hist, xedges, yedges = np.histogram2d(
        samples[:, 0], samples[:, 1], bins=bins, range=[limits, limits]
    )
    hist = hist + 1e-6
    free_energy = -kBT * np.log(hist / hist.sum())
    free_energy -= np.min(free_energy)
    x = (xedges[:-1] + xedges[1:]) / 2
    y = (yedges[:-1] + yedges[1:]) / 2
    return np.meshgrid(x, y), free_energy


def calculate_einstein_diffusion(positions, dt):
    n_frames = positions.shape[0]
    msd = np.empty(n_frames)
    for t in range(n_frames):
        displacement = positions[t] - positions[0]
        msd[t] = np.mean(np.sum(displacement**2, axis=-1))
    time = np.arange(n_frames) * dt
    fit_start = n_frames // 2
    slope, _ = np.polyfit(time[fit_start:], msd[fit_start:], 1)
    return slope / 4.0, msd, time


def calculate_green_kubo_diffusion(velocities, dt):
    n_frames = velocities.shape[0]
    vacf = np.empty(n_frames)
    for t in range(n_frames):
        v_current = velocities[: n_frames - t]
        v_future = velocities[t:n_frames]
        vacf[t] = np.mean(np.sum(v_current * v_future, axis=-1))
    return np.trapezoid(vacf, dx=dt) / 2.0, vacf


def plot_results(
    true_samples, generated_samples, dynamics_trajectory=None, kBT=0.4, output_path=None
):
    fig = plt.figure(figsize=(14, 10), dpi=200)
    grid = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(grid[0, 0])
    x_grid = np.linspace(-2.5, 2.5, 200)
    y_grid = np.linspace(-2.0, 2.0, 200)
    X, Y = np.meshgrid(x_grid, y_grid)
    contour = ax1.contour(X, Y, potential(X, Y), levels=12, cmap="Greys", alpha=0.5)
    ax1.clabel(contour, inline=True, fontsize=8)
    hb = ax1.hexbin(generated_samples[:, 0], generated_samples[:, 1], gridsize=45, cmap="plasma", mincnt=1)
    fig.colorbar(hb, ax=ax1, label="Generated density")
    ax1.set_title("Phase-space score model: position density")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")

    ax2 = fig.add_subplot(grid[0, 1])
    ax2.hist(true_samples[:, 4], bins=50, density=True, alpha=0.5, label="MD")
    ax2.hist(generated_samples[:, 4], bins=50, density=True, alpha=0.7, label="Generated")
    ax2.set_title("Potential-energy distribution")
    ax2.set_xlabel("V(x, y)")
    ax2.set_ylabel("Density")
    ax2.legend()

    ax3 = fig.add_subplot(grid[1, 0])
    (X_free, Y_free), generated_free_energy = free_energy_surface(generated_samples, kBT=kBT)
    (_, _), true_free_energy = free_energy_surface(true_samples, kBT=kBT)
    image = ax3.contourf(X_free, Y_free, generated_free_energy.T, levels=20, cmap="viridis_r")
    fig.colorbar(image, ax=ax3).set_label("Generated free energy")
    ax3.contour(X_free, Y_free, true_free_energy.T, levels=8, colors="white", linewidths=0.7)
    ax3.set_title("Generated free energy, MD contours")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")

    ax4 = fig.add_subplot(grid[1, 1])
    ax4.plot(generated_samples[:, 0], label="x", alpha=0.8, linewidth=0.7)
    ax4.plot(generated_samples[:, 2], label="vx", alpha=0.7, linewidth=0.7)
    if dynamics_trajectory is not None:
        ax4.plot(
            dynamics_trajectory[:, 0],
            label="dynamics x",
            alpha=0.8,
            linewidth=0.7,
        )
        ax4.plot(
            dynamics_trajectory[:, 2],
            label="dynamics vx",
            alpha=0.7,
            linewidth=0.7,
        )
    ax4.set_title("Score samples and learned dynamics")
    ax4.set_xlabel("Generated or rollout step")
    ax4.legend()

    plt.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=200)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).parent / "train_data" / "double_well_data_100000000steps_50000burn_20260918_115938.npz",
    )
    parser.add_argument("--train-steps", type=int, default=10_000)
    parser.add_argument("--max-samples", type=int, default=200_000)
    args = parser.parse_args()

    train_data = load_phase_space(args.data, max_samples=args.max_samples)
    current_states, following_states = load_phase_space_transitions(
        args.data, max_samples=args.max_samples
    )
    mean = train_data.mean(dim=0).to(DEVICE)
    std = train_data.std(dim=0).clamp_min(1e-6).to(DEVICE)
    loader = DataLoader(TensorDataset(train_data), batch_size=512, shuffle=True)

    model = PhaseSpaceScoreNet().to(DEVICE)
    train_score_model(model, loader, mean, std, steps=args.train_steps)
    generated_samples = sample_annealed_langevin(model, mean, std)

    dynamics_model = PhaseSpaceDynamicsNet().to(DEVICE)
    transition_loader = DataLoader(
        TensorDataset(current_states, following_states), batch_size=512, shuffle=True
    )
    train_dynamics_model(
        dynamics_model, transition_loader, mean, std, steps=args.train_steps
    )
    dynamics_trajectory = rollout_dynamics(
        dynamics_model, current_states[0], mean, std, n_steps=500
    )

    with np.load(args.data) as data_file:
        true_samples = np.column_stack(
            (data_file["positions"][:5000], data_file["velocities"][:5000], data_file["energy"][:5000])
        )
        D_einstein, _, _ = calculate_einstein_diffusion(data_file["positions"], 0.005)
        D_green_kubo, _ = calculate_green_kubo_diffusion(data_file["velocities"], 0.005)

    output_path = Path(__file__).parent / "predictions" / f"phase_space_score_{datetime.now():%Y%m%d_%H%M%S}.png"
    plot_results(
        true_samples,
        generated_samples,
        dynamics_trajectory=dynamics_trajectory,
        output_path=output_path,
    )
    print(f"Einstein diffusion coefficient: {D_einstein:.5f}")
    print(f"Green-Kubo diffusion coefficient: {D_green_kubo:.5f}")
    print(f"Visualization saved as '{output_path}'")


if __name__ == "__main__":
    main()