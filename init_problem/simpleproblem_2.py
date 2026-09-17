from datetime import date
from datetime import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_positions(path):
    with np.load(path) as data:
        return torch.tensor(data["positions"], dtype=torch.float32)


class ScoreNet(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x):
        return self.net(x)


def train_score_model(model, loader, epochs=400, sigma=0.1, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    for _ in range(epochs):
        for (x,) in loader:
            x = x.to(DEVICE)
            noise = torch.randn_like(x)
            noisy_x = x + sigma * noise
            target_score = -noise / sigma

            loss = ((model(noisy_x) - target_score) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model




################# predicting new samples using the trained model #################

def sample_langevin(model, n_samples=5000, dt=0.01, n_steps=1000):
    model.eval()
    x_gen = torch.randn(n_samples, 2, device=DEVICE) * 1.5

    with torch.no_grad():
        for _ in range(n_steps):
            score = model(x_gen)
            noise_step = torch.randn_like(x_gen)
            x_gen = x_gen + score * dt + np.sqrt(2 * dt) * noise_step

    return x_gen.cpu().numpy()


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
    """Compute D from the MSD slope."""
    n_frames = positions.shape[0]
    msd = np.empty(n_frames)

    for t in range(n_frames):
        displacement = positions[t] - positions[0]
        msd[t] = np.mean(np.sum(displacement ** 2, axis=-1))

    time = np.arange(n_frames) * dt
    fit_start = n_frames // 2
    slope, _ = np.polyfit(time[fit_start:], msd[fit_start:], 1)
    return slope / 4.0, msd, time


def calculate_green_kubo_diffusion(velocities, dt):
    """Compute D from the velocity autocorrelation function."""
    n_frames = velocities.shape[0]
    vacf = np.empty(n_frames)

    for t in range(n_frames):
        v_current = velocities[: n_frames - t]
        v_future = velocities[t:n_frames]
        vacf[t] = np.mean(np.sum(v_current * v_future, axis=-1))

    return np.trapezoid(vacf, dx=dt) / 2.0, vacf


def plot_results(true_samples, true_energy, gen_samples, kBT=0.4):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)

    axes[0].plot(true_energy[:2000], color="#2b5c8f", lw=0.8, alpha=0.8)
    axes[0].set_xlabel("Sample Index")
    axes[0].set_ylabel("Potential Energy $U(x,y)$ [$k_B T$]")
    axes[0].set_title("Energy Trajectory (MD Ground-Truth)")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].hist(true_energy, bins=50, density=True, color="#4682b4", edgecolor="black", alpha=0.7)
    axes[1].set_xlabel("Potential Energy $U(x,y)$")
    axes[1].set_ylabel("Probability Density")
    axes[1].set_title("Boltzmann Energy Distribution")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    (X, Y), free_energy_gen = free_energy_surface(gen_samples, kBT=kBT)
    (_, _), free_energy_true = free_energy_surface(true_samples, kBT=kBT)
    contour = axes[2].contourf(X, Y, free_energy_gen.T, levels=20, cmap="viridis_r")
    fig.colorbar(contour, ax=axes[2]).set_label("Free Energy $\Delta F$ [$k_B T$]")
    axes[2].contour(X, Y, free_energy_true.T, levels=8, colors="white", linewidths=0.7, alpha=0.7)
    axes[2].set_xlabel("Coordinate $x$ (Collective Variable)")
    axes[2].set_ylabel("Coordinate $y$")
    axes[2].set_title("Model Free Energy (White=MD Ground Truth)")

    plt.tight_layout()
    plt.savefig(f"boltzmann_md_visualization_comparison_{date}_ {time}.png", dpi=300)
    plt.show()
    print(f"Visualization saved as 'boltzmann_md_visualization_comparison_{date}_{time}.png'")


#### compare to the OG training data distribution and diffusion coefficient ##
def main():
    root = Path(__file__).resolve().parent
    data_path = root / "train_data" / "double_well_data_10Msteps.npz"

    train_data = load_positions(data_path)
    loader = DataLoader(TensorDataset(train_data), batch_size=256, shuffle=True)

    model = ScoreNet().to(DEVICE)
    train_score_model(model, loader)
    gen_samples = sample_langevin(model)

    with np.load(data_path) as data_file:
        true_samples = data_file["positions"][:5000]
        if "energy" in data_file:
            true_energy = data_file["energy"][:5000]
        else:
            A1, x1, y1, s1 = 3.0, -1.0, 0.0, 0.5
            A2, x2, y2, s2 = 3.0, 1.0, 0.0, 0.5
            g1 = A1 * np.exp(-((true_samples[:, 0] - x1) ** 2 + (true_samples[:, 1] - y1) ** 2) / (2 * s1 ** 2))
            g2 = A2 * np.exp(-((true_samples[:, 0] - x2) ** 2 + (true_samples[:, 1] - y2) ** 2) / (2 * s2 ** 2))
            true_energy = -(g1 + g2) + 0.2 * (true_samples[:, 0] ** 2 + true_samples[:, 1] ** 2) / 2

        trajectory_positions = data_file["positions"]
        trajectory_velocities = data_file["velocities"]

    plot_results(true_samples, true_energy, gen_samples)

    dt = 0.005
    D_einstein, _, _ = calculate_einstein_diffusion(trajectory_positions, dt)
    D_green_kubo, _ = calculate_green_kubo_diffusion(trajectory_velocities, dt)

    print("--- Diffusion Coefficient (D) Results ---")
    print(f"Einstein (MSD) Method:    {D_einstein:.5f}")
    print(f"Green-Kubo (VACF) Method: {D_green_kubo:.5f}")


if __name__ == "__main__":
    main()