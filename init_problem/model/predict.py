import csv
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .model import DEVICE


LOG_FIELDS = [
    "timestamp",
    "png",
    "model_version",
    "einstein_diffusion",
    "kramers_barrier_height",
    "kramers_transition_rate",
    "device",
]


def potential(x, y):
    g1 = 3.0 * np.exp(-((x + 1.0) ** 2 + y**2) / (2 * 0.5**2))
    g2 = 3.0 * np.exp(-((x - 1.0) ** 2 + y**2) / (2 * 0.5**2))
    return -(g1 + g2) + 0.1 * (x**2 + y**2)


def sample_score(model, mean, std, n_samples=1_000, n_levels=12, steps_per_level=6):
    """Generate equilibrium samples with annealed Langevin dynamics."""
    model.eval()
    state = torch.randn(n_samples, 5, device=DEVICE)
    sigmas = torch.exp(torch.linspace(np.log(1.0), np.log(0.02), n_levels, device=DEVICE))
    with torch.no_grad():
        for sigma in sigmas:
            sigma_batch = torch.full((n_samples,), sigma, device=DEVICE)
            step_size = 0.02 * sigma**2
            for _ in range(steps_per_level):
                state = state + step_size * model(state, sigma_batch)
                state = state + torch.sqrt(2 * step_size) * torch.randn_like(state)
    samples = state.cpu().numpy() * std.cpu().numpy() + mean.cpu().numpy()
    samples[:, 4] = potential(samples[:, 0], samples[:, 1])
    return samples


def calculate_einstein_diffusion(positions, dt=0.005):
    """Estimate diffusion from the late-time slope of the MSD."""
    displacement = positions - positions[0]
    time = np.arange(len(positions)) * dt
    msd = np.sum(displacement**2, axis=1)
    slope, _ = np.polyfit(time[len(time) // 2 :], msd[len(time) // 2 :], 1)
    return slope / 4.0


def calculate_kramers_rate(kBT=0.4, mass=1.0, gamma=1.0):
    """Estimate the underdamped barrier-crossing rate for this potential."""
    dx = 1e-4
    well = potential(-1.0, 0.0)
    barrier = potential(0.0, 0.0)
    barrier_height = barrier - well
    well_curvature = (
        potential(-1.0 + dx, 0.0)
        - 2 * well
        + potential(-1.0 - dx, 0.0)
    ) / dx**2
    barrier_curvature = (
        potential(dx, 0.0) - 2 * barrier + potential(-dx, 0.0)
    ) / dx**2
    omega_well = np.sqrt(well_curvature / mass)
    omega_barrier = np.sqrt(abs(barrier_curvature) / mass)
    transmission = np.sqrt(1 + (gamma / (2 * omega_barrier)) ** 2) - gamma / (
        2 * omega_barrier
    )
    rate = transmission * omega_well / (2 * np.pi) * np.exp(-barrier_height / kBT)
    return rate, barrier_height


def log_prediction(
    output_path,
    model_version,
    einstein_diffusion,
    barrier_height,
    kramers_rate,
    device=DEVICE,
):
    """Append run metadata and register older PNGs without known metadata."""
    output_path = Path(output_path)
    log_path = output_path.parent / "prediction_log.csv"
    existing_rows = []
    if log_path.exists():
        with log_path.open(newline="", encoding="utf-8") as log_file:
            existing_rows = list(csv.DictReader(log_file))

    current_png = output_path.name
    known_pngs = {row["png"] for row in existing_rows}
    for png_path in sorted(output_path.parent.glob("*.png")):
        relative_png = png_path.name
        if relative_png != current_png and relative_png not in known_pngs:
            existing_rows.append(
                {
                    "timestamp": "",
                    "png": relative_png,
                    "model_version": "unknown_existing",
                    "einstein_diffusion": "",
                    "kramers_barrier_height": "",
                    "kramers_transition_rate": "",
                    "device": "",
                }
            )

    if current_png not in {row["png"] for row in existing_rows}:
        existing_rows.append(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "png": current_png,
                "model_version": model_version,
                "einstein_diffusion": f"{einstein_diffusion:.8f}",
                "kramers_barrier_height": f"{barrier_height:.8f}",
                "kramers_transition_rate": f"{kramers_rate:.8f}",
                "device": device,
            }
        )

    with log_path.open("w", newline="", encoding="utf-8") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=LOG_FIELDS)
        writer.writeheader()
        writer.writerows(existing_rows)
    return log_path


def plot_results(true_samples, generated, output_path, show=False):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=160)
    x = np.linspace(-2.5, 2.5, 200)
    y = np.linspace(-2.0, 2.0, 200)
    X, Y = np.meshgrid(x, y)
    axes[0, 0].contour(X, Y, potential(X, Y), levels=12, cmap="Greys", alpha=0.5)
    axes[0, 0].hexbin(generated[:, 0], generated[:, 1], gridsize=35, cmap="plasma", mincnt=1)
    axes[0, 0].set_title("Generated position density")
    axes[0, 1].hist(true_samples[:, 4], bins=40, density=True, alpha=0.5, label="MD")
    axes[0, 1].hist(generated[:, 4], bins=40, density=True, alpha=0.7, label="Generated")
    axes[0, 1].set_title("Potential energy")
    axes[0, 1].legend()
    axes[1, 0].plot(generated[:, 0], linewidth=0.7)
    axes[1, 0].set_title("Generated x samples")
    axes[1, 1].plot(generated[:, 2], linewidth=0.7)
    axes[1, 1].set_title("Generated vx samples")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    if show:
        plt.show()
    plt.close(fig)
