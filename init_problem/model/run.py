import argparse
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from .data import load_equilibrium_bundle
from .model import DEVICE, PhaseSpaceScoreNet
from .predict import (
    calculate_einstein_diffusion,
    calculate_kramers_rate,
    log_prediction,
    plot_results,
    sample_score,
)
from .train import train_score


DEFAULT_DATA = Path(__file__).resolve().parents[1] / "train_data" / "double_well_data_100000000steps_50000burn_20260918_115938.npz"


def main():
    parser = argparse.ArgumentParser(description="Train the phase-space score and dynamics models.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--train-steps", type=int, default=2_000)
    parser.add_argument("--max-samples", type=int, default=50_000)
    parser.add_argument("--generated-samples", type=int, default=1_000)
    parser.add_argument("--n-levels", type=int, default=12)
    parser.add_argument("--steps-per-level", type=int, default=6)
    parser.add_argument("--diffusion-frames", type=int, default=50_000)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    train_data, true_samples, positions, _ = load_equilibrium_bundle(
        args.data, args.max_samples, args.diffusion_frames
    )
    mean = train_data.mean(dim=0).to(DEVICE)
    std = train_data.std(dim=0).clamp_min(1e-6).to(DEVICE)
    score_loader = DataLoader(TensorDataset(train_data), batch_size=512, shuffle=True)
    score_model = train_score(PhaseSpaceScoreNet().to(DEVICE), score_loader, mean, std, args.train_steps)
    generated = sample_score(score_model, mean, std, args.generated_samples, args.n_levels, args.steps_per_level)
    einstein = calculate_einstein_diffusion(positions)
    kramers_rate, barrier_height = calculate_kramers_rate()

    output = args.data.parent / ".." / "predictions" / f"phase_space_score_{datetime.now():%Y%m%d_%H%M%S}.png"
    output.parent.mkdir(exist_ok=True)
    plot_results(true_samples, generated, output, show=args.show)
    log_path = log_prediction(
        output,
        model_version="equilibrium_score_v1_small",
        einstein_diffusion=einstein,
        barrier_height=barrier_height,
        kramers_rate=kramers_rate,
    )
    print(f"Device: {DEVICE}")
    print(f"Einstein diffusion coefficient: {einstein:.5f}")
    print(f"Kramers barrier height: {barrier_height:.5f}")
    print(f"Kramers transition rate: {kramers_rate:.5f}")
    print(f"Visualization saved as '{output.resolve()}'")
    print(f"Prediction log saved as '{log_path.resolve()}'")


if __name__ == "__main__":
    main()
