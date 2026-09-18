"""Run the current equilibrium phase-space score-model pipeline."""

from datetime import datetime
from pathlib import Path

from torch.utils.data import DataLoader, TensorDataset

from model.data import load_equilibrium_bundle
from model.model import DEVICE, PhaseSpaceScoreNet
from model.predict import (
    calculate_einstein_diffusion,
    calculate_kramers_rate,
    log_prediction,
    plot_results,
    sample_score,
)
from model.train import train_score


DATA_PATH = Path(__file__).parent / "train_data" / (
    "double_well_data_100000000steps_50000burn_20260918_115938.npz"
)
TRAIN_STEPS = 2_0000
MAX_TRAINING_SAMPLES = 50_0000
GENERATED_SAMPLES = 2_000


# 1. Load positions, velocities, energies, and diagnostic MD data.
train_data, true_samples, positions, _ = load_equilibrium_bundle(
    DATA_PATH,
    max_samples=MAX_TRAINING_SAMPLES,
    diffusion_frames=50_000,
)

# 2. Standardize the five-dimensional state: x, y, vx, vy, and energy.
mean = train_data.mean(dim=0).to(DEVICE)
std = train_data.std(dim=0).clamp_min(1e-6).to(DEVICE)
loader = DataLoader(TensorDataset(train_data), batch_size=512, shuffle=True)

# 3. Create and train the current score model.
model = PhaseSpaceScoreNet().to(DEVICE)
model = train_score(model, loader, mean, std, steps=TRAIN_STEPS)

# 4. Generate equilibrium predictions with annealed Langevin sampling.
generated = sample_score(
    model,
    mean,
    std,
    n_samples=GENERATED_SAMPLES,
    n_levels=12,
    steps_per_level=6,
)

# 5. Calculate the current diagnostics and save the comparison plot.
einstein_diffusion = calculate_einstein_diffusion(positions)
kramers_rate, barrier_height = calculate_kramers_rate()
output_path = Path(__file__).parent / "predictions" / (
    f"phase_space_score_{datetime.now():%Y%m%d_%H%M%S}.png"
)
output_path.parent.mkdir(exist_ok=True)
plot_results(true_samples, generated, output_path)
log_path = log_prediction(
    output_path,
    model_version="equilibrium_score_v1_small",
    einstein_diffusion=einstein_diffusion,
    barrier_height=barrier_height,
    kramers_rate=kramers_rate,
)

print(f"Device: {DEVICE}")
print(f"Einstein diffusion coefficient: {einstein_diffusion:.5f}")
print(f"Kramers barrier height: {barrier_height:.5f}")
print(f"Kramers transition rate: {kramers_rate:.5f}")
print(f"Visualization saved as '{output_path}'")
print(f"Prediction log saved as '{log_path}'")
