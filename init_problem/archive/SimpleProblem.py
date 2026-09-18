import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
device = "cuda" if torch.cuda.is_available() else "cpu"

# training data - change 
#data = torch.randn(10000, 1) + 5
npz_file = np.load("double_well_data.npz")
data = torch.tensor(npz_file["positions"], dtype=torch.float32)

loader = DataLoader(
    TensorDataset(data),
    batch_size=64,
    shuffle=True
)
'''
# MD trajectory data load
positions = torch.tensor(
    WIP_data,
    dtype=torch.float32)
# positions fx [timesteps, particles, dimensions], so what about velocity?

dataset = TensorDataset(positions)
loader = DataLoader(
    dataset,
    batch_size=64,
    #shuffle=True 
'''

model = nn.Sequential(
    nn.Linear(2, 64),
    nn.ReLU(),
    nn.Linear(64, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 2)
)
'''model = nn.Sequential( nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 1) )'''

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#training-----------------------------------------------------------------------
for _ in range(10000):
    x, = next(iter(loader))
    #x = torch.randn(64, 1) + 5
    noise = torch.randn_like(x)                                      # should the gaussian noise addition scale with timestep ?
    noisy = x + noise                               

    prediction = model(noisy)
    loss = ((prediction - noise) ** 2).mean()                        # MSE loss function  maybe this is too simple

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

x = torch.randn(1, 1)

dt=0.005 # i think

for _ in range(200):
    #x -= model(x) * 0.05                                            # 'Euler' step, follow learned vector field (each param is a vector i think)
    x = x + model(x) * dt + g(t) * torch.randn_like(x) * dt**0.5     # Euler-Maruyama SDE step

print(x.item())




model.eval()
n_samples = 5000

# Start from 2D isotropic Gaussian noise
x_gen = torch.randn(n_samples, 2, device=device)

# Simple reverse-time sampling integration (200 Euler steps)
dt_sample = 0.05
with torch.no_grad():
    for _ in range(200):
        # Follow reverse field: drift = -model(x)
        drift = -model(x_gen)
        noise_step = torch.randn_like(x_gen)
        x_gen = x_gen + drift * dt_sample + noise_step * (dt_sample ** 0.5)

# Convert generated samples back to NumPy
gen_samples = x_gen.cpu().numpy()

# Get true ground-truth positions from your dataset for comparison
data = np.load("double_well_data.npz")
true_samples = data["positions"][:n_samples]

# ==========================================
# 2. Side-by-Side Visualization Pipeline
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

# Panel 1: Ground-Truth Langevin Distribution
hb1 = axes[0].hexbin(
    true_samples[:, 0], true_samples[:, 1], 
    gridsize=45, cmap='plasma', mincnt=1
)
fig.colorbar(hb1, ax=axes[0], label='Sample Count')
axes[0].set_title("Ground-Truth Simulation $p(x,y)$")
axes[0].set_xlabel("X position")
axes[0].set_ylabel("Y position")
axes[0].set_xlim([-2.5, 2.5])
axes[0].set_ylim([-2.0, 2.0])
axes[0].grid(True, linestyle='--', alpha=0.3)

# Panel 2: Diffusion Model Generated Samples
hb2 = axes[1].hexbin(
    gen_samples[:, 0], gen_samples[:, 1], 
    gridsize=45, cmap='plasma', mincnt=1
)
fig.colorbar(hb2, ax=axes[1], label='Sample Count')
axes[1].set_title("Generated Samples from Model $p_{\\theta}(x,y)$")
axes[1].set_xlabel("X position")
axes[1].set_ylabel("Y position")
axes[1].set_xlim([-2.5, 2.5])
axes[1].set_ylim([-2.0, 2.0])
axes[1].grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig("diffusion_vs_ground_truth.png", dpi=300)
plt.show()

print("Visualization saved as 'diffusion_vs_ground_truth.png'")








