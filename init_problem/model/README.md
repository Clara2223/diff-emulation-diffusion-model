# Model package

Run from `init_problem`:

```bash
../special_energy/bin/python -m model.run --train-steps 1000 --max-samples 50000
```

The top-level script is a launcher. Everything used by the model is in this folder:

- `data.py`: loads positions, velocities, and energies.
- `model.py`: the current small score network.
- `train.py`: trains the score network.
- `predict.py`: generates samples, calculates diagnostics, and makes plots.
- `run.py`: connects the files and runs the experiment.
- `future/model.py`: saved future experiment for time-step dynamics; it is not used yet.

The current model learns equilibrium phase-space samples from `(x, y, vx, vy, energy)`. It does not learn the time direction yet. The generated samples are saved in `init_problem/predictions/`.

For a quick local run:

```bash
../special_energy/bin/python -m model.run --train-steps 500 --max-samples 10000
```

The default model is deliberately small. Diffusion is estimated with the Einstein mean-squared-displacement method. The Kramers rule is also printed as a barrier-crossing-rate estimate, using the same potential parameters as `Tiny_MD_sim.ipynb`.

Each PNG in `init_problem/predictions/` is registered in `prediction_log.csv` with its model version, Einstein diffusion, Kramers barrier height, Kramers transition rate, and device.
