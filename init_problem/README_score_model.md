# Phase-space score model

## Previous model: `archive/simpleproblem_2.py`

The original script learned a two-dimensional score field from positions only. Its network was an ordinary multilayer perceptron:

```text
(x, y) -> (score_x, score_y)
```

Training used one Gaussian noise level (`sigma=0.1`) and denoising score matching. For a noisy point

$$
\tilde{x}=x+\sigma\epsilon,
\qquad \epsilon\sim\mathcal{N}(0,I),
$$

the target was $-\epsilon/\sigma$. The learned vector field approximated $\nabla_x\log p_\sigma(x)$, where $p_\sigma$ is a smoothed positional equilibrium distribution.

Sampling used overdamped Langevin dynamics:

$$
x_{t+1}=x_t+s_\theta(x_t)\,\Delta t+\sqrt{2\Delta t}\,\epsilon_t.
$$

This was a single-noise-level score model, not yet a full diffusion model. It learned positional equilibrium statistics, not velocities or time-ordered molecular dynamics.

## Current model: `model_phase_space_score.py`

The improved model uses the aligned arrays saved by the MD simulation:

```text
(x, y, vx, vy, energy) -> (score_x, score_y, score_vx, score_vy, score_energy)
```

The energy is the potential energy $V(x,y)$ from the same configuration. It is therefore a useful consistency feature, but it is not independent of position. The model now sees position, velocity, and energy together and learns a score for their joint equilibrium distribution.

The main improvements are:

- phase-space state instead of position-only state;
- standardization of all five state variables;
- a noise-conditioned network with a Fourier-style noise embedding;
- log-uniform training over many noise levels from `sigma_min` to `sigma_max`;
- annealed Langevin sampling from high noise down to low noise;
- random subsampling of very large trajectory files for practical training;
- command-line controls for the data path, training steps, and sample count;
- comparison plots for position density, energy, free energy, and generated velocity.

In addition, `PhaseSpaceDynamicsNet` is trained on consecutive saved frames. It receives the current standardized `(x, y, vx, vy, energy)` and predicts the next standardized state. This is the part that learns one-step dynamics from the MD trajectory. A rollout is included in the final plot.

During sampling, the five-dimensional state is updated jointly. At the end, generated energy is reset to $V(x,y)$ because energy is deterministic in this toy system. This keeps the reported energy physically consistent; it also makes clear that the current model is learning a phase-space equilibrium distribution, not discovering a new energy law.

## What this model does and does not learn

The score component learns a distribution over observed phase-space states. The dynamics component learns a one-step approximation to the temporal transition rule

$$
(x_t,v_t)\longrightarrow(x_{t+\Delta t},v_{t+\Delta t}).
$$

The score component's annealed-Langevin steps are sampling steps, not physical MD time steps. The learned dynamics rollout is time-directed, but it is only a one-step supervised approximation and has not yet been validated for long-term stability or physical fidelity. The diffusion coefficients printed by the script are still measured from the original MD trajectory.

## Next natural steps

1. Add held-out validation data and report score loss at every noise level.
2. Compare against the analytic positional equilibrium density $p(x,y)\propto\exp[-V(x,y)/(k_BT)]$, not only against a finite MD trajectory.
3. Test multiple chains, step sizes, noise schedules, and well-occupancy ratios.
4. Replace direct next-state regression with a probabilistic transition model or conditional diffusion model, so thermal uncertainty is represented rather than averaged away.
5. Validate the dynamics rollout against short-time displacement, velocity autocorrelation, energy drift, and barrier-crossing statistics.
6. Use the validated transition model to evaluate time correlations, barrier-crossing rates, and diffusion coefficients.
7. Add checkpointing, reproducible seeds, and experiment configuration files.

## What is needed for a full diffusion model?

A full score-based diffusion model needs a continuous or discrete noise/time variable $t$ and a score network $s_\theta(z,t)$ trained across a complete noise schedule. The current model already has the essential ingredients: noise-conditioned score matching and annealed Langevin sampling.

To make it a conventional diffusion implementation, the next pieces are:

- define a forward SDE or variance-preserving/noise-exploding schedule;
- train with the corresponding time-dependent score objective and loss weighting;
- start from the terminal prior at the largest noise level;
- solve the reverse-time SDE or probability-flow ODE, usually with a documented numerical solver;
- validate likelihood or sample quality across the whole schedule;
- separate equilibrium generation from physical dynamics unless the reverse process is explicitly designed to preserve the desired dynamics.

The filename `model_phase_space_score.py` describes the present capability without claiming that it is already a complete diffusion model.