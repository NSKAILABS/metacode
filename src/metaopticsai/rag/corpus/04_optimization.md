# Optimization Strategies

## Heuristic (analytical)
- FOM = mean_transmission² · min(1, phase_coverage / 2π).
- Free; produces a Strehl *estimate* not the true value.
- Useful for early sanity checks and to bound expectations.

## Gradient-based
- Build a `LensAssembly` with `set_structures_variable=True`.
- Choose `FigureOfMerit.LOG_STREHL_RATIO` for stable gradients near the
  diffraction limit.
- Adam with learning rate ~1e-9 (TF default scale for SI-unit positions) is
  a reasonable starting point.
- Watch for vanishing gradients far from any focal spot — initialize from
  the analytic ideal phase profile, not random.

## Surrogate-assisted
- Sample the parameter space (typically 30–100 RCWA evaluations).
- Train a small MLP / Fourier-feature network to predict (amplitude, phase).
- Run outer optimization through the surrogate (orders of magnitude faster).
- Re-validate the final design with full RCWA — surrogate error compounds at
  the parameter boundaries.

## Practical advice
- Always cap iterations: heuristic ≤ 20, gradient ≤ 200, surrogate ≤ 500.
- Save the FOM history alongside the final design for reproducibility.
- If Strehl < 0.5 after optimization, the meta-atom library is the
  bottleneck — go back to RCWA sweep and check phase coverage.