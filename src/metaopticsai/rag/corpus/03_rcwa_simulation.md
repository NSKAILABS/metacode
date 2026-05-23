# RCWA Simulation Notes

Rigorous Coupled Wave Analysis solves Maxwell's equations in stratified
periodic structures by Fourier-decomposing the in-plane permittivity. Each
layer's S-matrix is built and the stack is then assembled.

## Key knobs
- **xy_harmonics**: number of plane-wave orders retained per direction.
  Convergence is geometry-dependent — typical values are 5–11. More harmonics
  are needed for high-index contrast and sharp corners.
- **resolution**: real-space sampling used to FFT-build the permittivity
  Fourier coefficients. Should be ≥ several samples per smallest feature.
- **minibatch_size**: how many parameter samples to evaluate at once. Larger =
  faster on GPU but more memory.

## Convergence checks
- **Energy conservation**: R + T + A ≈ 1 must hold at lossless interfaces.
  Drifts > 1 % signal under-resolved harmonics.
- **Reciprocity**: swap source/detector — fields must agree up to symmetry.

## Time / convention pitfalls
The two common time conventions (`exp(-iωt)` vs `exp(+iωt)`) flip the sign of
the imaginary part of refractive index and the phase. The handle-based
artifact system carries an explicit `time_convention` field so downstream
tools (analysis, propagation) can normalize.

## Backend choice
- **metabox3 backend**: differentiable, TF-based. Slow per-point but supports
  gradient descent through it.
- **S4 backend** (phoebe-p/S4 fork): mature reference C++ implementation,
  non-differentiable. Use for cross-validation, not optimization.
- **analytical backend**: closed-form approximation for early prototyping.
  Does NOT solve Maxwell's equations — use as a sanity check only.