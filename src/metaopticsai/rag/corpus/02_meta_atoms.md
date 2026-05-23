# Meta-Atom Selection

The unit cell ("meta-atom") is the elementary scatterer. Its geometry determines
the phase φ and amplitude t imprinted on the wavefront at every position on the
metalens.

## Geometric families

### Circular pillars (Cylinder)
- **Variable**: diameter `d`.
- **Pro**: simplest; isotropic response.
- **Con**: limited phase tunability per unit-cell footprint; height usually
  needs to be tall to cover 2π.

### Rectangular pillars (Fin)
- **Variables**: `(w_x, w_y)`.
- **Pro**: birefringence allows polarization-dependent operation.
- **Con**: 2-D sweep at design time; sensitivity to fabrication anisotropy.

### Cross / + shaped (Cross)
- **Variables**: arm widths.
- **Pro**: large polarizability per footprint → shorter pillars feasible.
- **Con**: more demanding lithography.

## Materials at visible / NIR
- **TiO₂** (n ≈ 2.35–2.5 in VIS): low loss, EBL- or NIL-friendly.
- **GaN** (n ≈ 2.32): visible, integrates with III-V LEDs.
- **a-Si** / **c-Si** (n ≈ 3.5): NIR / SWIR — strong dispersion, lossy below
  ~700 nm.
- **SiN** (n ≈ 2.0): visible / NIR; lossless; modest phase tunability.

## Height / period heuristics
- `period ≈ 0.4–0.6 × λ / n_substrate` for a NA ~0.5 lens.
- `height` chosen so the max-diameter pillar accumulates 2π phase:
  `h ≈ λ / (n_pillar − 1)` is a useful first guess.