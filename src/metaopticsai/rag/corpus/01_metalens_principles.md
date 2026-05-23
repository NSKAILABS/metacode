# Metalens Design Principles

A metalens replaces a bulky refractive optic with a flat surface patterned by
sub-wavelength scatterers (meta-atoms). The phase profile across the surface
must mimic that of an ideal converging lens:

    φ(r) = (2π / λ) · (f − √(f² + r²))

where `r` is the radial coordinate, `f` is the focal length and `λ` is the
free-space wavelength. The phase is **wrapped** to the [0, 2π) interval and
imprinted by varying meta-atom geometry locally.

## Key figures of merit
- **Numerical aperture**: NA = sin(arctan(D / 2f)). High NA (>0.5) is hard for
  meta-atom phase libraries because they need both full 2π phase coverage *and*
  high transmission near the lens edge.
- **Strehl ratio**: |⟨e^{iφ_err}⟩|² over the aperture. Strehl > 0.8 is
  diffraction-limited; > 0.95 is excellent.
- **Focusing efficiency**: power in the diffraction-limited focal spot divided
  by total transmitted power.

## Design constraints
1. **Period < λ / NA_max** to suppress higher-order diffraction (sub-wavelength
   sampling Nyquist).
2. **Phase coverage ≥ 2π** must be achievable by sweeping the meta-atom geometry.
3. **Transmission > 0.7** averaged across the swept geometry is a healthy target.

## Common failure modes
- Insufficient phase coverage (< 0.9 × 2π) → severe focusing degradation.
- Inverse design without fabrication constraints → impossible-to-build features.
- Single-wavelength optimization → strong chromatic aberration off-design.