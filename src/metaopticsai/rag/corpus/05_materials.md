# Materials & Wavelength Selection

## Visible (380–700 nm)
- **TiO₂**: n ≈ 2.35–2.5 across visible. Anatase deposited via ALD, then etched.
  Industry standard for visible metalenses.
- **GaN**: n ≈ 2.30–2.45. Higher temperature processing but compatible with
  III-V LED stacks for integrated emitters.
- **SiN**: n ≈ 2.0. Lower index → taller pillars needed for 2π coverage but
  much cheaper than TiO₂.

## Near-IR (700–1700 nm)
- **a-Si**: n ≈ 3.4–3.8, modest loss below 1 µm. Best phase tunability per
  unit footprint anywhere in this band.
- **c-Si**: lower loss than a-Si but harder to pattern.
- **Ge**: very high index (~4), use for MWIR (3–5 µm).

## SWIR / MWIR (1.7–5 µm)
- **Si** remains usable.
- **CaF₂, ZnSe** for refractive substrates.

## Loss budget
- Even k = 1e-3 contributes ~5 % per pillar in NIR. List k values from the
  Material CSVs and choose the lowest available in the operating band.

## Substrate index
- Fused silica (n ≈ 1.46) is the default. Higher-index substrates (sapphire
  n ≈ 1.77) raise the diffraction efficiency ceiling at the cost of harder
  AR coating.