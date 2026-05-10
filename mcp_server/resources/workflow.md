# MetaOpticsAI — Photonics Workflow Guide

This is the canonical guide for chaining MetaOpticsAI tools end-to-end.
Always follow this order for a *new* metalens design from scratch:

```
  ┌────────────────────────────────────────────────────────────────────┐
  │  1. MATERIAL    →  list_materials, get_material_index              │
  │  2. UNIT CELL   →  run_rcwa_sweep   (phase-vs-diameter library)    │
  │  3. PHASE MASK  →  generate_phase_mask  (FZL / axicon / SPP)       │
  │  4. PROPAGATE   →  propagate_field, analyze_psf, analyze_mtf       │
  │  5. METRICS     →  zernike_decompose, compute_mtf_volume            │
  │  6. EXPORT      →  export_gds                                       │
  └────────────────────────────────────────────────────────────────────┘
```

Every tool that produces a large array (a sweep table, a 2-D phase mask, a
PSF) returns a short **handle** — pass that handle to downstream tools
instead of re-uploading the array.

---

## Stage 1 — Material selection
- Call `list_materials` first to see what refractive-index tables are
  bundled. The defaults cover **TiO₂, Si, GaN, Si₃N₄, ZnO, ZrO₂, Nb₂O₅,
  MoO₃, N-BK7, and quartz** across visible–IR ranges.
- Call `get_material_index(name, wavelength_nm)` to fetch the complex index
  *n + ik* at any wavelength inside the tabulated range.
- **Rule of thumb:** TiO₂ for visible, Si for NIR (≥ 900 nm), GaN for UV,
  Si₃N₄ for CMOS-compatible visible.

## Stage 2 — Unit-cell library
- Call `run_rcwa_sweep` with the chosen material, pillar height, and
  period. Targets:
  - **Phase coverage ≥ 0.95 × 2π** (otherwise increase `pillar_height_nm`)
  - **Mean amplitude ≥ 0.85** (otherwise the lens leaks energy)
- The returned handle is the design library. Pass it to
  `optimize_metalens` and `export_gds`.

## Stage 3 — Phase-mask generation
- For a focusing lens use `mask_type="fzl"` and supply `focal_length_um`.
- For a Bessel-beam axicon use `mask_type="axicon"` with `cone_angle_deg`.
- For a vortex-beam plate use `mask_type="spp"` with `spp_charge`.
- Always quantise (`n_levels=8` is a fabrication-friendly default).

## Stage 4 — Propagation & PSF
- Call `analyze_psf` on the mask handle to get the focal-plane PSF.
  Inspect Strehl ratio (target ≥ 0.8 — Maréchal criterion) and FWHM.
- Use `analyze_mtf` for spatial-frequency response.

## Stage 5 — Aberration diagnostics
- `zernike_decompose` reports the dominant Noll terms. If `Z₄`
  (defocus) is large, your focal length is off; if `Z₁₁` dominates,
  spherical aberration is the issue.

## Stage 6 — Fabrication-ready export
- `export_gds` writes a KLayout-clean GDSII file to `outputs/`.
- Default is cylindrical pillars; pass `shape="Cross"` or `"Fin"` for
  polarisation-sensitive designs.

---

## Heuristics for choosing parameters

| Wavelength | Material | Period (nm) | Pillar height (nm) | Comment              |
|-----------:|:---------|------------:|-------------------:|:---------------------|
|  405 nm    | TiO₂     | 200–250     | 500–600            | UV/blue              |
|  532 nm    | TiO₂     | 300–400     | 500–700            | green laser standard |
|  633 nm    | TiO₂/Si₃N₄ | 350–450   | 500–700            | He-Ne                |
|  850 nm    | Si       | 400–500     | 400–500            | NIR                  |
|  1550 nm   | Si       | 600–900     | 500–800            | telecom              |

## When NA matters
- NA ≤ 0.4 → forgiving; standard parameters work.
- NA 0.4 – 0.7 → tighten the diameter range and aim for ≥ 0.95 phase
  coverage.
- NA > 0.7 → expect Strehl drops; consider differentiable optimisation
  via `optimize_lens_assembly`.

## When something's off
- *Strehl < 0.5* → almost always insufficient phase coverage. Increase
  pillar height.
- *PSF too wide* → focal length mismatch or low NA.
- *Multiple side-lobes* → aliasing, increase `pixel_size_um` resolution
  or `pad_factor`.