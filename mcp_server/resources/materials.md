# Refractive-Index Library — Quick Reference

The bundled CSV tables in `metabox3/material_data/` are loaded by
`metabox3.rcwa.Material(name)` and exposed through the `get_material_index`
MCP tool. Each table provides *n* (real index) and, where applicable, *k*
(extinction coefficient) at hundreds of sampled wavelengths.

| Material  | Range (µm)     | n @ 532 nm | n @ 1550 nm | Notes                       |
|:----------|:---------------|-----------:|------------:|:----------------------------|
| TiO₂      | 0.43 – 1.53    | 2.66       | 2.45        | Visible workhorse, low loss |
| Si        | 0.21 – 0.83 (vis), 2 – 20 (IR) | 4.05 (532) | 3.47 | Telecom + NIR               |
| GaN       | 0.35 – 10      | 2.42       | 2.32        | UV/violet, wide bandgap     |
| Si₃N₄     | 0.31 – 5.5     | 2.06       | 1.99        | CMOS-compatible, visible    |
| ZnO       | 0.45 – 4.0     | 2.04       | 1.93        | Piezoelectric, transparent  |
| ZrO₂      | 0.36 – 5.1     | 2.18       | 2.10        | High-index, low loss vis    |
| Nb₂O₅     | 0.25 – 2.5     | 2.36       | 2.24        | High-k, sputtered films     |
| MoO₃      | 0.019 – 24     | 2.23       | 2.06        | Phonon-polariton material   |
| N-BK7     | 0.30 – 2.5     | 1.52       | 1.50        | Standard substrate glass    |
| Quartz    | 0.21 – 6.7     | 1.46       | 1.44        | UV-grade fused silica       |

## When to choose which

- **Visible (400–700 nm):** TiO₂ first (highest contrast, best phase
  coverage in 500–700 nm range). Si₃N₄ if CMOS process compatibility is
  required. GaN for blue/UV.
- **NIR (700–1700 nm):** crystalline Si is the default — cheap, high
  contrast, well-characterised.
- **Mid/Far IR:** MoO₃, GaN, or Si in the long-wavelength branch.
- **Substrates:** N-BK7 for visible imaging, quartz for UV transparency.

## How to use

```python
# Inside an MCP tool call
get_material_index(name="TiO2", wavelength_nm=532.0)
# → {"n": 2.66, "k": 0.0, "complex": "2.66+0j", "wavelength_nm": 532.0}
```

Out-of-range wavelengths raise an error — always check the
`Range` column above before requesting a value at the edge.

## Adding new materials

Drop a CSV with two header sections (`wl,n` then `wl,k`) into
`metabox3/material_data/`. Wavelengths are in **micrometres**. The new
material is auto-discovered by `list_materials` on next server restart.