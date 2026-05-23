# Fabrication & GDS Export

## Process flow (typical TiO₂ visible metalens)
1. **Substrate**: 500 µm fused silica, RCA cleaned.
2. **Resist**: 200–400 nm PMMA or hydrogen silsesquioxane (HSQ).
3. **Lithography**: e-beam (EBL) for prototyping, deep-UV / NIL for volume.
4. **Pattern transfer**: lift-off (PMMA) or direct etch (HSQ).
5. **TiO₂ deposition**: ALD to fill the patterned resist for lift-off; or
   blanket ALD + RIE for top-down etch.
6. **AR coating**: optional SiO₂ planarization for high-NA designs.

## GDS file structure
- **Single cell** named after the design with all polygons at the same layer
  for simple processes.
- **Layer assignment**: layer 1 = pillar top-down; layer 2 (optional) = AR
  mask; layer 100 = chip-frame / alignment marks.
- **Units**: nanometers; gdspy precision = 1 nm.
- **DRC**: minimum feature size 30 nm for 100-keV EBL; 80 nm for 365 nm DUV.

## Quantization
The continuous phase profile is binned into `n_levels` (default 8) discrete
phases. Each level maps to a meta-atom geometry from the RCWA sweep that best
matches the target phase. Trade-off:
- Few levels (4) → strong quantization noise, easier fabrication.
- Many levels (16+) → near-continuous phase but tightens DRC.

## Aperture shape
- **Circular**: standard for rotationally-symmetric optics.
- **Rectangular**: easier dicing; off-axis chromatic correction works better
  with anamorphic apertures.
- The `is_circular` flag in `export_gds` controls this.

## Sanity checks before tape-out
1. Visualize the central 10 µm × 10 µm region — look for missing pillars.
2. Confirm the bounding box matches `diameter_um × diameter_um`.
3. Spot-check pillar diameters against the sweep table.
4. Confirm file size is sane (< 50 MB for a typical 100-µm lens).