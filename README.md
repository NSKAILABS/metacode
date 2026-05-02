# MetaOpticsAI

**AI-Automated Photonics Design Platform for Metalens & Metasurface Engineering**

MetaOpticsAI is a Opensource Research Community in nanophotonics. 

The RCWA engine implements the formulation from:

> G. Yoon and J. Rho, "MAXIM: Metasurfaces-oriented electromagnetic wave simulation software with intuitive graphical user interfaces," *Computer Physics Communications*, vol. 264, p. 107846, 2021.

---


## Project Structure

```
metacode/
├── requirements.txt                 # Python dependencies
├── README.md
│
├── core/                            # Computation engines
│   ├── __init__.py
│   ├── meta_data.py                 # FDTD dataset management & interpolation
│   ├── phase_engine.py              # Phase mask generation (FZL, Axicon, SPP)
│   ├── gds_engine.py                # GDSII layout generation (gdsfactory 9.34.1)
│   ├── design.py                    # DesignState dataclass (pipeline state container)
│   └── rcwa_engine.py               # RCWA solver (MAXIM paper implementation)
|   |__ automl.py
│
├── analysis/                        # Optical performance metrics
│   ├── __init__.py
│   ├── psf.py                       # Point Spread Function (Fraunhofer diffraction)
│   ├── mtf.py                       # Modulation Transfer Function
│   ├── strehl.py                    # Strehl ratio (Maréchal criterion)
│   ├── encircled_energy.py          # Encircled energy curve
│   ├── zernike.py                   # Zernike polynomial decomposition (Noll indexing)
│   └── farfield.py                  # Far-field intensity at arbitrary distance
├── fdtd_data/                       # Directory for user-uploaded FDTD Excel files
├── assets/                          # Icons and images (placeholder)
└── output/                          # Generated GDS and plot files
```

---

## Installation

### Requirements

- **Python** 3.11.9
- **OS**: Windows 10/11 (also works on Linux/macOS with PyQt6 support)

### Setup

```bash
# Clone or download the project
cd MetaOpticsAI

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ≥ 1.24 | Numerical computation |
| scipy | ≥ 1.10 | Eigenvalue solvers, interpolation |
| matplotlib | ≥ 3.7 | Plotting (embedded in GUI) |
| gdsfactory | 9.34.1 | GDSII layout generation |
| openpyxl | ≥ 3.1 | Excel file reading (.xlsx) |
| xlrd | ≥ 2.0 | Legacy Excel file reading (.xls) |
| Pillow | ≥ 10.0 | Image loading for phase masks |


---

## Core Modules

### RCWA Engine (`core/rcwa_engine.py`)

Implements the full MAXIM paper formulation:

- **Fourier expansion** — Analytic sinc-based for rectangles (Eq. 39), FFT-based numerical method for circles and polygons (Section 3.3)
- **BTTB construction** — Block Toeplitz with Toeplitz Blocks matrix from Fourier coefficients (Eqs. 40–41)
- **Eigenvalue equation** — P and Q matrices (Eqs. 10–11), PQ eigenvalue problem (Eq. 9) via `scipy.linalg.eig`
- **Single-layer S-matrix** — Coupling coefficients and [Tf, Rb; Rf, Tb] matrix (Eqs. 22–28)
- **Redheffer star product** — Multilayer interconnection S^T = S^A ⊗ S^B (Eq. 29)
- **Boundary S-matrices** — Input/output media matching (Eqs. 33–37)
- **Diffraction efficiency** — Per-order calculation with proper kz normalization (Eq. 38)
- **Parametric sweep** — Sweep over radius, width, height, or wavelength with progress callback

### GDS Engine (`core/gds_engine.py`)

Uses **gdsfactory 9.34.1** (replacing the legacy gdspy):

- Component-based layout with `gf.Component`, `add_ref`, `dmove`
- Automatic PDK activation via `gf.gpdk.PDK.activate()`
- Unit cell caching for performance (identical dimensions reuse the same component)
- Supports Cylinder (circle), Cross (12-point polygon), and Fin (rotated rectangle) shapes

### Phase Engine (`core/phase_engine.py`)

- `fresnel_zone_lens()` — Hyperbolic phase profile with optional circular aperture
- `axicon()` — Conical lens for Bessel beam generation
- `spiral_phase_plate()` — Vortex beam generator with integer topological charge
- `from_image()` — Import grayscale image, map pixel values [0, 255] → [0, 2π)
- `quantize_phase()` — Discretize continuous phase into N uniformly-spaced levels

### Meta Data Store (`core/meta_data.py`)

Built-in FDTD library with 8 wavelengths:

| Wavelength | Material | Shape | Height (nm) | Period (nm) |
|------------|----------|-------|-------------|-------------|
| 405 nm | TiO₂ | Cylinder | 330 | 200 |
| 532 nm | TiO₂ | Fin | 600 | 325 |
| 633 nm | Silicon | Cylinder | 280 | 250 |
| 715 nm | Nb₂O₅ | Cross | 1050 | 500 |
| 850 nm | Silicon | Cylinder | 550 | 340 |
| 1064 nm | Silicon | Cylinder | 680 | 450 |
| 8.8 μm | Silicon | Cross | 1850 | 7250 |
| 410 μm | Silicon | Cylinder | 230000 | 200000 |

Supports user upload of custom Excel files (.xlsx/.xls) with dimension, phase, and transmission columns.

---

## Key Equations

| Name | Formula |
|------|---------|
| Fresnel Zone Lens | φ(r) = 2π − (2π/λ)(√(r² + f²) − f) mod 2π |
| Axicon | φ(r) = k(R − r)tan(α) mod 2π |
| Spiral Phase Plate | φ(θ) = l·θ mod 2π |
| PSF | \|FFT{A · exp(jφ)}\|² |
| MTF | \|FFT{PSF}\| / max |
| Strehl Ratio | max(PSF_aberrated) / max(PSF_ideal) |
| Encircled Energy | ∫∫_{r'<r} PSF / ∫∫ PSF |
| Zernike Decomposition | φ = Σ cⱼ Zⱼ(ρ, θ) |
| RCWA Eigenvalue (Eq. 9) | PQ [eₓ; eᵧ] = (jκ/k₀)² [eₓ; eᵧ] |
| Redheffer Star (Eq. 29) | S^T = S^A ⊗ S^B |
| Diffraction Efficiency (Eq. 38) | η = \|t_{mn}\|² Re(kz_{mn}) / Re(kz₀) |

---


## References

1. G. Yoon and J. Rho, "MAXIM: Metasurfaces-oriented electromagnetic wave simulation software with intuitive graphical user interfaces," *Comp. Phys. Comm.*, vol. 264, p. 107846, 2021.
2. M. Khorasaninejad et al., "Metalenses at visible wavelengths: Diffraction-limited focusing and subwavelength resolution imaging," *Science*, vol. 352, pp. 1190–1194, 2016.
3. N. Yu and F. Capasso, "Flat optics with designer metasurfaces," *Nature Materials*, vol. 13, pp. 139–150, 2014.
4. M. G. Moharam and T. K. Gaylord, "Rigorous coupled-wave analysis of planar-grating diffraction," *J. Opt. Soc. Am.*, vol. 71, pp. 811–818, 1981.
5. L. Li, "Formulation and comparison of two recursive matrix algorithms for modeling layered diffraction gratings," *J. Opt. Soc. Am. A*, vol. 13, pp. 1024–1035, 1996.
6. W. T. Chen et al., "A broadband achromatic metalens for focusing and imaging in the visible," *Nature Nanotechnology*, vol. 13, pp. 220–226, 2018.
7. A. Arbabi et al., "Dielectric metasurfaces for complete control of phase and polarization," *Nature Nanotechnology*, vol. 10, pp. 937–943, 2015.
8. S. Molesky et al., "Inverse design in nanophotonics," *Nature Photonics*, vol. 12, pp. 659–670, 2018.
9. A. Taflove and S. C. Hagness, *Computational Electrodynamics: The Finite-Difference Time-Domain Method*, 3rd ed., Artech House, 2005.
10. P. Lalanne and G. M. Morris, "Highly improved convergence of the coupled-wave method for TM polarization," *J. Opt. Soc. Am. A*, vol. 13, pp. 779–784, 1996.

---

## License

This project is developed for academic research and educational purposes at NIT Hamirpur under NSK AI Labs. Contact the maintainer for licensing inquiries.

---

## Author

**Dishant** — Co-Founder & CTO, Unisole Empower | B.Tech, NIT Hamirpur
Computational Physics · Nanophotonics · Scientific Machine Learning