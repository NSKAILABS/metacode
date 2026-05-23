"""Analysis tools — PSF, MTF, Zernike decomposition."""
from __future__ import annotations

import numpy as np

from metaopticsai.analysis import (
    compute_psf, compute_mtf, compute_strehl_ratio, zernike_decomposition,
)
from metaopticsai.tools.base import BaseTool
from metaopticsai.tools.schemas.analysis import PSFInput, MTFInput, ZernikeInput


class AnalyzePSFTool(BaseTool):
    name = "analyze_psf"
    description = (
        "Compute the point-spread function for a stored phase mask. "
        "Returns a handle to the PSF plus Strehl ratio and FWHM."
    )
    Input = PSFInput
    needs_store = True

    def _run(self, p: PSFInput):
        art = self.store.get(p.mask_handle, expected_kind="phase_mask")
        mask = art.payload

        psf, x_um, y_um = compute_psf(
            mask, p.wavelength_nm, p.pixel_size_um,
            p.focal_length_um, p.pad_factor,
        )
        strehl, _, _ = compute_strehl_ratio(mask, p.pad_factor)

        # FWHM along x through center
        row = psf[psf.shape[0] // 2, :]
        above = np.where(row >= row.max() / 2.0)[0]
        if len(above) and len(x_um) > 1:
            fwhm = float((above.max() - above.min()) * (x_um[1] - x_um[0]))
        else:
            fwhm = 0.0

        handle = self.store.put(
            "psf",
            {"psf_2d": psf, "x_um": x_um, "y_um": y_um,
             "strehl_ratio": float(strehl), "fwhm_um": fwhm},
            wavelength_nm=p.wavelength_nm,
        )
        return {
            "handle": handle,
            "strehl_ratio": float(strehl),
            "fwhm_x_um": round(fwhm, 4),
            "shape": list(psf.shape),
            "wavelength_nm": p.wavelength_nm,
        }


class AnalyzeMTFTool(BaseTool):
    name = "analyze_mtf"
    description = (
        "Compute the radially-averaged MTF and the diffraction-limit MTF "
        "from a stored PSF. Returns handle plus MTF50 (lp/mm)."
    )
    Input = MTFInput
    needs_store = True

    def _run(self, p: MTFInput):
        psf_art = self.store.get(p.psf_handle, expected_kind="psf")
        psf = psf_art.payload["psf_2d"]
        freqs, mtf_radial, mtf_diff = compute_mtf(
            psf, p.pixel_size_um, p.wavelength_nm, p.focal_length_um,
        )
        if (mtf_radial < 0.5).any():
            mtf50 = float(np.interp(0.5, mtf_radial[::-1], freqs[::-1]))
        else:
            mtf50 = 0.0
        handle = self.store.put(
            "mtf",
            {
                "freqs_lpmm": freqs.tolist(),
                "mtf_radial": mtf_radial.tolist(),
                "mtf_diffraction_limit": mtf_diff.tolist(),
            },
        )
        return {
            "handle": handle,
            "mtf50_lpmm": round(mtf50, 2),
            "cutoff_lpmm": float(freqs[-1]) if len(freqs) else 0.0,
        }


class ZernikeDecomposeTool(BaseTool):
    name = "zernike_decompose"
    description = (
        "Decompose a stored phase mask into Noll-indexed Zernike coefficients. "
        "Returns top aberrations + full coefficient handle."
    )
    Input = ZernikeInput
    needs_store = True

    def _run(self, p: ZernikeInput):
        art = self.store.get(p.mask_handle, expected_kind="phase_mask")
        coeffs, names = zernike_decomposition(art.payload, n_terms=p.n_terms)
        handle = self.store.put(
            "zernike", {"coefficients": coeffs, "names": names},
        )
        sorted_terms = sorted(coeffs.items(), key=lambda kv: abs(kv[1]), reverse=True)
        return {
            "handle": handle,
            "top_aberrations": [
                {"noll": j, "name": names.get(j, f"Z_{j}"), "coeff_rad": round(v, 4)}
                for j, v in sorted_terms[:5]
            ],
            "n_terms_analyzed": p.n_terms,
        }
