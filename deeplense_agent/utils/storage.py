"""
Data & Artifact Storage Utilities
=================================
Handles saving simulation outputs to standard formats:
- FITS (astronomy standard with physical header metadata)
- NPY (raw NumPy arrays for machine learning ingestion)
- PNG (contrast-stretched publication-ready previews)
- JSON (structured Pydantic metadata)
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from deeplense_agent.utils.schemas import SimulationMetadata, SimulationRequest


def export_simulation_artifacts(
    simulation_id: str,
    output_dir: str,
    request: SimulationRequest,
    sim_results: Dict[str, Any],
) -> SimulationMetadata:
    """
    Saves all scientific and visual artifacts for a simulation run.
    """
    os.makedirs(output_dir, exist_ok=True)
    img_data = sim_results["image_observed"].astype(np.float32)
    delta_pix = request.observation.pixel_scale

    # 1. Save NumPy array (.npy)
    npy_filename = f"{simulation_id}.npy"
    npy_path = os.path.join(output_dir, npy_filename)
    np.save(npy_path, img_data)

    # 2. Save Astronomical FITS file (.fits)
    fits_filename = f"{simulation_id}.fits"
    fits_path = os.path.join(output_dir, fits_filename)
    hdu = fits.PrimaryHDU(img_data)
    header = hdu.header
    header["SIM_ID"] = (simulation_id, "Unique simulation ID")
    header["MODEL"] = (request.model_configuration.value, "DeepLenseSim model tier")
    header["SUB_TYPE"] = (request.substructure.substructure_type.value, "Dark matter substructure class")
    header["TELESCOP"] = (request.observation.telescope_name[:20], "Simulated instrument")
    header["PIXSCALE"] = (delta_pix, "arcsec per pixel")
    header["Z_LENS"] = (request.z_lens, "Lens redshift")
    header["Z_SRC"] = (request.z_source, "Source redshift")
    header["THETA_E"] = (request.lens.theta_E, "Einstein radius (arcsec)")
    header["MAGNIF"] = (sim_results["observables"].estimated_magnification, "Lensing magnification")
    hdu.writeto(fits_path, overwrite=True)

    # 3. Save publication-quality PNG preview (.png)
    png_filename = f"{simulation_id}.png"
    png_path = os.path.join(output_dir, png_filename)
    _save_rendered_png(img_data, delta_pix, png_path, request)

    # 4. Save Multi-panel Diagnostic Plot (.png)
    diag_filename = f"{simulation_id}_diagnostic.png"
    diag_path = os.path.join(output_dir, diag_filename)
    _save_diagnostic_figure(sim_results, delta_pix, diag_path, request)

    # 5. Build and Save JSON metadata
    metadata = SimulationMetadata(
        simulation_id=simulation_id,
        timestamp=datetime.utcnow().isoformat() + "Z",
        model_configuration=request.model_configuration,
        substructure_type=request.substructure.substructure_type,
        cosmology={"z_lens": request.z_lens, "z_source": request.z_source},
        lens_parameters=request.lens.model_dump(),
        source_parameters=request.source.model_dump(),
        observation_parameters=request.observation.model_dump(),
        substructures_detail=sim_results["substructures_detail"],
        observables=sim_results["observables"],
        image_paths={
            "fits": fits_path,
            "npy": npy_path,
            "png": png_path,
            "diagnostic": diag_path,
        },
        random_seed=request.random_seed,
    )

    json_path = os.path.join(output_dir, f"{simulation_id}_metadata.json")
    with open(json_path, "w") as f:
        json.dump(metadata.model_dump(), f, indent=2)

    return metadata


def _save_rendered_png(img: np.ndarray, delta_pix: float, save_path: str, request: SimulationRequest):
    """Renders a single high-contrast image preview with arcsecond coordinates."""
    num_pix = img.shape[0]
    extent_half = (num_pix * delta_pix) / 2.0
    extent = [-extent_half, extent_half, -extent_half, extent_half]

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    # Log-like contrast stretch
    stretched = np.arcsinh(img * 2.0)
    im = ax.imshow(stretched, origin="lower", extent=extent, cmap="inferno")
    ax.set_title(
        f"{request.model_configuration.value} | {request.substructure.substructure_type.value.upper()}\n"
        f"{request.observation.telescope_name}",
        fontsize=10,
        pad=10,
    )
    ax.set_xlabel("Arcseconds", fontsize=9)
    ax.set_ylabel("Arcseconds", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="arcsinh(Flux)")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def _save_diagnostic_figure(sim_results: Dict[str, Any], delta_pix: float, save_path: str, request: SimulationRequest):
    """
    Renders a 4-panel diagnostic figure:
    1. Observed Mock Image (with PSF & noise)
    2. Clean Lensed Light (noise-free)
    3. Convergence map kappa (mass distribution)
    4. Substructure residual map ΔI = I_sub - I_no_sub
    """
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), dpi=130)
    num_pix = sim_results["image_observed"].shape[0]
    extent_half = (num_pix * delta_pix) / 2.0
    extent = [-extent_half, extent_half, -extent_half, extent_half]

    # Panel 1: Observed Image
    im0 = axes[0].imshow(np.arcsinh(sim_results["image_observed"] * 2.0), origin="lower", extent=extent, cmap="inferno")
    axes[0].set_title(f"Observed Image ({request.model_configuration.value})", fontsize=10)
    axes[0].set_xlabel("Arcseconds")
    axes[0].set_ylabel("Arcseconds")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Panel 2: Clean Ray-traced Light
    im1 = axes[1].imshow(sim_results["image_clean"], origin="lower", extent=extent, cmap="inferno")
    axes[1].set_title("Clean Lensed Light", fontsize=10)
    axes[1].set_xlabel("Arcseconds")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3: Convergence Mass Density (kappa)
    im2 = axes[2].imshow(sim_results["convergence_kappa"], origin="lower", extent=extent, cmap="magma")
    axes[2].set_title("Mass Density (Convergence $\\kappa$)", fontsize=10)
    axes[2].set_xlabel("Arcseconds")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    # Panel 4: Substructure Residual (ΔI = I - I_smooth)
    residual = sim_results["image_clean"] - sim_results["image_smooth_control"]
    vmax = max(1e-4, float(np.max(np.abs(residual))))
    im3 = axes[3].imshow(residual, origin="lower", extent=extent, cmap="bwr", vmin=-vmax, vmax=vmax)
    axes[3].set_title("Substructure Residual $\\Delta I$", fontsize=10)
    axes[3].set_xlabel("Arcseconds")
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    plt.suptitle(
        f"DeepLenseSim Diagnostic: Substructure = {request.substructure.substructure_type.value.upper()} | "
        f"Magnification = {sim_results['observables'].estimated_magnification}x",
        fontsize=12,
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

