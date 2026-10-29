"""
Core Gravitational Lensing Engine (lenstronomy Integration)
===========================================================
Executes gravitational ray-tracing, surface brightness evaluation,
PSF convolution, and noise realization using lenstronomy.
Computes physical lensing observables (flux, magnification, convergence kappa).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple
import numpy as np
from scipy.ndimage import gaussian_filter

from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Util import util

from deeplense_agent.utils.schemas import (
    SimulationRequest,
    ModelConfiguration,
    LensingObservables,
)
from deeplense_agent.core.substructure import generate_substructure_profiles


def simulate_lensing_realization(
    request: SimulationRequest,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Executes a complete physical ray-tracing simulation for a single realization.

    Returns a dictionary containing:
        image_observed: Final 2D numpy array with PSF convolution & noise
        image_clean: Noise-free ray-traced lensed light
        image_unlensed: Unlensed source light (for magnification reference)
        image_smooth_control: Realization without substructure (for ΔI residual)
        convergence_kappa: 2D convergence (surface mass density) map
        observables: LensingObservables object
        substructures_detail: Catalog of perturbers
    """
    effective_seed = seed if seed is not None else request.random_seed
    rng = np.random.default_rng(effective_seed)

    # 1. Coordinate Grid Setup
    num_pix = request.observation.num_pix
    delta_pix = request.observation.pixel_scale
    x_grid, y_grid = util.make_grid(num_pix=num_pix, delta_pix=delta_pix)
    x_2d = x_grid.reshape((num_pix, num_pix))
    y_2d = y_grid.reshape((num_pix, num_pix))

    # 2. Macro-Lens Deflector Model (SIE + External Shear)
    macro_lens_models = ["EPL", "SHEAR"]
    macro_kwargs_lens = [
        {
            "theta_E": request.lens.theta_E,
            "gamma": 2.0,  # Isothermal slope (gamma=2.0)
            "e1": request.lens.e1,
            "e2": request.lens.e2,
            "center_x": request.lens.center_x,
            "center_y": request.lens.center_y,
        },
        {
            "gamma1": request.lens.gamma1,
            "gamma2": request.lens.gamma2,
        },
    ]

    # 3. Substructure Realization
    sub_models, sub_kwargs, sub_details = generate_substructure_profiles(
        config=request.substructure,
        theta_E=request.lens.theta_E,
        z_lens=request.z_lens,
        z_source=request.z_source,
        seed=effective_seed,
    )

    full_lens_models = macro_lens_models + sub_models
    full_kwargs_lens = macro_kwargs_lens + sub_kwargs

    lens_model = LensModel(lens_model_list=full_lens_models)
    lens_model_smooth = LensModel(lens_model_list=macro_lens_models)

    # 4. Source Light Model
    # For Model_IV: include complex clumpy starburst knots
    source_model_list = ["SERSIC_ELLIPSE"]
    kwargs_source = [
        {
            "amp": request.source.amp,
            "R_sersic": request.source.R_sersic,
            "n_sersic": request.source.n_sersic,
            "e1": request.source.e1,
            "e2": request.source.e2,
            "center_x": request.source.center_x,
            "center_y": request.source.center_y,
        }
    ]

    if request.model_configuration == ModelConfiguration.MODEL_IV:
        # Add 3 clumpy star-forming knots surrounding source center
        clump_radii = [0.08, 0.05, 0.06]
        clump_amps = [25.0, 18.0, 20.0]
        offsets = [(0.10, 0.08), (-0.08, 0.12), (0.05, -0.10)]
        for r_clump, a_clump, (dx, dy) in zip(clump_radii, clump_amps, offsets):
            source_model_list.append("SERSIC_ELLIPSE")
            kwargs_source.append({
                "amp": a_clump,
                "R_sersic": r_clump,
                "n_sersic": 1.0,
                "e1": 0.0,
                "e2": 0.0,
                "center_x": request.source.center_x + dx,
                "center_y": request.source.center_y + dy,
            })

    light_model = LightModel(light_model_list=source_model_list)

    # 5. Ray-Tracing: Image Plane -> Source Plane
    x_source, y_source = lens_model.ray_shooting(x_grid, y_grid, full_kwargs_lens)
    flux_lensed_1d = light_model.surface_brightness(x_source, y_source, kwargs_source)
    image_clean = flux_lensed_1d.reshape((num_pix, num_pix))

    # Ray-trace smooth macro-model (for residual calculation)
    x_source_smooth, y_source_smooth = lens_model_smooth.ray_shooting(x_grid, y_grid, macro_kwargs_lens)
    flux_smooth_1d = light_model.surface_brightness(x_source_smooth, y_source_smooth, kwargs_source)
    image_smooth_control = flux_smooth_1d.reshape((num_pix, num_pix))

    # Unlensed source surface brightness (for magnification calculation)
    flux_unlensed_1d = light_model.surface_brightness(x_grid, y_grid, kwargs_source)
    image_unlensed = flux_unlensed_1d.reshape((num_pix, num_pix))

    # Convergence map kappa (projected mass density)
    kappa_1d = lens_model.kappa(x_grid, y_grid, full_kwargs_lens)
    convergence_kappa = kappa_1d.reshape((num_pix, num_pix))

    # 6. Telescope PSF Convolution
    # Convert PSF FWHM in arcseconds to pixel sigma: sigma_pix = FWHM / (2.355 * delta_pix)
    psf_sigma_pix = request.observation.psf_fwhm / (2.355 * delta_pix)
    image_convolved = gaussian_filter(image_clean, sigma=psf_sigma_pix)
    image_smooth_convolved = gaussian_filter(image_smooth_control, sigma=psf_sigma_pix)

    # 7. Noise Simulation (Poisson Shot Noise + Gaussian Sky Background)
    exp_time = request.observation.exposure_time
    bg_rms = request.observation.background_rms

    # Photon counts: photon_flux = flux * exposure_time
    photon_counts = np.maximum(0.0, image_convolved * exp_time)
    shot_noise_counts = rng.poisson(photon_counts)
    shot_noise_flux = shot_noise_counts / exp_time

    # Sky background noise
    sky_noise = rng.normal(loc=0.0, scale=bg_rms, size=(num_pix, num_pix))
    image_observed = shot_noise_flux + sky_noise

    # 8. Compute Lensing Observables
    total_lensed_flux = float(np.sum(image_clean) * (delta_pix ** 2))
    total_unlensed_flux = float(np.sum(image_unlensed) * (delta_pix ** 2))
    magnification = float(total_lensed_flux / max(1e-6, total_unlensed_flux))

    observables = LensingObservables(
        total_flux=round(total_lensed_flux, 3),
        peak_flux=round(float(np.max(image_observed)), 3),
        estimated_magnification=round(magnification, 2),
        einstein_radius_arcsec=request.lens.theta_E,
        num_substructures_injected=len(sub_details),
    )

    return {
        "image_observed": image_observed,
        "image_clean": image_clean,
        "image_unlensed": image_unlensed,
        "image_smooth_control": image_smooth_convolved,
        "convergence_kappa": convergence_kappa,
        "observables": observables,
        "substructures_detail": sub_details,
        "field_of_view_arcsec": num_pix * delta_pix,
    }
