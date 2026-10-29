"""
Dark Matter Substructure Engine for DeepLenseSim
================================================
Generates physical dark matter substructure models:
1. CDM (Cold Dark Matter): Population of discrete subhalos drawn from a power-law
   subhalo mass function (dN/dM ~ M^-1.9) distributed near the Einstein radius.
2. Axion (Ultra-Light Axion / Fuzzy Dark Matter): Wave interference vortex defects
   and localized potential fluctuations perturbing the critical curve.
3. No-Substructure: Pure smooth macro-model (control sample).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple
import numpy as np
from deeplense_agent.utils.schemas import (
    SubstructureConfig,
    SubstructureType,
    CDMSubstructureConfig,
    AxionSubstructureConfig,
)


def generate_substructure_profiles(
    config: SubstructureConfig,
    theta_E: float,
    z_lens: float,
    z_source: float,
    seed: Optional[int] = None,
) -> Tuple[List[str], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Constructs the lenstronomy lens model components for the requested dark matter substructure.

    Returns:
        lens_model_list: List of lenstronomy profile names (e.g. ['POINT_MASS', 'SIS', 'GAUSSIAN'])
        kwargs_lens_list: List of parameter dictionaries for each profile
        substructures_detail: Structured catalog metadata describing each perturber
    """
    if config.substructure_type == SubstructureType.NO_SUB:
        return [], [], []

    rng = np.random.default_rng(seed)

    if config.substructure_type == SubstructureType.CDM:
        return _generate_cdm_subhalos(
            config.cdm_config or CDMSubstructureConfig(),
            theta_E,
            rng,
        )

    elif config.substructure_type == SubstructureType.AXION:
        return _generate_axion_vortices(
            config.axion_config or AxionSubstructureConfig(),
            theta_E,
            rng,
        )

    return [], [], []


def _generate_cdm_subhalos(
    cdm_cfg: CDMSubstructureConfig,
    theta_E: float,
    rng: np.random.Generator,
) -> Tuple[List[str], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Generates CDM subhalos using SIS (Singular Isothermal Sphere) or POINT_MASS profiles.
    Subhalo positions are clustered around the Einstein ring where lensing sensitivity peaks.
    """
    lens_model_list: List[str] = []
    kwargs_lens_list: List[Dict[str, Any]] = []
    substructures_detail: List[Dict[str, Any]] = []

    m_min = cdm_cfg.min_mass_solar
    m_max = cdm_cfg.max_mass_solar
    n_sub = cdm_cfg.num_subhalos

    # Draw masses from power-law dN/dM ~ M^-1.9
    gamma_mass = -1.9
    u = rng.uniform(0.0, 1.0, size=n_sub)
    if gamma_mass != -1.0:
        exp = gamma_mass + 1.0
        masses = (m_min**exp + u * (m_max**exp - m_min**exp)) ** (1.0 / exp)
    else:
        masses = m_min * (m_max / m_min) ** u

    for i, mass in enumerate(masses):
        # Position subhalo near the Einstein ring: r ~ theta_E * [0.8, 1.2]
        r = theta_E * rng.uniform(0.75, 1.25)
        phi = rng.uniform(0.0, 2.0 * math.pi)
        x = float(r * math.cos(phi))
        y = float(r * math.sin(phi))

        # Scale subhalo Einstein radius based on mass: theta_E_sub ~ 0.05" * (M / 1e8 M_sun)^0.5
        theta_E_sub = float(0.04 * math.sqrt(mass / 1e8))
        theta_E_sub = max(0.005, min(0.20, theta_E_sub))

        lens_model_list.append("SIS")
        kwargs_lens_list.append({
            "theta_E": theta_E_sub,
            "center_x": x,
            "center_y": y,
        })

        substructures_detail.append({
            "subhalo_index": i + 1,
            "type": "cdm_sis_subhalo",
            "mass_solar": float(mass),
            "theta_E_arcsec": theta_E_sub,
            "center_x_arcsec": round(x, 4),
            "center_y_arcsec": round(y, 4),
            "distance_from_center_arcsec": round(float(r), 4),
        })

    return lens_model_list, kwargs_lens_list, substructures_detail


def _generate_axion_vortices(
    axion_cfg: AxionSubstructureConfig,
    theta_E: float,
    rng: np.random.Generator,
) -> Tuple[List[str], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Generates Ultra-Light Axion (ULA) vortex perturbations.
    Modeled as localized Gaussian potential fluctuations perturbing the primary lens potential.
    """
    lens_model_list: List[str] = []
    kwargs_lens_list: List[Dict[str, Any]] = []
    substructures_detail: List[Dict[str, Any]] = []

    sigma_core = axion_cfg.vortex_core_radius
    amp_base = axion_cfg.perturbation_amplitude
    n_vortices = axion_cfg.num_vortices

    for i in range(n_vortices):
        # Position vortex along the critical curve (Einstein ring)
        r = theta_E * rng.uniform(0.85, 1.15)
        phi = rng.uniform(0.0, 2.0 * math.pi)
        x = float(r * math.cos(phi))
        y = float(r * math.sin(phi))

        # Alternating sign or random phase representing constructive/destructive wave interference
        sign = rng.choice([-1.0, 1.0])
        amp = float(sign * amp_base * rng.uniform(0.8, 1.2))

        # In lenstronomy, GAUSSIAN provides a localized potential bump/well
        lens_model_list.append("GAUSSIAN")
        kwargs_lens_list.append({
            "amp": amp,
            "sigma": sigma_core,
            "center_x": x,
            "center_y": y,
        })

        substructures_detail.append({
            "vortex_index": i + 1,
            "type": "axion_wave_vortex",
            "amplitude": round(amp, 6),
            "core_radius_sigma_arcsec": sigma_core,
            "center_x_arcsec": round(x, 4),
            "center_y_arcsec": round(y, 4),
            "distance_from_center_arcsec": round(float(r), 4),
        })

    return lens_model_list, kwargs_lens_list, substructures_detail

