"""
Pydantic Schemas for DeepLenseSim Agentic Workflow
==================================================
Defines strongly-typed Pydantic v2 models for:
- Macro-lens deflector parameters (SIE, Shear)
- Background source galaxy parameters (Sérsic)
- Dark matter substructures (CDM subhalos and Axion wave vortex perturbations)
- Telescope & observation configurations (Model_I to Model_IV)
- Simulation requests and structured execution plans
- Ground-truth simulation metadata, observables, and file artifact paths
- Human-in-the-loop clarification items and user review contracts
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator, ValidationError


class ModelConfiguration(str, Enum):
    """DeepLenseSim Model tiers representing distinct astrophysical and instrument setups."""
    MODEL_I = "Model_I"      # Synthetic Baseline (idealized, SIE + Sérsic, 64x64/150x150)
    MODEL_II = "Model_II"    # Euclid Space Telescope (0.10"/pix, 0.16" PSF, VIS band noise)
    MODEL_III = "Model_III"  # Hubble Space Telescope (0.04"/pix, 0.08" PSF, WFC3/ACS high-res)
    MODEL_IV = "Model_IV"    # Complex morphologies / Multi-component starburst sources


class SubstructureType(str, Enum):
    """The dark matter substructure class to simulate."""
    NO_SUB = "no_sub"  # Smooth macro-model with no substructure (control group)
    CDM = "cdm"        # Cold Dark Matter subhalos (point mass / SIS subhalos)
    AXION = "axion"    # Ultra-Light Axion / wave dark matter vortex perturbations


class LensParameters(BaseModel):
    """Parameters defining the foreground deflector galaxy."""
    profile_type: str = Field(default="SIE", description="Mass profile (Singular Isothermal Ellipsoid)")
    theta_E: float = Field(default=1.25, ge=0.2, le=3.5, description="Einstein radius in arcseconds")
    center_x: float = Field(default=0.0, ge=-2.0, le=2.0, description="Lens center x in arcsec")
    center_y: float = Field(default=0.0, ge=-2.0, le=2.0, description="Lens center y in arcsec")
    e1: float = Field(default=0.10, ge=-0.5, le=0.5, description="Ellipticity component 1")
    e2: float = Field(default=0.05, ge=-0.5, le=0.5, description="Ellipticity component 2")
    gamma1: float = Field(default=0.03, ge=-0.2, le=0.2, description="External shear component 1")
    gamma2: float = Field(default=0.01, ge=-0.2, le=0.2, description="External shear component 2")


class SourceParameters(BaseModel):
    """Parameters defining the background source galaxy light distribution."""
    profile_type: str = Field(default="SERSIC_ELLIPSE", description="Surface brightness profile")
    amp: float = Field(default=100.0, gt=0, description="Source flux amplitude (intensity)")
    R_sersic: float = Field(default=0.25, ge=0.02, le=2.0, description="Half-light radius in arcseconds")
    n_sersic: float = Field(default=1.5, ge=0.5, le=6.0, description="Sérsic index (1.0=exponential disk, 4.0=elliptical)")
    center_x: float = Field(default=0.05, ge=-1.5, le=1.5, description="Source center offset x in arcsec")
    center_y: float = Field(default=0.02, ge=-1.5, le=1.5, description="Source center offset y in arcsec")
    e1: float = Field(default=0.05, ge=-0.5, le=0.5, description="Source ellipticity component 1")
    e2: float = Field(default=-0.05, ge=-0.5, le=0.5, description="Source ellipticity component 2")


class CDMSubstructureConfig(BaseModel):
    """Parameters for Cold Dark Matter (CDM) subhalos."""
    num_subhalos: int = Field(default=3, ge=1, le=20, description="Number of subhalos to inject")
    min_mass_solar: float = Field(default=1e7, ge=1e6, le=1e10, description="Minimum subhalo mass in M_sun")
    max_mass_solar: float = Field(default=1e9, ge=1e7, le=1e11, description="Maximum subhalo mass in M_sun")
    radial_placement: str = Field(default="near_einstein_ring", description="Subhalo spatial distribution")

    @model_validator(mode="after")
    def validate_mass_range(self) -> CDMSubstructureConfig:
        if self.min_mass_solar >= self.max_mass_solar:
            raise ValueError(f"min_mass_solar ({self.min_mass_solar:.1e}) must be < max_mass_solar ({self.max_mass_solar:.1e})")
        return self


class AxionSubstructureConfig(BaseModel):
    """Parameters for Ultra-Light Axion / wave dark matter vortex perturbations."""
    vortex_core_radius: float = Field(default=0.15, ge=0.02, le=0.40, description="Vortex core radius in arcsec")
    perturbation_amplitude: float = Field(default=0.015, ge=1e-4, le=0.10, description="Relative potential perturbation amplitude")
    num_vortices: int = Field(default=2, ge=1, le=8, description="Number of vortex centers")


class SubstructureConfig(BaseModel):
    """Container for dark matter substructure settings."""
    substructure_type: SubstructureType = Field(default=SubstructureType.NO_SUB)
    cdm_config: Optional[CDMSubstructureConfig] = None
    axion_config: Optional[AxionSubstructureConfig] = None

    @model_validator(mode="after")
    def sync_configs(self) -> SubstructureConfig:
        if self.substructure_type == SubstructureType.CDM and self.cdm_config is None:
            self.cdm_config = CDMSubstructureConfig()
        elif self.substructure_type == SubstructureType.AXION and self.axion_config is None:
            self.axion_config = AxionSubstructureConfig()
        return self


class ObservationConfig(BaseModel):
    """Telescope detector, optical PSF, and noise specifications."""
    telescope_name: str = Field(default="Generic Instrument", description="Name of instrument/telescope")
    pixel_scale: float = Field(ge=0.01, le=0.5, description="Pixel scale in arcsec/pixel")
    num_pix: int = Field(default=150, ge=32, le=512, description="Image grid dimensions (N x N pixels)")
    psf_fwhm: float = Field(ge=0.01, le=0.5, description="PSF full-width at half-maximum in arcsec")
    exposure_time: float = Field(gt=0, description="Exposure integration time in seconds")
    background_rms: float = Field(ge=0, description="Sky background noise standard deviation")

    @field_validator("num_pix")
    def ensure_even_grid(cls, v: int) -> int:
        if v % 2 != 0:
            raise ValueError(f"num_pix must be an even integer, got {v}")
        return v


class SimulationRequest(BaseModel):
    """
    Validated user simulation specification.
    """
    model_configuration: ModelConfiguration = Field(default=ModelConfiguration.MODEL_I)
    substructure_type: SubstructureType = Field(default=SubstructureType.NO_SUB)
    num_images: int = Field(default=1, ge=1, le=50, description="Number of images to generate")
    z_lens: float = Field(default=0.5, ge=0.05, le=2.0, description="Lens redshift")
    z_source: float = Field(default=1.5, ge=0.1, le=6.0, description="Source redshift")
    random_seed: Optional[int] = Field(default=42, description="Random seed for noise & realizations")

    lens: LensParameters = Field(default_factory=LensParameters)
    source: SourceParameters = Field(default_factory=SourceParameters)
    substructure: SubstructureConfig = Field(default_factory=SubstructureConfig)
    observation: ObservationConfig

    @model_validator(mode="after")
    def validate_physics(self) -> SimulationRequest:
        # Physical constraint: Lens must be closer to observer than source
        if self.z_lens >= self.z_source:
            raise ValueError(
                f"Physical impossibility: Lens redshift z_lens ({self.z_lens}) must be strictly less than "
                f"source redshift z_source ({self.z_source})."
            )
        # Geometrical constraint: Einstein radius should fit inside field of view
        fov_arcsec = self.observation.num_pix * self.observation.pixel_scale
        if self.lens.theta_E >= (fov_arcsec / 2.0):
            raise ValueError(
                f"Geometry error: Einstein radius theta_E ({self.lens.theta_E}\") exceeds half the "
                f"field of view ({fov_arcsec / 2.0:.2f}\" = {self.observation.num_pix} pix * {self.observation.pixel_scale}\"/pix)."
            )
        return self


# ==============================================================================
# Human-in-the-loop (HITL) Models
# ==============================================================================

class ClarificationQuestion(BaseModel):
    """A targeted question the agent asks to clarify missing parameters."""
    parameter_name: str = Field(description="Name of the missing or ambiguous parameter")
    question_text: str = Field(description="The question prompt for the human user")
    suggested_options: List[str] = Field(default_factory=list, description="Recommended answers")
    default_value: Any = Field(description="Default value used if user skips")


class ExecutionPlanPreview(BaseModel):
    """Summary of the simulation plan presented to the user for approval before running."""
    model_configuration: ModelConfiguration
    substructure_type: SubstructureType
    telescope_name: str
    num_images: int
    grid_resolution: str
    field_of_view_arcsec: float
    redshifts: str
    estimated_runtime_seconds: float
    summary_description: str
    ready_for_execution: bool = True


# ==============================================================================
# Simulation Output & Metadata Models
# ==============================================================================

class LensingObservables(BaseModel):
    """Astrophysical observables computed from the simulated ray-tracing."""
    total_flux: float = Field(description="Total integrated flux across image")
    peak_flux: float = Field(description="Peak pixel flux surface brightness")
    estimated_magnification: float = Field(description="Total flux magnification over unlensed source")
    einstein_radius_arcsec: float = Field(description="Measured / configured Einstein radius")
    num_substructures_injected: int = Field(default=0, description="Count of injected perturbers")


class SimulationMetadata(BaseModel):
    """Comprehensive ground truth metadata saved alongside generated images."""
    simulation_id: str
    timestamp: str
    model_configuration: ModelConfiguration
    substructure_type: SubstructureType
    cosmology: Dict[str, float]
    lens_parameters: Dict[str, Any]
    source_parameters: Dict[str, Any]
    observation_parameters: Dict[str, Any]
    substructures_detail: List[Dict[str, Any]] = Field(default_factory=list)
    observables: LensingObservables
    image_paths: Dict[str, str] = Field(description="Paths to generated FITS, NPY, and PNG files")
    random_seed: Optional[int] = None


# ==============================================================================
# Standardized DeepLenseSim Model Presets (Model_I to Model_IV)
# ==============================================================================

MODEL_PRESETS: Dict[ModelConfiguration, Dict[str, Any]] = {
    ModelConfiguration.MODEL_I: {
        "telescope_name": "DeepLense Baseline (Idealized Synthetic)",
        "pixel_scale": 0.05,       # arcsec / pixel
        "num_pix": 64,             # 64x64 or 150x150
        "psf_fwhm": 0.10,          # arcsec
        "exposure_time": 1000.0,   # seconds
        "background_rms": 0.010,
        "default_z_lens": 0.50,
        "default_z_source": 1.50,
        "default_theta_E": 1.20,
        "description": "Idealized synthetic lens with controlled Gaussian background noise; standard benchmark for ML representation learning.",
    },
    ModelConfiguration.MODEL_II: {
        "telescope_name": "Euclid Space Telescope (VIS Band)",
        "pixel_scale": 0.10,       # 0.10 arcsec / pixel for Euclid VIS detector
        "num_pix": 150,            # 150x150 grid (15" x 15" field of view)
        "psf_fwhm": 0.16,          # 0.16 arcsec FWHM diffraction-limited optical PSF
        "exposure_time": 1600.0,   # standard 4-dither survey exposure
        "background_rms": 0.008,
        "default_z_lens": 0.50,
        "default_z_source": 1.80,
        "default_theta_E": 1.30,
        "description": "Simulates wide-field space survey observation matching the ESA Euclid VIS instrument specifications.",
    },
    ModelConfiguration.MODEL_III: {
        "telescope_name": "Hubble Space Telescope (HST WFC3/ACS)",
        "pixel_scale": 0.04,       # 0.04 arcsec / pixel drizzled
        "num_pix": 150,            # 150x150 grid (6.0" x 6.0" field of view)
        "psf_fwhm": 0.08,          # sharp HST optical PSF
        "exposure_time": 2400.0,   # deep space orbit exposure
        "background_rms": 0.004,
        "default_z_lens": 0.40,
        "default_z_source": 2.00,
        "default_theta_E": 1.25,
        "description": "High-angular resolution space observation modeling the Hubble Space Telescope WFC3/ACS for fine arc distortion analysis.",
    },
    ModelConfiguration.MODEL_IV: {
        "telescope_name": "Complex Multi-Component Source Galaxy",
        "pixel_scale": 0.05,
        "num_pix": 150,            # 150x150 grid
        "psf_fwhm": 0.09,
        "exposure_time": 2000.0,
        "background_rms": 0.006,
        "default_z_lens": 0.45,
        "default_z_source": 1.70,
        "default_theta_E": 1.25,
        "description": "Realistic complex source morphologies with star-forming knots and Sérsic components testing domain adaptation.",
    },
}


def create_default_request(
    model_cfg: ModelConfiguration,
    substructure_type: SubstructureType = SubstructureType.NO_SUB,
    num_images: int = 1,
    num_pix: int = 150,
    random_seed: int = 42,
) -> SimulationRequest:
    """Creates a validated SimulationRequest initialized with standard instrument presets."""
    preset = MODEL_PRESETS[model_cfg]

    obs = ObservationConfig(
        telescope_name=preset["telescope_name"],
        pixel_scale=preset["pixel_scale"],
        num_pix=num_pix if num_pix else preset["num_pix"],
        psf_fwhm=preset["psf_fwhm"],
        exposure_time=preset["exposure_time"],
        background_rms=preset["background_rms"],
    )

    lens = LensParameters(
        theta_E=preset["default_theta_E"],
        e1=0.10,
        e2=0.05,
        gamma1=0.03,
        gamma2=0.01,
    )

    source = SourceParameters(
        amp=100.0,
        R_sersic=0.25,
        n_sersic=1.5,
        center_x=0.05,
        center_y=0.02,
        e1=0.05,
        e2=-0.05,
    )

    substructure = SubstructureConfig(substructure_type=substructure_type)

    return SimulationRequest(
        model_configuration=model_cfg,
        substructure_type=substructure_type,
        num_images=num_images,
        z_lens=preset["default_z_lens"],
        z_source=preset["default_z_source"],
        random_seed=random_seed,
        lens=lens,
        source=source,
        substructure=substructure,
        observation=obs,
    )


# ==============================================================================
# Comprehensive Physical Parameter Ranges & Formatting
# ==============================================================================

PARAMETER_RANGES: Dict[str, Dict[str, Any]] = {
    "theta_E": {
        "name": "Einstein Radius (θ_E)",
        "field": "lens.theta_E",
        "min": 0.20,
        "max": 3.50,
        "unit": "arcseconds (\")",
        "constraint": '0.20" ≤ θ_E ≤ 3.50" (and strictly θ_E < FOV / 2)',
        "description": "Deflector galaxy characteristic deflection radius. Typical values for massive ellipticals are 0.8\" to 2.0\". Must fit well inside the instrument field of view.",
    },
    "lens_e1": {
        "name": "Lens Ellipticity e1",
        "field": "lens.e1",
        "min": -0.50,
        "max": 0.50,
        "unit": "dimensionless",
        "constraint": "-0.50 ≤ e1 ≤ +0.50",
        "description": "Deflector mass distribution ellipticity component along coordinate axes.",
    },
    "lens_e2": {
        "name": "Lens Ellipticity e2",
        "field": "lens.e2",
        "min": -0.50,
        "max": 0.50,
        "unit": "dimensionless",
        "constraint": "-0.50 ≤ e2 ≤ +0.50",
        "description": "Deflector mass distribution ellipticity component at 45 degrees.",
    },
    "gamma1": {
        "name": "External Shear γ1",
        "field": "lens.gamma1",
        "min": -0.20,
        "max": 0.20,
        "unit": "dimensionless",
        "constraint": "-0.20 ≤ γ1 ≤ +0.20",
        "description": "Tidal gravitational shear from cosmic environment / cluster galaxies.",
    },
    "gamma2": {
        "name": "External Shear γ2",
        "field": "lens.gamma2",
        "min": -0.20,
        "max": 0.20,
        "unit": "dimensionless",
        "constraint": "-0.20 ≤ γ2 ≤ +0.20",
        "description": "Cross-component of external tidal gravitational shear.",
    },
    "lens_center_x": {
        "name": "Lens Center X",
        "field": "lens.center_x",
        "min": -2.00,
        "max": 2.00,
        "unit": "arcseconds (\")",
        "constraint": '-2.00" ≤ center_x ≤ +2.00"',
        "description": "Horizontal coordinate offset of deflector galaxy center from the optical axis.",
    },
    "lens_center_y": {
        "name": "Lens Center Y",
        "field": "lens.center_y",
        "min": -2.00,
        "max": 2.00,
        "unit": "arcseconds (\")",
        "constraint": '-2.00" ≤ center_y ≤ +2.00"',
        "description": "Vertical coordinate offset of deflector galaxy center from the optical axis.",
    },
    "amp": {
        "name": "Source Intensity Amplitude (amp)",
        "field": "source.amp",
        "min": 1e-4,
        "max": None,
        "unit": "flux counts",
        "constraint": "amp > 0 (Recommended: 10.0 to 2000.0, default 100.0)",
        "description": "Peak surface brightness flux amplitude of the background source galaxy. Must be strictly positive for physical photon emission.",
    },
    "R_sersic": {
        "name": "Source Half-Light Radius (R_sersic)",
        "field": "source.R_sersic",
        "min": 0.02,
        "max": 2.00,
        "unit": "arcseconds (\")",
        "constraint": '0.02" ≤ R_sersic ≤ 2.00"',
        "description": "Effective radius containing 50% of the background source galaxy light.",
    },
    "n_sersic": {
        "name": "Source Sérsic Index (n_sersic)",
        "field": "source.n_sersic",
        "min": 0.50,
        "max": 6.00,
        "unit": "dimensionless",
        "constraint": "0.50 ≤ n_sersic ≤ 6.00",
        "description": "Logarithmic slope of source surface brightness profile (n=1 for exponential disks, n=4 for de Vaucouleurs ellipticals).",
    },
    "source_center_x": {
        "name": "Source Center X",
        "field": "source.center_x",
        "min": -1.50,
        "max": 1.50,
        "unit": "arcseconds (\")",
        "constraint": '-1.50" ≤ center_x ≤ +1.50"',
        "description": "Source offset relative to deflector optical axis. Must be close to caustics for strong lensing arcs.",
    },
    "source_center_y": {
        "name": "Source Center Y",
        "field": "source.center_y",
        "min": -1.50,
        "max": 1.50,
        "unit": "arcseconds (\")",
        "constraint": '-1.50" ≤ center_y ≤ +1.50"',
        "description": "Source vertical offset relative to deflector optical axis.",
    },
    "source_e1": {
        "name": "Source Ellipticity e1",
        "field": "source.e1",
        "min": -0.50,
        "max": 0.50,
        "unit": "dimensionless",
        "constraint": "-0.50 ≤ e1 ≤ +0.50",
        "description": "Intrinsic source galaxy light ellipticity component 1.",
    },
    "source_e2": {
        "name": "Source Ellipticity e2",
        "field": "source.e2",
        "min": -0.50,
        "max": 0.50,
        "unit": "dimensionless",
        "constraint": "-0.50 ≤ e2 ≤ +0.50",
        "description": "Intrinsic source galaxy light ellipticity component 2.",
    },
    "z_lens": {
        "name": "Deflector Lens Redshift (z_lens)",
        "field": "z_lens",
        "min": 0.05,
        "max": 2.00,
        "unit": "dimensionless (z)",
        "constraint": "0.05 ≤ z_lens ≤ 2.00 (and strictly z_lens < z_source)",
        "description": "Cosmological redshift of foreground deflector galaxy. Must be strictly closer to observer than the background source galaxy.",
    },
    "z_source": {
        "name": "Background Source Redshift (z_source)",
        "field": "z_source",
        "min": 0.10,
        "max": 6.00,
        "unit": "dimensionless (z)",
        "constraint": "0.10 ≤ z_source ≤ 6.00 (and strictly z_source > z_lens)",
        "description": "Cosmological redshift of background lensed galaxy. Must be strictly behind the deflector galaxy.",
    },
    "num_pix": {
        "name": "Grid Dimensions (num_pix)",
        "field": "observation.num_pix",
        "min": 32,
        "max": 512,
        "unit": "pixels",
        "constraint": "32 ≤ num_pix ≤ 512 (MUST be an EVEN integer, e.g. 64, 128, 150, 256)",
        "description": "Image pixel grid resolution (N x N) for 2D ray-tracing and Fast Fourier Transform (FFT) PSF convolutions.",
    },
    "pixel_scale": {
        "name": "Pixel Scale",
        "field": "observation.pixel_scale",
        "min": 0.01,
        "max": 0.50,
        "unit": "arcsec/pixel",
        "constraint": '0.01" ≤ pixel_scale ≤ 0.50" / pixel',
        "description": "Detector angular plate scale (e.g. 0.04\" for HST WFC3, 0.10\" for Euclid VIS).",
    },
    "psf_fwhm": {
        "name": "PSF FWHM",
        "field": "observation.psf_fwhm",
        "min": 0.01,
        "max": 0.50,
        "unit": "arcseconds (\")",
        "constraint": '0.01" ≤ psf_fwhm ≤ 0.50"',
        "description": "Optical Point Spread Function Full-Width at Half-Maximum representing optical diffraction and seeing.",
    },
    "exposure_time": {
        "name": "Exposure Time",
        "field": "observation.exposure_time",
        "min": 1e-3,
        "max": None,
        "unit": "seconds (s)",
        "constraint": "exposure_time > 0 s (Recommended: 100.0 to 10000.0 s)",
        "description": "Sensor exposure integration time determining Poisson shot noise level.",
    },
    "background_rms": {
        "name": "Background Noise RMS",
        "field": "observation.background_rms",
        "min": 0.0,
        "max": None,
        "unit": "flux counts",
        "constraint": "background_rms ≥ 0.0 (Recommended: 0.001 to 0.05)",
        "description": "Standard deviation of Gaussian sky background noise.",
    },
    "num_images": {
        "name": "Number of Realizations (num_images)",
        "field": "num_images",
        "min": 1,
        "max": 50,
        "unit": "count",
        "constraint": "1 ≤ num_images ≤ 50 (Recommended: 1 to 5 for fast interactive response)",
        "description": "Number of independent mock lensing realizations with distinct random seeds to simulate.",
    },
    "substructure_type": {
        "name": "Dark Matter Substructure",
        "field": "substructure_type",
        "min": None,
        "max": None,
        "unit": "categorical",
        "constraint": "Allowed options: 'no_sub', 'cdm', 'axion'",
        "description": "'no_sub' (smooth macro control), 'cdm' (Cold Dark Matter subhalos), 'axion' (Ultra-Light Axion wave-vortex defects).",
    },
    "model_configuration": {
        "name": "Telescope / Instrument Tier",
        "field": "model_configuration",
        "min": None,
        "max": None,
        "unit": "categorical",
        "constraint": "Allowed options: 'Model_I', 'Model_II', 'Model_III', 'Model_IV'",
        "description": "Instrument preset: Model_I (Baseline 64x64), Model_II (Euclid VIS), Model_III (Hubble HST), Model_IV (Complex Starburst).",
    },
    "num_subhalos": {
        "name": "CDM Number of Subhalos",
        "field": "substructure.cdm_config.num_subhalos",
        "min": 1,
        "max": 20,
        "unit": "count",
        "constraint": "1 ≤ num_subhalos ≤ 20",
        "description": "Number of Cold Dark Matter subhalos injected into the deflector Einstein ring region.",
    },
    "vortex_core_radius": {
        "name": "Axion Vortex Core Radius",
        "field": "substructure.axion_config.vortex_core_radius",
        "min": 0.02,
        "max": 0.40,
        "unit": "arcseconds (\")",
        "constraint": '0.02" ≤ vortex_core_radius ≤ 0.40"',
        "description": "Characteristic quantum de Broglie wave-vortex interference core scale for Ultra-Light Axion dark matter.",
    },
    "perturbation_amplitude": {
        "name": "Axion Perturbation Amplitude",
        "field": "substructure.axion_config.perturbation_amplitude",
        "min": 0.0001,
        "max": 0.10,
        "unit": "dimensionless",
        "constraint": "0.0001 ≤ perturbation_amplitude ≤ 0.10",
        "description": "Relative gravitational potential perturbation scale from axion interference fluctuations.",
    },
    "num_vortices": {
        "name": "Axion Number of Vortices",
        "field": "substructure.axion_config.num_vortices",
        "min": 1,
        "max": 8,
        "unit": "count",
        "constraint": "1 ≤ num_vortices ≤ 8",
        "description": "Number of interfering quantum vortex nodes placed across the Einstein ring.",
    },
}


def resolve_parameter_keys(loc: tuple, msg: str) -> List[str]:
    """Resolves Pydantic error loc and msg into matching keys in PARAMETER_RANGES."""
    keys: List[str] = []
    loc_str = ".".join(str(x) for x in loc).lower()
    msg_lower = msg.lower()

    if "theta_e" in loc_str or "theta_e" in msg_lower or "einstein radius" in msg_lower:
        keys.append("theta_E")
    if "amp" in loc_str or "intensity" in msg_lower or "brightness" in msg_lower:
        keys.append("amp")
    if "r_sersic" in loc_str or "half-light" in msg_lower or "sersic radius" in msg_lower:
        keys.append("R_sersic")
    if "n_sersic" in loc_str or "sersic index" in msg_lower:
        keys.append("n_sersic")
    if "z_lens" in loc_str or "lens redshift" in msg_lower:
        keys.append("z_lens")
    if "z_source" in loc_str or "source redshift" in msg_lower:
        keys.append("z_source")
    if "num_pix" in loc_str or "num_pix" in msg_lower or "even integer" in msg_lower:
        keys.append("num_pix")
    if "pixel_scale" in loc_str or "pixel_scale" in msg_lower:
        keys.append("pixel_scale")
    if "psf_fwhm" in loc_str or "psf" in msg_lower:
        keys.append("psf_fwhm")
    if "exposure_time" in loc_str or "exposure" in msg_lower:
        keys.append("exposure_time")
    if "background_rms" in loc_str or "background" in msg_lower:
        keys.append("background_rms")
    if "num_images" in loc_str or "realization" in msg_lower:
        keys.append("num_images")
    if "substructure_type" in loc_str:
        keys.append("substructure_type")
    if "model_configuration" in loc_str:
        keys.append("model_configuration")

    # Lens / Source distinction for e1, e2, center
    if "e1" in loc_str:
        keys.append("lens_e1" if "lens" in loc_str else "source_e1")
    if "e2" in loc_str:
        keys.append("lens_e2" if "lens" in loc_str else "source_e2")
    if "center_x" in loc_str:
        keys.append("lens_center_x" if "lens" in loc_str else "source_center_x")
    if "center_y" in loc_str:
        keys.append("lens_center_y" if "lens" in loc_str else "source_center_y")
    if "gamma1" in loc_str:
        keys.append("gamma1")
    if "gamma2" in loc_str:
        keys.append("gamma2")

    # Root / Model validators
    if "physical impossibility" in msg_lower and "lens redshift" in msg_lower:
        if "z_lens" not in keys:
            keys.append("z_lens")
        if "z_source" not in keys:
            keys.append("z_source")

    if "geometry error" in msg_lower or "exceeds half the field of view" in msg_lower:
        if "theta_E" not in keys:
            keys.append("theta_E")
        if "num_pix" not in keys:
            keys.append("num_pix")

    if "vortex_core_radius" in loc_str:
        keys.append("vortex_core_radius")
    if "perturbation_amplitude" in loc_str:
        keys.append("perturbation_amplitude")
    if "num_vortices" in loc_str:
        keys.append("num_vortices")
    if "num_subhalos" in loc_str:
        keys.append("num_subhalos")

    return keys


def format_validation_error_with_ranges(error: Union[ValidationError, Exception, str]) -> str:
    """
    Translates a ValidationError or physical failure into an informative,
    astrophysically accurate message specifying the exact valid range.
    """
    lines = ["⚠️ **Invalid Physical Parameter Specified:**\n"]
    seen_params = set()

    if isinstance(error, ValidationError):
        for err in error.errors():
            loc = err.get("loc", ())
            msg = err.get("msg", "")
            inp = err.get("input", None)
            param_keys = resolve_parameter_keys(loc, msg)

            if param_keys:
                for k in param_keys:
                    if k in PARAMETER_RANGES and k not in seen_params:
                        seen_params.add(k)
                        info = PARAMETER_RANGES[k]
                        lines.append(f"• **{info['name']}** (`{info['field']}`)")
                        if inp is not None and not isinstance(inp, dict):
                            lines.append(f"  - **Attempted Value**: `{inp}`")
                        lines.append(f"  - **Correct Allowed Range**: `{info['constraint']}`")
                        lines.append(f"  - **Astrophysics**: {info['description']}")
                        if "physical impossibility" in msg.lower() or "geometry error" in msg.lower():
                            lines.append(f"  - **Physical Constraint**: *{msg}*")
                        lines.append("")
            else:
                lines.append(f"• **Validation Error**: `{msg}`\n")
    else:
        err_str = str(error)
        lines.append(f"• **Validation Error**: `{err_str}`\n")
        keys = resolve_parameter_keys((), err_str)
        for k in keys:
            if k in PARAMETER_RANGES and k not in seen_params:
                seen_params.add(k)
                info = PARAMETER_RANGES[k]
                lines.append(f"• **{info['name']}** (`{info['field']}`)")
                lines.append(f"  - **Correct Allowed Range**: `{info['constraint']}`")
                lines.append(f"  - **Astrophysics**: {info['description']}\n")

    lines.append("💡 *Please provide a value within the allowed physical range to proceed.*")
    return "\n".join(lines)


def get_parameter_range_summary(param_name: Optional[str] = None) -> str:
    """
    Returns markdown documentation for specific parameter range or all parameters.
    """
    if param_name and param_name.strip().lower() != "all":
        # Search for matching parameter
        p_clean = param_name.strip().lower()
        matched_k = None
        for k, info in PARAMETER_RANGES.items():
            if p_clean in k.lower() or p_clean in info["name"].lower() or p_clean in info["field"].lower():
                matched_k = k
                break

        if matched_k:
            info = PARAMETER_RANGES[matched_k]
            return (
                f"📏 **Physical Range Specification for {info['name']}:**\n\n"
                f"• **Field Path**: `{info['field']}`\n"
                f"• **Allowed Range**: `{info['constraint']}`\n"
                f"• **Units**: `{info['unit']}`\n"
                f"• **Astrophysics**: {info['description']}"
            )

    # Return full table of all parameters
    table = [
        "| Parameter | Field | Allowed Physical Range | Units | Notes |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]
    for k, info in PARAMETER_RANGES.items():
        name = info["name"]
        field = f"`{info['field']}`"
        constraint = f"`{info['constraint']}`"
        unit = info["unit"]
        notes = info["description"][:60] + "..."
        table.append(f"| {name} | {field} | {constraint} | {unit} | {notes} |")

    return (
        "### 🌌 DeepLenseSim Physical Parameter Specifications & Allowed Ranges\n\n"
        + "\n".join(table)
    )


RECOGNIZED_PARAMETER_NAMES: set[str] = {
    # Deflector Lens
    "theta_e", "einstein_radius", "theta",
    "e1", "lens_e1", "e2", "lens_e2",
    "gamma1", "shear1", "g1", "gamma2", "shear2", "g2", "y1", "y2",
    "center_x", "lens_center_x", "center_y", "lens_center_y", "center",
    "lens", "lenses", "deflector",
    # Source galaxy
    "amp", "intensity", "amplitude", "flux", "brightness",
    "r_sersic", "half_light_radius", "radius", "sersic_radius",
    "n_sersic", "sersic_index", "sersic_n", "index_n",
    "source_center_x", "source_center_y", "source_center",
    "source_e1", "source_e2", "source", "galaxy", "galaxies",
    # Cosmological redshifts
    "z_lens", "lens_redshift", "z_source", "source_redshift", "z_src", "redshift", "redshifts",
    # Optics & Observation
    "num_pix", "grid_resolution", "resolution", "pixels", "pix",
    "pixel_scale", "scale", "psf_fwhm", "psf", "fwhm",
    "exposure_time", "exposure", "exptime",
    "background_rms", "background_noise", "noise",
    # Presets, Dark Matter & Realizations
    "model", "models", "model_configuration", "telescope", "tier", "instrument",
    "model_i", "model_ii", "model_iii", "model_iv", "model 1", "model 2", "model 3", "model 4",
    "euclid", "hst", "hubble", "starburst", "startburst", "complex", "synthetic", "baseline",
    "substructure_type", "substructure", "substructures", "sub_structure", "sub structure",
    "no_sub", "cdm", "axion",
    "num_images", "realizations", "images", "samples", "random_seed", "seed",
    "num_subhalos", "vortex_core_radius", "perturbation_amplitude", "num_vortices",
    "system", "simulation", "simulations", "image", "observation", "mock",
}

IGNORED_PARAMETER_WORDS: set[str] = {
    "model", "models", "tier", "telescope", "instrument", "lens", "lenses",
    "deflector", "source", "galaxy", "galaxies", "system", "simulation", "simulations",
    "image", "images", "substructure", "substructures", "sub_structure", "sub structure",
    "perturber", "perturbers", "sample", "samples", "observation", "preset", "setup",
    "parameter", "parameters", "param", "params",
}


def check_for_unrecognized_parameters(text: str) -> Optional[str]:
    """
    Checks if a prompt specifies any parameter that is NOT in the recognized
    DeepLenseSim input parameters. If any unrecognized parameter assignment is found,
    returns its name; otherwise returns None.
    """
    import re
    t = text.lower().strip()
    if t in ["yes", "no", "proceed", "cancel", "ok", "run", "stop", "abort", "reset"]:
        return None

    # Match parameter assignment expressions like:
    # 1. "set/change/need/want/make <param> to/=/is <val>"
    # 2. "<param> = <val>"
    # 3. "<param> <number>"
    patterns = [
        r"(?:set|change|modify|need|want|use|make)\s+([a-zA-Z_][a-zA-Z0-9_\s]{0,25}?)\s*(?:=|to|is|of|:)\s*[-+]?[0-9]",
        r"([a-zA-Z_][a-zA-Z0-9_]{1,20})\s*(?:=|\:)\s*[-+]?[0-9]",
        r"(?:set|change|modify|need|want|use|make)\s+([a-zA-Z_][a-zA-Z0-9_]{1,20})\s+[-+]?[0-9]",
    ]
    for p in patterns:
        m = re.search(p, t)
        if m:
            cand = m.group(1).strip()
            cand_clean = re.sub(r"^(?:the|a|an)\s+", "", cand).strip().lower()
            if cand_clean in IGNORED_PARAMETER_WORDS:
                continue
            cand_tokens = cand_clean.replace("-", "_").split()
            if any(tok in IGNORED_PARAMETER_WORDS for tok in cand_tokens):
                continue
            cand_key = "_".join(cand_tokens)
            if cand_key not in RECOGNIZED_PARAMETER_NAMES and not any(k in RECOGNIZED_PARAMETER_NAMES for k in cand_tokens):
                return cand_clean

    return None

    return None


def format_unrecognized_parameter_error(param_name: str) -> str:
    """
    Returns an informative error message when an input parameter is not in recognized parameters.
    """
    return (
        f"⚠️ **Unrecognized Simulation Parameter:** `{param_name}` is not an input parameter to DeepLenseSim.\n\n"
        "**DeepLenseSim accepts only the following modifiable parameters:**\n"
        "• **Deflector (Lens)**: `theta_E` (Einstein radius), `e1`, `e2` (ellipticity), `gamma1`, `gamma2` (external shear), `center_x`, `center_y`\n"
        "• **Source Galaxy**: `amp` (intensity/brightness), `R_sersic` (half-light radius), `n_sersic` (Sérsic index), `center_x`, `center_y`, `e1`, `e2`\n"
        "• **Cosmology**: `z_lens` (deflector redshift), `z_source` (source redshift)\n"
        "• **Observation Optics**: `model_configuration` (Model_I to Model_IV), `num_pix`, `pixel_scale`, `psf_fwhm`, `exposure_time`, `background_rms`\n"
        "• **Dark Matter Substructures**: `substructure_type` (`no_sub`, `cdm`, `axion`), `num_images` (realizations), `random_seed`\n\n"
        "💡 *Click **📖 Parameter Reference** in the top right to view all 23 valid parameters and their physical ranges.*"
    )


