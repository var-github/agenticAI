"""
DeepLenseSim AI Studio — Streamlit Front-End
============================================
Interactive web application for strong gravitational lensing simulations powered by:
- Pydantic AI & Groq LLM (qwen/qwen3.8-27b)
- Conversational Human-in-the-Loop (HITL) Chat Workflow
- lenstronomy ray-tracing engine
- Model_I to Model_IV instrument benchmarks
- Cold Dark Matter (CDM) subhalos & Ultra-Light Axion wave-vortex perturbers
- Multi-format artifact downloads (FITS, NPY, PNG, JSON)
"""

import os
import sys
import json
import glob
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import ValidationError

import streamlit as st
import numpy as np
from dotenv import load_dotenv

# Ensure project root is in sys.path
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

from deeplense_agent.utils.schemas import (
    ModelConfiguration,
    SubstructureType,
    SimulationRequest,
    SimulationMetadata,
    ExecutionPlanPreview,
    LensParameters,
    SourceParameters,
    ObservationConfig,
    SubstructureConfig,
    MODEL_PRESETS,
    create_default_request,
    PARAMETER_RANGES,
    format_validation_error_with_ranges,
    get_parameter_range_summary,
    check_for_unrecognized_parameters,
    format_unrecognized_parameter_error,
    IGNORED_PARAMETER_WORDS,
)
from deeplense_agent.core.pipeline import DeepLensePipeline
from deeplense_agent.agent.workflow import (
    create_deeplense_agent,
    inspect_model_capabilities,
    validate_simulation_plan,
)
from deeplense_agent.agent.hitl import clarify_simulation_parameters

# ==============================================================================
# Page Configuration & Styling
# ==============================================================================

st.set_page_config(
    page_title="DeepLenseSim AI Studio",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="collapsed",
)

OUTPUT_DIR = "sim_output"

st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.2rem;
    }

    /* Sticky Parameter Reference Button - locked in same position on screen as you scroll */
    div.st-key-sticky_ref_btn,
    div[class*="st-key-sticky_ref_btn"],
    div.st-key-btn_param_ref,
    div[class*="st-key-btn_param_ref"] {
        position: fixed !important;
        top: 55px !important;
        right: 2.2rem !important;
        z-index: 999999 !important;
        width: auto !important;
    }
    div.st-key-sticky_ref_btn button,
    div.st-key-btn_param_ref button,
    div[class*="st-key-sticky_ref_btn"] button,
    div[class*="st-key-btn_param_ref"] button {
        position: fixed !important;
        top: 55px !important;
        right: 2.2rem !important;
        z-index: 999999 !important;
        width: auto !important;
        background: #FFFFFF !important;
        border: 1.5px solid #4F46E5 !important;
        color: #4F46E5 !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.22) !important;
        padding: 0.45rem 1.1rem !important;
        transition: all 0.2s ease-in-out !important;
        cursor: pointer !important;
    }
    div.st-key-sticky_ref_btn button:hover,
    div.st-key-btn_param_ref button:hover,
    div[class*="st-key-sticky_ref_btn"] button:hover,
    div[class*="st-key-btn_param_ref"] button:hover {
        background: #4F46E5 !important;
        color: #FFFFFF !important;
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.45) !important;
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# Helper Functions for Simulation Results Display
# ==============================================================================

def render_realization_artifacts(r: SimulationMetadata, unique_prefix: str = ""):
    """Renders metrics, images, and download buttons for a single realization."""
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Magnification", f"{r.observables.estimated_magnification:.2f}x")
    m2.metric("Total Flux", f"{r.observables.total_flux:.1f}")
    m3.metric("Peak Flux", f"{r.observables.peak_flux:.1f}")
    m4.metric("Injected Perturbers", r.observables.num_substructures_injected)

    col_img, col_diag = st.columns([1, 1.4])
    with col_img:
        st.markdown("**Mock Observed Lens Image** (PSF + Noise)")
        if os.path.exists(r.image_paths.get("png", "")):
            st.image(r.image_paths["png"], use_container_width=True)
    with col_diag:
        st.markdown("**4-Panel Astrophysical Diagnostic Plot**")
        if os.path.exists(r.image_paths.get("diagnostic", "")):
            st.image(r.image_paths["diagnostic"], use_container_width=True)

    d1, d2, d3, d4 = st.columns(4)
    if os.path.exists(r.image_paths.get("fits", "")):
        with open(r.image_paths["fits"], "rb") as f:
            d1.download_button(
                label="📥 FITS Data",
                data=f.read(),
                file_name=os.path.basename(r.image_paths["fits"]),
                mime="image/fits",
                key=f"{unique_prefix}_fits_{r.simulation_id}",
            )
    if os.path.exists(r.image_paths.get("npy", "")):
        with open(r.image_paths["npy"], "rb") as f:
            d2.download_button(
                label="📥 NumPy Array",
                data=f.read(),
                file_name=os.path.basename(r.image_paths["npy"]),
                mime="application/octet-stream",
                key=f"{unique_prefix}_npy_{r.simulation_id}",
            )
    if os.path.exists(r.image_paths.get("png", "")):
        with open(r.image_paths["png"], "rb") as f:
            d3.download_button(
                label="📥 PNG Preview",
                data=f.read(),
                file_name=os.path.basename(r.image_paths["png"]),
                mime="image/png",
                key=f"{unique_prefix}_png_{r.simulation_id}",
            )
    json_path = os.path.join(
        os.path.dirname(r.image_paths["fits"]),
        f"{r.simulation_id}_metadata.json",
    )
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            meta_str = f.read()
        d4.download_button(
            label="📥 Metadata JSON",
            data=meta_str,
            file_name=f"{r.simulation_id}_metadata.json",
            mime="application/json",
            key=f"{unique_prefix}_json_{r.simulation_id}",
        )


def diff_simulation_requests(old_req: SimulationRequest, new_req: SimulationRequest) -> List[str]:
    """Compares two SimulationRequests and returns a list of human-readable differences."""
    diffs = []
    if old_req.model_configuration != new_req.model_configuration:
        diffs.append(f"• **Telescope**: `{old_req.model_configuration.value}` → `{new_req.model_configuration.value}`")
    if old_req.substructure_type != new_req.substructure_type:
        diffs.append(f"• **Substructure**: `{old_req.substructure_type.value}` → `{new_req.substructure_type.value}`")
    if old_req.num_images != new_req.num_images:
        diffs.append(f"• **Realizations**: `{old_req.num_images}` → `{new_req.num_images}`")
    if old_req.z_lens != new_req.z_lens:
        diffs.append(f"• **Lens Redshift z_lens**: `{old_req.z_lens}` → `{new_req.z_lens}`")
    if old_req.z_source != new_req.z_source:
        diffs.append(f"• **Source Redshift z_source**: `{old_req.z_source}` → `{new_req.z_source}`")

    # Lens attributes
    for attr in ["theta_E", "e1", "e2", "gamma1", "gamma2", "center_x", "center_y"]:
        v_old = getattr(old_req.lens, attr, None)
        v_new = getattr(new_req.lens, attr, None)
        if v_old != v_new and v_old is not None and v_new is not None:
            diffs.append(f"• **Lens {attr}**: `{v_old}` → `{v_new}`")

    # Source attributes
    for attr in ["amp", "R_sersic", "n_sersic", "center_x", "center_y", "e1", "e2"]:
        v_old = getattr(old_req.source, attr, None)
        v_new = getattr(new_req.source, attr, None)
        if v_old != v_new and v_old is not None and v_new is not None:
            label = "Source Intensity (amp)" if attr == "amp" else f"Source {attr}"
            diffs.append(f"• **{label}**: `{v_old}` → `{v_new}`")

    # Observation attributes
    for attr in ["num_pix", "pixel_scale", "psf_fwhm", "exposure_time", "background_rms"]:
        v_old = getattr(old_req.observation, attr, None)
        v_new = getattr(new_req.observation, attr, None)
        if v_old != v_new and v_old is not None and v_new is not None:
            diffs.append(f"• **Observation {attr}**: `{v_old}` → `{v_new}`")

    return diffs


def apply_plan_modifications(
    base_req: SimulationRequest,
    text: str,
) -> Tuple[Optional[SimulationRequest], Optional[str]]:
    """
    Applies natural language modifications to an existing pending SimulationRequest.
    Uses Groq LLM (qwen/qwen3.8-27b) for full semantic understanding of ANY physical parameter,
    with an internal pattern-matcher fallback.

    Returns:
        (updated_request, None) if modifications are physically valid.
        (None, error_message_with_ranges) if any invalid parameter is specified.
    """
    # 1. Try Groq LLM for complete semantic understanding across all parameters
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        try:
            from openai import OpenAI
            client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=groq_key)
            curr_json = base_req.model_dump_json(indent=2)
            prompt = (
                "You are an expert astrophysics parameter modifier for strong gravitational lensing simulations.\n"
                "Given the CURRENT_REQUEST JSON below, update the parameters requested by the user.\n"
                "CRITICAL INSTRUCTION FOR UNRECOGNIZED PARAMETERS:\n"
                "The ONLY recognized simulation input parameters for DeepLenseSim are: theta_E, e1, e2, gamma1, gamma2, center_x, center_y (or center), "
                "amp (intensity), R_sersic (half-light radius), n_sersic, z_lens, z_source, num_pix, pixel_scale, psf_fwhm, exposure_time, background_rms, "
                "model_configuration (tier), substructure_type (no_sub, cdm, axion), num_images (realizations), random_seed, num_subhalos, vortex_core_radius, perturbation_amplitude, num_vortices.\n"
                "RULES FOR UNRECOGNIZED PARAMETERS:\n"
                "- Ordinary conversational words, entity nouns, and command verbs (e.g. 'model', 'models', 'lens', 'lenses', 'system', 'substructure', 'sub structure', 'telescope', 'galaxy', 'simulation', 'image', 'mock') are NEVER unrecognized parameters.\n"
                "- 'model' refers to model_configuration (e.g. Model_I to Model_IV).\n"
                "- ONLY return 'unrecognized_parameter' if the user explicitly attempts to set an unsupported physics property that is not modeled in gravitational lensing ray-tracing (for example: temperature, metallicity, spin, dark energy w, velocity dispersion, color, interstellar dust, etc.).\n"
                "If an unrecognized parameter is detected, return: {\"unrecognized_parameter\": \"<parameter_name>\"}.\n\n"
                "CRITICAL INSTRUCTION FOR VALID SIMULATION PARAMETERS:\n"
                "You MUST apply the user's requested parameter changes directly into the JSON fields, "
                "EVEN IF the values are extreme, invalid, or out-of-bounds (for example: 'center 100' -> set lens.center_x=100.0 and lens.center_y=100.0; "
                "'theta_e 15' -> set lens.theta_E=15.0; 'intensity -50' -> set source.amp=-50.0).\n"
                "Our backend validator will evaluate physical limits and provide feedback to the user. "
                "NEVER silently ignore, omit, or reset the user's requested modification.\n"
                "If the user says 'center' without specifying x or y, apply to both center_x and center_y. "
                "If the user says 'center' without specifying lens or source, apply to lens center (or source center if source is mentioned).\n"
                "Return ONLY a valid JSON object matching the SimulationRequest schema (or the unrecognized_parameter JSON object).\n\n"
                f"CURRENT_REQUEST:\n{curr_json}\n\n"
                f"USER_MODIFICATION:\n{text}\n"
            )
            resp = client.chat.completions.create(
                model="qwen/qwen3.8-27b",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            content = resp.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.strip("`").lstrip("json").strip()
            try:
                parsed_dict = json.loads(content)
                if "unrecognized_parameter" in parsed_dict:
                    param = str(parsed_dict.get("unrecognized_parameter", "")).strip().lower()
                    if param not in IGNORED_PARAMETER_WORDS and param:
                        return None, format_unrecognized_parameter_error(param)

                updated = SimulationRequest.model_validate(parsed_dict)
                # If nothing changed but the user attempted a modification, fall through to fallback
                if updated.model_dump() != base_req.model_dump():
                    return updated, None
            except ValidationError as val_err:
                return None, format_validation_error_with_ranges(val_err)
        except Exception:
            pass  # Fall through to pattern matcher

    # 2. Fast pattern-matcher fallback
    t = text.lower()

    # Check for unrecognized parameters in text
    unrec = check_for_unrecognized_parameters(text)
    if unrec:
        return None, format_unrecognized_parameter_error(unrec)
    req_dict = base_req.model_dump()

    # 1. Dark matter substructure
    if any(k in t for k in ["axion", "vortex", "wave"]):
        req_dict["substructure_type"] = SubstructureType.AXION.value
        req_dict["substructure"] = {"substructure_type": SubstructureType.AXION.value}
    elif any(k in t for k in ["cdm", "subhalo", "cold dark matter"]):
        req_dict["substructure_type"] = SubstructureType.CDM.value
        req_dict["substructure"] = {"substructure_type": SubstructureType.CDM.value}
    elif any(k in t for k in ["no_sub", "smooth", "macro", "control"]):
        req_dict["substructure_type"] = SubstructureType.NO_SUB.value
        req_dict["substructure"] = {"substructure_type": SubstructureType.NO_SUB.value}

    # 2. Telescope / Model Tier
    new_tier = None
    if any(k in t for k in ["model_iv", "model 4", "model iv", "complex", "starburst"]):
        new_tier = ModelConfiguration.MODEL_IV
    elif any(k in t for k in ["model_iii", "model 3", "model iii", "hst", "hubble"]):
        new_tier = ModelConfiguration.MODEL_III
    elif any(k in t for k in ["model_ii", "model 2", "model ii", "euclid"]):
        new_tier = ModelConfiguration.MODEL_II
    elif any(k in t for k in ["model_i", "model 1", "model i", "synthetic", "baseline"]):
        new_tier = ModelConfiguration.MODEL_I

    if new_tier and new_tier.value != req_dict["model_configuration"]:
        preset = MODEL_PRESETS[new_tier]
        req_dict["model_configuration"] = new_tier.value
        req_dict["observation"]["telescope_name"] = preset["telescope_name"]
        req_dict["observation"]["pixel_scale"] = preset["pixel_scale"]
        req_dict["observation"]["psf_fwhm"] = preset["psf_fwhm"]
        req_dict["observation"]["exposure_time"] = preset["exposure_time"]
        req_dict["observation"]["background_rms"] = preset["background_rms"]
        req_dict["z_lens"] = preset["default_z_lens"]
        req_dict["z_source"] = preset["default_z_source"]

    # 3. Realization counts
    num_match = re.search(r"(\d+)\s*(?:[a-z0-9_-]+\s*)?(?:images|samples|realizations|simulations|lenses)", t)
    if num_match:
        req_dict["num_images"] = int(num_match.group(1))
    elif "single" in t or "one image" in t:
        req_dict["num_images"] = 1

    # 4. Grid resolution
    res_match = re.search(r"(\d+)\s*(?:x\s*(\d+)|\s*pixels?|\s*resolution)", t)
    if res_match:
        req_dict["observation"]["num_pix"] = int(res_match.group(1))

    # 5. Einstein Radius theta_E
    theta_match = re.search(r"(?:theta_?e|einstein radius|theta)\s*(?:=|to|is|of)?\s*([+-]?[0-9]*\.?[0-9]+)", t)
    if theta_match:
        req_dict["lens"]["theta_E"] = float(theta_match.group(1))

    # 6. Source Intensity / Amplitude
    amp_match = re.search(r"(?:source\s+)?(?:intensity|amplitude|amp|flux|brightness)\s*(?:=|to|is|of)?\s*([+-]?[0-9]*\.?[0-9]+)", t)
    if amp_match:
        req_dict["source"]["amp"] = float(amp_match.group(1))

    # 7. Source Sérsic Radius
    r_match = re.search(r"(?:r_?sersic|half[- ]light radius|source radius|sersic radius)\s*(?:=|to|is|of)?\s*([+-]?[0-9]*\.?[0-9]+)", t)
    if r_match:
        req_dict["source"]["R_sersic"] = float(r_match.group(1))

    # 8. Source Sérsic Index
    n_match = re.search(r"(?:n_?sersic|sersic index|index n|sersic n)\s*(?:=|to|is|of)?\s*([+-]?[0-9]*\.?[0-9]+)", t)
    if n_match:
        req_dict["source"]["n_sersic"] = float(n_match.group(1))

    # 9. Cosmological Redshifts
    zl_match = re.search(r"(?:z_?lens|lens redshift)\s*(?:=|to|is|of)?\s*([+-]?[0-9]*\.?[0-9]+)", t)
    if zl_match:
        req_dict["z_lens"] = float(zl_match.group(1))

    zs_match = re.search(r"(?:z_?source|source redshift|z_?src)\s*(?:=|to|is|of)?\s*([+-]?[0-9]*\.?[0-9]+)", t)
    if zs_match:
        req_dict["z_source"] = float(zs_match.group(1))

    # 10. Exposure Time
    exp_match = re.search(r"(?:exposure(?: time)?|exptime)\s*(?:=|to|is|of)?\s*([+-]?[0-9]*\.?[0-9]+)", t)
    if exp_match:
        req_dict["observation"]["exposure_time"] = float(exp_match.group(1))

    # 11. Ellipticities & Shears
    e1_match = re.search(r"(?:lens\s+)?e1\s*(?:=|to|is|of)?\s*([+-]?[0-9]*\.?[0-9]+)", t)
    if e1_match:
        req_dict["lens"]["e1"] = float(e1_match.group(1))

    e2_match = re.search(r"(?:lens\s+)?e2\s*(?:=|to|is|of)?\s*([+-]?[0-9]*\.?[0-9]+)", t)
    if e2_match:
        req_dict["lens"]["e2"] = float(e2_match.group(1))

    g1_match = re.search(r"(?:gamma1|shear1|g1)\s*(?:=|to|is|of)?\s*([+-]?[0-9]*\.?[0-9]+)", t)
    if g1_match:
        req_dict["lens"]["gamma1"] = float(g1_match.group(1))

    g2_match = re.search(r"(?:gamma2|shear2|g2)\s*(?:=|to|is|of)?\s*([+-]?[0-9]*\.?[0-9]+)", t)
    if g2_match:
        req_dict["lens"]["gamma2"] = float(g2_match.group(1))

    # 12. Spatial Centers (center, center_x, center_y)
    target = "source" if ("source" in t and "center" in t) else "lens"
    cx_match = re.search(r"(?:center_?x|x[-_ ]center)\s*(?:=|to|is|of)?\s*([+-]?[0-9]*\.?[0-9]+)", t)
    if cx_match:
        req_dict[target]["center_x"] = float(cx_match.group(1))

    cy_match = re.search(r"(?:center_?y|y[-_ ]center)\s*(?:=|to|is|of)?\s*([+-]?[0-9]*\.?[0-9]+)", t)
    if cy_match:
        req_dict[target]["center_y"] = float(cy_match.group(1))

    if not cx_match and not cy_match and "center" in t:
        c_pair_match = re.search(r"center\s*(?:=|to|is|of)?\s*\(?([+-]?[0-9]*\.?[0-9]+)\s*,\s*([+-]?[0-9]*\.?[0-9]+)\)?", t)
        if c_pair_match:
            req_dict[target]["center_x"] = float(c_pair_match.group(1))
            req_dict[target]["center_y"] = float(c_pair_match.group(2))
        else:
            c_single_match = re.search(r"center\s*(?:=|to|is|of)?\s*([+-]?[0-9]*\.?[0-9]+)", t)
            if c_single_match:
                val = float(c_single_match.group(1))
                req_dict[target]["center_x"] = val
                req_dict[target]["center_y"] = val

    # Validate resulting dictionary against SimulationRequest schema
    try:
        updated = SimulationRequest.model_validate(req_dict)
        if updated.model_dump() == base_req.model_dump():
            unrec = check_for_unrecognized_parameters(text)
            if unrec:
                return None, format_unrecognized_parameter_error(unrec)
            return None, "💡 *No parameter changes were recognized in your input. Try specifying a parameter like 'theta_E = 1.35', 'source intensity 200', or click '📖 Parameter Reference'.*"
        return updated, None
    except ValidationError as val_err:
        return None, format_validation_error_with_ranges(val_err)
    except Exception as err:
        return None, format_validation_error_with_ranges(err)


def parse_initial_simulation_request(text: str) -> Tuple[Optional[SimulationRequest], Optional[str], bool]:
    """
    Parses a user's initial simulation prompt into a validated SimulationRequest.
    Uses Groq LLM (qwen/qwen3.8-27b) with robust typo tolerance (e.g. startburst -> Model_IV,
    no sub structure -> no_sub, y2 -> gamma2), with an internal pattern-matcher fallback.

    Returns:
        (req, None, False): Valid request formulated.
        (None, error_msg, False): Out-of-bounds parameter detected (e.g. y2 = 1.1).
        (None, None, True): Prompt is completely underspecified (needs HITL clarification).
    """
    t = text.lower().strip()
    clean_t = re.sub(r"^(?:please\s+|can\s+you\s+|i\s+want\s+to\s+|help\s+me\s+)", "", t).strip()
    vague_commands = [
        "simulate", "simulate a lens", "simulate lens", "run simulation",
        "simulate gravitational lens", "generate lens", "start simulation",
        "create lens", "model a lens", "run", "simulate a system"
    ]
    if clean_t in vague_commands:
        return None, None, True

    # 1. Groq LLM Semantic Extraction
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        try:
            from openai import OpenAI
            client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=groq_key)
            base_template = create_default_request(ModelConfiguration.MODEL_I).model_dump_json(indent=2)
            prompt = (
                "You are an expert astrophysics simulation architect for strong gravitational lensing simulations.\n"
                "Your task is to parse the USER_PROMPT and generate a complete, JSON-serialized SimulationRequest.\n"
                "Rules:\n"
                "1. Understand natural language variations, typos, and astrophysics terms:\n"
                "   - 'startburst' / 'starburst' / 'complex' -> model_configuration: 'Model_IV'\n"
                "   - 'euclid' -> model_configuration: 'Model_II'\n"
                "   - 'hst' / 'hubble' -> model_configuration: 'Model_III'\n"
                "   - 'synthetic' / 'baseline' -> model_configuration: 'Model_I'\n"
                "   - 'no sub structure' / 'no substructure' / 'smooth' / 'macro' -> substructure_type: 'no_sub'\n"
                "   - 'cdm' / 'subhalo' -> substructure_type: 'cdm'\n"
                "   - 'axion' / 'vortex' / 'wave' -> substructure_type: 'axion'\n"
                "   - 'y2' / 'shear2' / 'gamma2' -> lens.gamma2\n"
                "   - 'y1' / 'shear1' / 'gamma1' -> lens.gamma1\n"
                "   - 'half light radius' / 'radius' / 'R_sersic' -> source.R_sersic\n"
                "   - 'redshift' -> if single redshift, set z_lens (or z_lens and z_source if two given)\n"
                "   - 'intensity' / 'brightness' -> source.amp\n"
                "   - 'center' -> lens.center_x and lens.center_y (or source.center_x/y if source center)\n"
                "2. When a telescope tier is selected, initialize the corresponding observation preset (e.g. Model_IV: 150x150, 0.05\"/pix, 2000s exposure; Model_III: 0.04\"/pix, 2400s; Model_II: 0.10\"/pix, 1600s).\n"
                "3. CRITICAL: Always apply the user's requested parameter values into the JSON fields EVEN IF they are outside valid physical ranges "
                "(e.g. gamma2=1.1, theta_E=15.0, amp=-50.0). Our backend validator will evaluate them.\n"
                "4. CRITICAL FOR UNRECOGNIZED PARAMETERS:\n"
                "   - Ordinary conversational words, entity nouns, and command verbs (e.g. 'model', 'models', 'lens', 'lenses', 'system', 'substructure', 'sub structure', 'telescope', 'galaxy', 'simulation', 'image', 'mock', 'sample') are NEVER unrecognized parameters.\n"
                "   - 'model' refers to model_configuration (e.g. Model_I to Model_IV).\n"
                "   - ONLY return 'unrecognized_parameter' if the user explicitly attempts to configure an unsupported physics property not modeled in gravitational lensing ray-tracing (such as temperature, metallicity, spin, dark energy, velocity dispersion, color, dust, etc.).\n"
                "5. Clarification & Underspecification:\n"
                "   - Return {\"underspecified\": true} if the user prompt lacks a telescope tier (no mention of Model_I..Model_IV, Euclid, HST, Starburst, synthetic) and does not specify specific physical parameters.\n"
                "   - Return {\"underspecified\": true} if the prompt is completely generic without specifying a telescope tier or substructure (e.g. 'simulate', 'run simulation', 'create lens').\n"
                "6. Return ONLY a valid JSON object matching the SimulationRequest schema (or the unrecognized_parameter / underspecified JSON object).\n\n"
                f"DEFAULT_TEMPLATE:\n{base_template}\n\n"
                f"USER_PROMPT:\n{text}\n"
            )
            resp = client.chat.completions.create(
                model="qwen/qwen3.8-27b",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            content = resp.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.strip("`").lstrip("json").strip()
            try:
                parsed_dict = json.loads(content)
                if "unrecognized_parameter" in parsed_dict:
                    param = str(parsed_dict.get("unrecognized_parameter", "")).strip().lower()
                    if param not in IGNORED_PARAMETER_WORDS and param:
                        return None, format_unrecognized_parameter_error(param), False
                if parsed_dict.get("underspecified"):
                    return None, None, True

                req = SimulationRequest.model_validate(parsed_dict)
                return req, None, False
            except ValidationError as val_err:
                return None, format_validation_error_with_ranges(val_err), False
        except Exception:
            pass  # Fall through to pattern matcher

    # 2. Fast Pattern-Matcher Fallback
    unrec = check_for_unrecognized_parameters(text)
    if unrec:
        return None, format_unrecognized_parameter_error(unrec), False
    has_sub = any(s in t for s in ["cdm", "axion", "no_sub", "no sub", "without substructure", "smooth", "macro", "control", "subhalo", "vortex"])
    has_tier = any(m in t for m in ["model_i", "model_ii", "model_iii", "model_iv", "model 1", "model 2", "model 3", "model 4", "euclid", "hst", "hubble", "starburst", "startburst", "complex", "synthetic"])
    has_params = any(p in t for p in ["theta", "radius", "shear", "y1", "y2", "gamma", "redshift", "z_lens", "z_source", "intensity", "amp", "center", "pix", "resolution"])

    if (not has_tier and not has_params) or (not has_sub and not has_tier):
        return None, None, True  # Genuinely underspecified

    tier = ModelConfiguration.MODEL_I
    if any(k in t for k in ["starburst", "startburst", "complex", "model_iv", "model 4"]):
        tier = ModelConfiguration.MODEL_IV
    elif any(k in t for k in ["hst", "hubble", "model_iii", "model 3"]):
        tier = ModelConfiguration.MODEL_III
    elif any(k in t for k in ["euclid", "model_ii", "model 2"]):
        tier = ModelConfiguration.MODEL_II

    sub_type = SubstructureType.NO_SUB
    if any(k in t for k in ["cdm", "subhalo", "cold dark matter"]):
        sub_type = SubstructureType.CDM
    elif any(k in t for k in ["axion", "vortex", "wave"]):
        sub_type = SubstructureType.AXION

    base_req = create_default_request(model_cfg=tier, substructure_type=sub_type)
    req, err = apply_plan_modifications(base_req, text)
    if err:
        return None, err, False
    return req or base_req, None, False


# ==============================================================================
# Parameter Reference Content & Dialog
# ==============================================================================

def render_parameter_reference_content(key_prefix: str = "ref"):
    """Renders comprehensive physical parameter reference documentation."""
    st.markdown("### 🌌 DeepLenseSim Physical Parameter Reference & Bounds")
    st.markdown(
        "Complete reference guide for all physical, geometrical, instrument, and substructure parameters. "
        "Any of these parameters can be directly configured in the **Manual Playground** or modified in natural language via the **Chat Agent** "
        "(e.g. *\"change theta_E to 1.4\"*, *\"need source intensity 250\"*, *\"center lens to (0.1, -0.05)\"*, *\"set z_lens to 0.4\"*)."
    )

    search = st.text_input(
        "🔍 Search parameters or keywords...",
        placeholder="Filter by name, unit, or keyword (e.g. theta_E, intensity, redshift, center, axion)...",
        key=f"{key_prefix}_search",
    )
    search_lower = search.strip().lower() if search else ""

    categories = {
        "🔭 1. Deflector (Macro-Lens) Geometry (SIE + Shear)": [
            "theta_E", "lens_e1", "lens_e2", "gamma1", "gamma2", "lens_center_x", "lens_center_y"
        ],
        "🌌 2. Background Source Galaxy (Sérsic Profile)": [
            "amp", "R_sersic", "n_sersic", "source_center_x", "source_center_y", "source_e1", "source_e2"
        ],
        "🪐 3. Cosmological Redshifts": [
            "z_lens", "z_source"
        ],
        "📷 4. Telescope Optics & Sensor Grid": [
            "num_pix", "pixel_scale", "psf_fwhm", "exposure_time", "background_rms"
        ],
        "🌀 5. Dark Matter Substructure Models": [
            "substructure_type", "num_subhalos", "vortex_core_radius", "perturbation_amplitude", "num_vortices"
        ],
        "⚙️ 6. Instrument Presets & Realizations": [
            "model_configuration", "num_images"
        ],
    }

    found_any = False
    for cat_title, param_keys in categories.items():
        matched_keys = []
        for k in param_keys:
            if k in PARAMETER_RANGES:
                info = PARAMETER_RANGES[k]
                text_to_match = f"{k} {info['name']} {info['field']} {info['constraint']} {info['unit']} {info['description']}".lower()
                if not search_lower or search_lower in text_to_match:
                    matched_keys.append(k)

        if matched_keys:
            found_any = True
            st.markdown(f"#### {cat_title}")
            for k in matched_keys:
                info = PARAMETER_RANGES[k]
                with st.container(border=True):
                    c1, c2, c3 = st.columns([1.5, 1.8, 1.2])
                    with c1:
                        st.markdown(f"**{info['name']}**")
                        st.code(info['field'], language="python")
                    with c2:
                        st.markdown(f"**Allowed Range:** `{info['constraint']}`")
                        st.caption(f"**Units:** `{info['unit']}`")
                    with c3:
                        clean_field = info['field'].split('.')[-1]
                        if "theta" in k.lower():
                            example_val = "1.4"
                        elif "amp" in k.lower():
                            example_val = "250"
                        elif "center" in k.lower():
                            example_val = "0.1"
                        elif "z_lens" in k.lower():
                            example_val = "0.4"
                        elif "z_source" in k.lower():
                            example_val = "2.2"
                        elif "num" in k.lower():
                            example_val = "2"
                        else:
                            example_val = "valid value"
                        st.markdown("**Chat Example:**")
                        st.caption(f"*\"set {clean_field} to {example_val}\"*")

                    st.markdown(f"**Astrophysics:** {info['description']}")

    if not found_any:
        st.warning(f"No parameters matched '{search}'. Try searching for 'theta', 'intensity', 'redshift', 'center', or 'substructure'.")

    st.divider()
    st.markdown("#### 💬 Quick Natural Language Cheat Sheet")
    st.markdown("""
    You can adjust any parameter on an active plan directly from chat. Examples:
    - `change theta_e to 1.35` — Sets deflector Einstein radius
    - `need source intensity 200` — Sets background galaxy brightness
    - `center lens to (0.05, -0.02)` — Offsets deflector center coordinates
    - `set z_lens to 0.4 and z_source to 2.2` — Updates deflector and source cosmological redshifts
    - `change to axion substructure` — Switches dark matter perturber to quantum wave vortices
    - `simulate 3 realizations` — Generates 3 mock images with randomized seeds
    - `switch to model_iii` or `use hubble telescope` — Reconfigures optical detector to HST drizzled specs
    """)


@st.dialog("📖 DeepLenseSim Physical Parameter & Range Reference", width="large")
def show_parameter_reference_dialog():
    render_parameter_reference_content(key_prefix="dialog_ref")


# ==============================================================================
# Header & Tabs
# ==============================================================================

st.markdown('<div class="main-title">🌌 DeepLenseSim AI Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Conversational Gravitational Lensing powered by Pydantic AI, Groq LLM, & lenstronomy</div>', unsafe_allow_html=True)

# Sticky locked Parameter Reference button
with st.container(key="sticky_ref_btn"):
    if st.button("📖 Parameter Reference", key="btn_param_ref", type="secondary"):
        show_parameter_reference_dialog()

tab_chat, tab_manual, tab_gallery = st.tabs([
    "Chat",
    "Manual Physics Playground",
    "📂 Simulation Gallery",
])


# ==============================================================================
# TAB 1: Conversational Human-in-the-Loop (HITL) Chat Mode
# ==============================================================================

with tab_chat:
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "type": "text",
                "content": (
                    "👋 **Welcome to DeepLenseSim AI Studio!**\n\n"
                    "I am your astrophysics simulation agent. You can ask me to simulate strong gravitational lensing "
                    "observations across **Model_I** to **Model_IV** with **CDM subhalos**, **Axion wave vortices**, or **smooth macro-models**.\n\n"
                    "Try asking something like:\n"
                    "- *\"Simulate 1 Euclid image with CDM subhalos at 150x150 resolution\"*\n"
                    "- *\"Generate an HST observation with axion wave perturbations\"*\n"
                    "- *\"Model IV starburst source galaxy with CDM perturbers\"*"
                ),
            }
        ]

    # Quick Suggestion Chips
    st.caption("Quick Simulation Presets:")
    q1, q2, q3, q4 = st.columns(4)
    chip_prompt = None
    if q1.button("🔭 Hubble + Axion", use_container_width=True, help="Simulate 1 HST image with Ultra-Light Axion wave vortices"):
        chip_prompt = "Simulate 1 HST image with Ultra-Light Axion wave vortices"
    if q2.button("🛰️ Euclid + CDM", use_container_width=True, help="Simulate 1 Euclid VIS image with CDM subhalos"):
        chip_prompt = "Simulate 1 Euclid VIS image with CDM subhalos"
    if q3.button("🌀 Starburst (Model IV)", use_container_width=True, help="Simulate 1 Model_IV complex starburst galaxy with source intensity 200 and no substructure"):
        chip_prompt = "Simulate 1 Model_IV complex starburst galaxy with source intensity 200 and no substructure"
    if q4.button("📊 Smooth Macro Control", use_container_width=True, help="Generate Model_I synthetic baseline control sample without substructure"):
        chip_prompt = "Generate Model_I synthetic baseline control sample without substructure"

    st.divider()

    # Render previous messages
    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            if msg.get("type") == "text":
                st.markdown(msg["content"])

            elif msg.get("type") == "clarification":
                st.markdown(msg["content"])
                st.info("💡 **Tip**: Reply with your preferred telescope (e.g. *Euclid*, *HST*, *Model_I*, *Model_IV*) and dark matter model (*CDM*, *Axion*, or *no_sub*).")

            elif msg.get("type") == "plan":
                plan = msg["plan"]
                st.markdown(msg.get("content", "### 📋 Proposed Execution Plan"))
                with st.container(border=True):
                    st.markdown(f"**Physical Summary**: {plan.get('summary_description', '')}")

                    p1, p2, p3, p4 = st.columns(4)
                    p1.markdown(f"**Telescope / Tier**: {plan.get('telescope_name')} (`{plan.get('model_configuration')}`)")
                    p2.markdown(f"**Substructure**: `{plan.get('substructure_type').upper()}`")
                    p3.markdown(f"**Resolution**: {plan.get('grid_resolution')} ({plan.get('field_of_view_arcsec')}\" FOV)")
                    p4.markdown(f"**Estimated Time**: ~{plan.get('estimated_runtime_seconds')}s for {plan.get('num_images')} image(s)")

                    with st.expander("🔍 View Detailed Physical Parameters (Full Configuration)"):
                        req_obj = msg["request"]
                        c_p1, c_p2 = st.columns(2)
                        with c_p1:
                            st.markdown("**Deflector Lens (SIE + Shear)**")
                            st.caption(
                                f"• Einstein Radius $\\theta_E$: `{req_obj.lens.theta_E}\"`\n"
                                f"• Ellipticity $(e_1, e_2)$: `({req_obj.lens.e1}, {req_obj.lens.e2})`\n"
                                f"• External Shear $(\\gamma_1, \\gamma_2)$: `({req_obj.lens.gamma1}, {req_obj.lens.gamma2})`\n"
                                f"• Center $(x, y)$: `({req_obj.lens.center_x}, {req_obj.lens.center_y})\"`"
                            )
                            st.markdown("**Cosmological Redshifts**")
                            st.caption(
                                f"• Lens Redshift $z_\\text{{lens}}$: `{req_obj.z_lens}`\n"
                                f"• Source Redshift $z_\\text{{source}}$: `{req_obj.z_source}`"
                            )
                        with c_p2:
                            st.markdown("**Source Galaxy (Sérsic Profile)**")
                            st.caption(
                                f"• Half-Light Radius $R_\\text{{sersic}}$: `{req_obj.source.R_sersic}\"`\n"
                                f"• Sérsic Index $n$: `{req_obj.source.n_sersic}`\n"
                                f"• Intensity Amplitude: `{req_obj.source.amp}`\n"
                                f"• Center $(x, y)$: `({req_obj.source.center_x}, {req_obj.source.center_y})\"`\n"
                                f"• Ellipticity $(e_1, e_2)$: `({req_obj.source.e1}, {req_obj.source.e2})`"
                            )
                            st.markdown("**Telescope & Observation Optics**")
                            st.caption(
                                f"• Pixel Scale: `{req_obj.observation.pixel_scale}\"/pix`\n"
                                f"• Grid Resolution: `{req_obj.observation.num_pix} x {req_obj.observation.num_pix}`\n"
                                f"• PSF FWHM: `{req_obj.observation.psf_fwhm}\"`\n"
                                f"• Exposure Time: `{req_obj.observation.exposure_time} s`\n"
                                f"• Background RMS: `{req_obj.observation.background_rms}`"
                            )
                        st.markdown("**Raw Ground-Truth Request Schema (JSON)**")
                        st.json(req_obj.model_dump())

                status = msg.get("status", "pending")
                if status == "pending":
                    btn_approve, btn_cancel = st.columns([1, 4])
                    with btn_approve:
                        if st.button("✅ Approve & Ray-Trace", key=f"approve_{idx}", type="primary"):
                            with st.spinner("Ray-tracing gravitational lens with lenstronomy..."):
                                pipeline = DeepLensePipeline(base_output_dir=OUTPUT_DIR)
                                results = pipeline.run(msg["request"])
                                msg["status"] = "approved"
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "type": "results",
                                    "results": results,
                                })
                                st.rerun()

                    with btn_cancel:
                        if st.button("❌ Cancel Plan", key=f"cancel_{idx}"):
                            msg["status"] = "cancelled"
                            st.session_state.messages.append({
                                "role": "assistant",
                                "type": "text",
                                "content": "Simulation plan was cancelled. What would you like to simulate instead?",
                            })
                            st.rerun()

                    st.caption("💬 *To adjust any physical or observational parameters, simply reply in chat below (e.g. 'need source intensity 200', 'change theta_e to 1.35', 'switch to axion').*")

                elif status == "approved":
                    st.caption("✅ *This plan was approved and executed.*")
                elif status == "cancelled":
                    st.caption("❌ *This plan was cancelled.*")
                elif status == "superseded":
                    st.caption("🔄 *This plan was updated with new modifications.*")

            elif msg.get("type") == "results":
                results = msg["results"]
                st.success(f"🎉 Ray-tracing complete! Successfully generated {len(results)} realization(s).")
                for r_idx, r in enumerate(results, 1):
                    st.markdown(f"#### Realization #{r_idx} — `{r.simulation_id}`")
                    render_realization_artifacts(r, unique_prefix=f"chat_{idx}_{r_idx}")
                    st.divider()

    # Chat Input & Reset Button Handler
    col_chat_input, col_chat_reset = st.columns([10, 1.4], vertical_alignment="bottom")
    with col_chat_input:
        user_input = st.chat_input("Enter your simulation request or reply to the agent...")
    with col_chat_reset:
        reset_clicked = st.button("🗑️ Reset Chat", use_container_width=True, help="Clear chat history and reset agent state")

    if reset_clicked:
        st.session_state.messages = [
            {
                "role": "assistant",
                "type": "text",
                "content": "Chat reset. What would you like to simulate next?",
            }
        ]
        st.session_state.pending_plan = None
        st.rerun()

    prompt_to_process = chip_prompt or user_input

    if prompt_to_process:
        # Check if previous assistant message was a clarification question
        is_clarif_reply = bool(st.session_state.messages and st.session_state.messages[-1].get("type") == "clarification")
        last_clarif_msg = st.session_state.messages[-1] if is_clarif_reply else None

        # Add user message
        st.session_state.messages.append({"role": "user", "type": "text", "content": prompt_to_process})
        effective_prompt = prompt_to_process
        prompt_lower = effective_prompt.lower()

        # Handle direct response to a clarification prompt
        if is_clarif_reply and last_clarif_msg:
            reply_lower = prompt_to_process.lower().strip()

            if reply_lower in ["no", "n", "cancel", "stop", "abort"]:
                st.session_state.messages.append({
                    "role": "assistant",
                    "type": "text",
                    "content": "Clarification cancelled. What would you like to simulate instead?",
                })
                st.rerun()

            # Check for unrecognized simulation parameters
            unrec = check_for_unrecognized_parameters(prompt_to_process)
            if unrec:
                st.session_state.messages.append({
                    "role": "assistant",
                    "type": "text",
                    "content": format_unrecognized_parameter_error(unrec),
                })
                st.rerun()

            # 1. Resolve Telescope Tier
            resolved_tier = None
            if any(k in reply_lower for k in ["starburst", "startburst", "complex", "model_iv", "model 4"]):
                resolved_tier = ModelConfiguration.MODEL_IV
            elif any(k in reply_lower for k in ["hst", "hubble", "model_iii", "model 3"]):
                resolved_tier = ModelConfiguration.MODEL_III
            elif any(k in reply_lower for k in ["euclid", "model_ii", "model 2"]):
                resolved_tier = ModelConfiguration.MODEL_II
            elif any(k in reply_lower for k in ["synthetic", "baseline", "model_i", "model 1"]):
                resolved_tier = ModelConfiguration.MODEL_I

            if resolved_tier is None and last_clarif_msg.get("known_tier"):
                try:
                    resolved_tier = ModelConfiguration(last_clarif_msg["known_tier"])
                except Exception:
                    resolved_tier = None

            # 2. Resolve Substructure
            resolved_sub = None
            if any(k in reply_lower for k in ["cdm", "subhalo", "cold dark matter"]):
                resolved_sub = SubstructureType.CDM
            elif any(k in reply_lower for k in ["axion", "vortex", "wave"]):
                resolved_sub = SubstructureType.AXION
            elif any(k in reply_lower for k in ["no_sub", "no sub", "without substructure", "smooth", "macro", "control"]):
                resolved_sub = SubstructureType.NO_SUB

            if resolved_sub is None and last_clarif_msg.get("known_sub"):
                try:
                    resolved_sub = SubstructureType(last_clarif_msg["known_sub"])
                except Exception:
                    resolved_sub = None

            # If both tier and substructure are now resolved, formulate the plan immediately!
            if resolved_tier and resolved_sub:
                base_req = create_default_request(model_cfg=resolved_tier, substructure_type=resolved_sub)
                init_p = last_clarif_msg.get("initial_prompt", "")
                combined_context = f"{init_p} {prompt_to_process}".strip()
                req, err = apply_plan_modifications(base_req, combined_context)
                if err and not err.startswith("💡"):
                    st.session_state.messages.append({
                        "role": "assistant",
                        "type": "text",
                        "content": err,
                    })
                    st.rerun()

                final_req = req or base_req
                validation = validate_simulation_plan(final_req)
                sub_label = {
                    SubstructureType.CDM: "Cold Dark Matter (CDM) subhalos",
                    SubstructureType.AXION: "Ultra-Light Axion wave-vortex defects",
                    SubstructureType.NO_SUB: "Smooth macro-model control sample",
                }.get(resolved_sub, str(resolved_sub.value))

                st.session_state.messages.append({
                    "role": "assistant",
                    "type": "plan",
                    "content": f"🔭 **Simulation Execution Plan Formulated** for **{final_req.observation.telescope_name}** with **{sub_label}**.\n\nPlease review and confirm below:",
                    "plan": validation["plan"],
                    "request": final_req,
                    "status": "pending",
                })
                st.rerun()

            elif resolved_sub and not resolved_tier:
                sub_label = {
                    SubstructureType.CDM: "Cold Dark Matter (CDM) subhalos",
                    SubstructureType.AXION: "Ultra-Light Axion wave-vortex perturbers",
                    SubstructureType.NO_SUB: "Smooth macro-model control sample",
                }.get(resolved_sub, str(resolved_sub.value))
                clarify_body = (
                    f"I've noted **{sub_label}**. Please choose your preferred telescope / instrument tier:\n\n"
                    "- **Model_I**: Synthetic baseline (0.05\"/pix, 64x64)\n"
                    "- **Model_II**: Euclid Space Telescope VIS band (0.10\"/pix, 150x150)\n"
                    "- **Model_III**: Hubble Space Telescope WFC3/ACS (0.04\"/pix, sharp 0.08\" PSF)\n"
                    "- **Model_IV**: Complex starburst source galaxy"
                )
                st.session_state.messages.append({
                    "role": "assistant",
                    "type": "clarification",
                    "content": clarify_body,
                    "known_tier": None,
                    "known_sub": resolved_sub.value,
                    "initial_prompt": f"{last_clarif_msg.get('initial_prompt', '')} {prompt_to_process}".strip(),
                })
                st.rerun()

            elif resolved_tier and not resolved_sub:
                clarify_body = (
                    f"I've configured optics for **{resolved_tier.value}**. Which dark matter substructure hypothesis would you like to simulate?\n\n"
                    "- **`cdm`**: Cold Dark Matter subhalos ($10^7-10^9 M_\\odot$)\n"
                    "- **`axion`**: Ultra-Light Axion wave-vortex interference defects\n"
                    "- **`no_sub`**: Smooth macro-model control sample"
                )
                st.session_state.messages.append({
                    "role": "assistant",
                    "type": "clarification",
                    "content": clarify_body,
                    "known_tier": resolved_tier.value,
                    "known_sub": None,
                    "initial_prompt": f"{last_clarif_msg.get('initial_prompt', '')} {prompt_to_process}".strip(),
                })
                st.rerun()

            else:
                st.session_state.messages.append({
                    "role": "assistant",
                    "type": "text",
                    "content": "⚠️ Please choose one of the available telescope tiers: **Model_I** (Baseline), **Model_II** (Euclid), **Model_III** (HST), or **Model_IV** (Starburst).",
                })
                st.rerun()

        # Check if user typed affirmative/negative response to a pending plan
        pending_plan_msg = next((m for m in reversed(st.session_state.messages) if m.get("type") == "plan" and m.get("status") == "pending"), None)

        if pending_plan_msg and prompt_lower.strip() in ["yes", "y", "proceed", "approve", "ok", "run"]:
            with st.spinner("Ray-tracing gravitational lens with lenstronomy..."):
                pipeline = DeepLensePipeline(base_output_dir=OUTPUT_DIR)
                results = pipeline.run(pending_plan_msg["request"])
                pending_plan_msg["status"] = "approved"
                st.session_state.messages.append({
                    "role": "assistant",
                    "type": "results",
                    "results": results,
                })
            st.rerun()

        elif pending_plan_msg and prompt_lower.strip() in ["no", "n", "cancel", "stop", "abort"]:
            pending_plan_msg["status"] = "cancelled"
            st.session_state.messages.append({
                "role": "assistant",
                "type": "text",
                "content": "Execution plan cancelled. What would you like to simulate instead?",
            })
            st.rerun()

        elif pending_plan_msg:
            # User wants to make changes to the pending plan via chat!
            with st.spinner("Applying modifications and re-validating execution plan with Groq..."):
                updated_req, error_msg = apply_plan_modifications(pending_plan_msg["request"], prompt_to_process)

                if error_msg:
                    # User specified an invalid parameter! Inform user with exact physical ranges
                    st.session_state.messages.append({
                        "role": "assistant",
                        "type": "text",
                        "content": error_msg,
                    })
                elif updated_req:
                    validation = validate_simulation_plan(updated_req)
                    diffs = diff_simulation_requests(pending_plan_msg["request"], updated_req)
                    diff_summary = "\n".join(diffs) if diffs else "• Parameters refreshed."
                    pending_plan_msg["status"] = "superseded"

                    st.session_state.messages.append({
                        "role": "assistant",
                        "type": "plan",
                        "content": f"✏️ **Execution Plan Updated!** Revised parameters:\n{diff_summary}\n\nPlease review and confirm below:",
                        "plan": validation["plan"],
                        "request": updated_req,
                        "status": "pending",
                    })
            st.rerun()

        else:
            # Check for parameter range queries
            is_range_query = any(q in prompt_lower for q in ["range of", "allowed range", "valid range", "parameter range", "allowed parameters", "limits of"])
            if is_range_query:
                param_spec = get_parameter_range_summary(prompt_to_process)
                st.session_state.messages.append({
                    "role": "assistant",
                    "type": "text",
                    "content": param_spec,
                })
                st.rerun()

            # If user asks a general physics/capability question (not a simulation run command)
            is_question = any(q in prompt_lower for q in ["what", "how", "explain", "help", "list", "specs", "capabilities"]) and not any(r in prompt_lower for r in ["simulate", "generate", "run", "produce", "want", "need", "starburst", "startburst", "model"])

            if is_question:
                with st.spinner("Consulting Groq LLM astrophysics engine..."):
                    try:
                        agent = create_deeplense_agent()
                        res = agent.run_sync(prompt_to_process)
                        assistant_response = res.output
                    except Exception as e:
                        assistant_response = f"Could not contact Groq LLM: {e}"
                st.session_state.messages.append({
                    "role": "assistant",
                    "type": "text",
                    "content": assistant_response,
                })
                st.rerun()

            else:
                # Use Groq LLM semantic parsing for the initial prompt
                with st.spinner("Formulating and physically validating simulation request with Groq..."):
                    req, error_msg, needs_clarify = parse_initial_simulation_request(effective_prompt)

                if needs_clarify:
                    p_lower = effective_prompt.lower()
                    already_sub = "cdm" if "cdm" in p_lower else ("axion" if "axion" in p_lower else ("no_sub" if any(s in p_lower for s in ["no_sub", "no sub", "smooth", "macro"]) else None))
                    already_tier = None
                    if any(k in p_lower for k in ["starburst", "startburst", "complex", "model_iv", "model 4"]):
                        already_tier = ModelConfiguration.MODEL_IV.value
                    elif any(k in p_lower for k in ["hst", "hubble", "model_iii", "model 3"]):
                        already_tier = ModelConfiguration.MODEL_III.value
                    elif any(k in p_lower for k in ["euclid", "model_ii", "model 2"]):
                        already_tier = ModelConfiguration.MODEL_II.value
                    elif any(k in p_lower for k in ["synthetic", "baseline", "model_i", "model 1"]):
                        already_tier = ModelConfiguration.MODEL_I.value

                    if already_sub and not already_tier:
                        sub_label = {
                            "cdm": "Cold Dark Matter (CDM) subhalos",
                            "axion": "Ultra-Light Axion wave-vortex perturbers",
                            "no_sub": "Smooth macro-model control sample",
                        }.get(already_sub, already_sub)
                        clarify_body = (
                            f"I'd be glad to simulate a gravitational lens with **{sub_label}** for you! "
                            "To configure the telescope optics accurately, please clarify:\n\n"
                            "**Which telescope / instrument tier?**\n"
                            "- **Model_I**: Synthetic baseline (0.05\"/pix, 64x64)\n"
                            "- **Model_II**: Euclid Space Telescope VIS band (0.10\"/pix, 150x150)\n"
                            "- **Model_III**: Hubble Space Telescope WFC3/ACS (0.04\"/pix, sharp 0.08\" PSF)\n"
                            "- **Model_IV**: Complex starburst source galaxy"
                        )
                    elif already_tier and not already_sub:
                        clarify_body = (
                            f"I'd be glad to configure the **{already_tier}** telescope optics for your gravitational lens simulation! "
                            "Please clarify:\n\n"
                            "**Which dark matter substructure hypothesis?**\n"
                            "- **`cdm`**: Cold Dark Matter subhalos ($10^7-10^9 M_\\odot$)\n"
                            "- **`axion`**: Ultra-Light Axion wave-vortex interference defects\n"
                            "- **`no_sub`**: Smooth macro-model control sample"
                        )
                    else:
                        clarify_body = (
                            "I'd be glad to simulate a gravitational lens for you! To configure the astrophysics and telescope optics accurately, please clarify:\n\n"
                            "1. **Which telescope / instrument tier?**\n"
                            "   - **Model_I**: Synthetic baseline (0.05\"/pix, 64x64)\n"
                            "   - **Model_II**: Euclid Space Telescope VIS band (0.10\"/pix, 150x150)\n"
                            "   - **Model_III**: Hubble Space Telescope WFC3/ACS (0.04\"/pix, sharp 0.08\" PSF)\n"
                            "   - **Model_IV**: Complex starburst source galaxy\n\n"
                            "2. **Which dark matter substructure hypothesis?**\n"
                            "   - **`cdm`**: Cold Dark Matter subhalos ($10^7-10^9 M_\\odot$)\n"
                            "   - **`axion`**: Ultra-Light Axion wave-vortex interference defects\n"
                            "   - **`no_sub`**: Smooth macro-model control sample"
                        )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "type": "clarification",
                        "content": clarify_body,
                        "known_tier": already_tier,
                        "known_sub": already_sub,
                        "initial_prompt": effective_prompt,
                    })
                    st.rerun()

                elif error_msg:
                    # User specified an out-of-bounds parameter! Return the exact allowed range
                    st.session_state.messages.append({
                        "role": "assistant",
                        "type": "text",
                        "content": error_msg,
                    })
                    st.rerun()

                elif req:
                    validation = validate_simulation_plan(req)
                    plan = validation["plan"]

                    st.session_state.messages.append({
                        "role": "assistant",
                        "type": "plan",
                        "content": "I have formulated and physically validated your simulation request. Please review and confirm the Execution Plan below:",
                        "plan": plan,
                        "request": req,
                        "status": "pending",
                    })
                    st.rerun()


# ==============================================================================
# TAB 2: Manual Physics Playground
# ==============================================================================

with tab_manual:
    st.markdown("### 🎛️ Granular Parameter Builder")
    st.markdown("Directly adjust physical parameters with real-time Pydantic validation before ray-tracing.")

    with st.form("manual_simulation_form"):
        col_m1, col_m2 = st.columns(2)

        with col_m1:
            st.markdown("#### 1. Setup & Substructure")
            m_tier = st.selectbox(
                "Model Configuration",
                options=[m.value for m in ModelConfiguration],
                index=1,
                help="Select instrument preset (Euclid, HST, Synthetic, Complex)",
            )
            m_sub = st.selectbox(
                "Dark Matter Substructure",
                options=[s.value for s in SubstructureType],
                index=1,
                format_func=lambda s: {
                    "no_sub": "no_sub (Smooth macro-model control)",
                    "cdm": "cdm (Cold Dark Matter subhalos)",
                    "axion": "axion (Ultra-Light Axion wave-vortex defects)",
                }.get(s, s),
            )
            m_num_images = st.slider("Number of Realizations", min_value=1, max_value=5, value=1)
            m_seed = st.number_input("Random Seed", value=42, step=1)

            st.markdown("#### 2. Deflector (Lens) Geometry")
            m_theta_E = st.slider("Einstein Radius θ_E (arcsec)", min_value=0.5, max_value=2.0, value=1.30, step=0.05)
            m_e1 = st.slider("Lens Ellipticity e1", min_value=-0.5, max_value=0.5, value=0.10, step=0.05)
            m_e2 = st.slider("Lens Ellipticity e2", min_value=-0.5, max_value=0.5, value=0.05, step=0.05)
            m_gamma1 = st.slider("External Shear γ1", min_value=-0.1, max_value=0.1, value=0.03, step=0.01)
            m_gamma2 = st.slider("External Shear γ2", min_value=-0.1, max_value=0.1, value=0.01, step=0.01)

        with col_m2:
            st.markdown("#### 3. Cosmological Redshifts")
            m_z_lens = st.number_input("Lens Redshift (z_lens)", value=0.50, min_value=0.05, max_value=5.0, step=0.05)
            m_z_source = st.number_input("Source Redshift (z_source)", value=1.80, min_value=0.10, max_value=10.0, step=0.05)

            st.markdown("#### 4. Background Source Galaxy (Sérsic)")
            m_amp = st.slider("Source Intensity (amp)", min_value=10.0, max_value=500.0, value=100.0, step=10.0)
            m_r_sersic = st.slider("Half-Light Radius R_sersic (arcsec)", min_value=0.05, max_value=1.0, value=0.25, step=0.05)
            m_n_sersic = st.slider("Sérsic Index n", min_value=0.5, max_value=4.0, value=1.5, step=0.25)
            m_src_x = st.slider("Source Center X (arcsec)", min_value=-0.5, max_value=0.5, value=0.05, step=0.01)
            m_src_y = st.slider("Source Center Y (arcsec)", min_value=-0.5, max_value=0.5, value=0.02, step=0.01)

            st.markdown("#### 5. Instrument Grid & Noise")
            col_pix1, col_pix2 = st.columns(2)
            with col_pix1:
                m_num_pix = st.selectbox("Grid Pixels", options=[64, 128, 150], index=2)
            with col_pix2:
                m_scale = st.number_input("Pixel Scale (arcsec/pix)", value=0.10, min_value=0.01, max_value=0.20, step=0.01)

        submit_manual = st.form_submit_button("🚀 Run Manual Simulation", type="primary")

    if submit_manual:
        try:
            req = SimulationRequest(
                model_configuration=ModelConfiguration(m_tier),
                substructure_type=SubstructureType(m_sub),
                num_images=m_num_images,
                z_lens=m_z_lens,
                z_source=m_z_source,
                random_seed=m_seed,
                lens=LensParameters(
                    theta_E=m_theta_E,
                    e1=m_e1,
                    e2=m_e2,
                    gamma1=m_gamma1,
                    gamma2=m_gamma2,
                ),
                source=SourceParameters(
                    amp=m_amp,
                    R_sersic=m_r_sersic,
                    n_sersic=m_n_sersic,
                    center_x=m_src_x,
                    center_y=m_src_y,
                ),
                substructure=SubstructureConfig(substructure_type=SubstructureType(m_sub)),
                observation=ObservationConfig(
                    telescope_name=MODEL_PRESETS[ModelConfiguration(m_tier)]["telescope_name"],
                    pixel_scale=m_scale,
                    num_pix=m_num_pix,
                    psf_fwhm=MODEL_PRESETS[ModelConfiguration(m_tier)]["psf_fwhm"],
                    exposure_time=MODEL_PRESETS[ModelConfiguration(m_tier)]["exposure_time"],
                    background_rms=MODEL_PRESETS[ModelConfiguration(m_tier)]["background_rms"],
                ),
            )

            with st.spinner("Ray-tracing simulation realization(s)..."):
                pipeline = DeepLensePipeline(base_output_dir=OUTPUT_DIR)
                results = pipeline.run(req)
                st.success(f"🎉 Generated {len(results)} realization(s)!")
                for r_idx, r in enumerate(results, 1):
                    st.markdown(f"#### Realization #{r_idx} — `{r.simulation_id}`")
                    render_realization_artifacts(r, unique_prefix=f"manual_{r_idx}")
                    st.divider()

        except Exception as e:
            st.error(format_validation_error_with_ranges(e))


# ==============================================================================
# TAB 3: Simulation Gallery & Inspection
# ==============================================================================

with tab_gallery:
    st.markdown("### 📂 Saved Simulation Runs")
    st.markdown(f"Inspect previously generated FITS, PNG, and metadata artifacts in `{OUTPUT_DIR}/`.")

    if not os.path.exists(OUTPUT_DIR):
        st.info(f"Directory `{OUTPUT_DIR}` does not exist yet. Run a simulation to generate artifacts!")
    else:
        run_dirs = sorted(
            [d for d in glob.glob(os.path.join(OUTPUT_DIR, "*")) if os.path.isdir(d)],
            reverse=True,
        )

        if not run_dirs:
            st.info("No saved simulation runs found yet.")
        else:
            selected_run = st.selectbox(
                "Select a Simulation Run Folder",
                options=run_dirs,
                format_func=lambda p: os.path.basename(p),
            )

            if selected_run:
                st.markdown(f"**Inspecting Run:** `{os.path.basename(selected_run)}`")

                png_files = sorted(glob.glob(os.path.join(selected_run, "*.png")))
                preview_pngs = [p for p in png_files if not p.endswith("_diagnostic.png")]
                diag_pngs = [p for p in png_files if p.endswith("_diagnostic.png")]
                json_files = sorted(glob.glob(os.path.join(selected_run, "*.json")))
                fits_files = sorted(glob.glob(os.path.join(selected_run, "*.fits")))

                c_gal1, c_gal2 = st.columns([1, 1.4])
                with c_gal1:
                    if preview_pngs:
                        st.markdown("**Observation Preview**")
                        st.image(preview_pngs[0], use_container_width=True)
                with c_gal2:
                    if diag_pngs:
                        st.markdown("**Diagnostic Plot**")
                        st.image(diag_pngs[0], use_container_width=True)

                if json_files:
                    with st.expander("📄 Ground-Truth JSON Metadata", expanded=False):
                        with open(json_files[0], "r") as f:
                            st.json(json.load(f))

                st.markdown("#### Files in Run:")
                for f_path in preview_pngs + diag_pngs + fits_files + json_files:
                    st.caption(f"• `{os.path.basename(f_path)}` ({round(os.path.getsize(f_path) / 1024, 1)} KB)")



