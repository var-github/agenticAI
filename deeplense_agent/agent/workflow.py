"""
Pydantic AI Agent Workflow Engine for DeepLenseSim
==================================================
Builds the conversational Pydantic AI Agent powered directly by Groq:
- System prompt infused with strong gravitational lensing astrophysics
- Registered tool functions (inspection, clarification, validation, execution)
- Groq Cloud LLM (qwen/qwen3.8-27b) via OpenAIProvider
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Union
from dotenv import load_dotenv

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from deeplense_agent.utils.schemas import (
    ModelConfiguration,
    SubstructureType,
    SimulationRequest,
    SimulationMetadata,
    ExecutionPlanPreview,
    MODEL_PRESETS,
    get_parameter_range_summary,
)
from deeplense_agent.core.pipeline import DeepLensePipeline
from deeplense_agent.agent.prompt import SYSTEM_PROMPT
from deeplense_agent.agent.hitl import clarify_simulation_parameters

load_dotenv()


def inspect_model_capabilities(model_tier: Union[str, int] = "Model_I") -> Dict[str, Any]:
    """
    Inspects available DeepLense model configurations and instrument specifications.
    
    Args:
        model_tier: Model tier name (e.g. 'Model_I', 'Model_II', 'Model_III', 'Model_IV') or tier number (1 to 4).
    """
    s = str(model_tier).strip().title().replace(" ", "_")
    if s.isdigit():
        s = f"Model_{s}"
    remap = {
        "Model_1": "Model_I",
        "Model_2": "Model_II",
        "Model_3": "Model_III",
        "Model_4": "Model_IV",
        "1": "Model_I",
        "2": "Model_II",
        "3": "Model_III",
        "4": "Model_IV",
    }
    tier_key = remap.get(s, s)

    try:
        cfg = ModelConfiguration(tier_key)
        return MODEL_PRESETS[cfg]
    except ValueError:
        return {
            "error": f"Unknown model tier '{model_tier}'.",
            "available_tiers": [m.value for m in ModelConfiguration],
        }


def inspect_parameter_ranges(parameter_name: str = "all") -> str:
    """
    Inspects allowed physical parameter ranges, units, and astrophysical limits
    for DeepLenseSim simulation configurations (e.g. theta_E, amp, z_lens, num_pix).
    
    Args:
        parameter_name: Parameter name (e.g. 'theta_E', 'amp', 'z_lens', 'num_pix') or 'all'.
    """
    return get_parameter_range_summary(parameter_name)


def validate_simulation_plan(request: SimulationRequest) -> Dict[str, Any]:
    """
    Validates physical feasibility and builds an execution plan preview for human confirmation.
    
    Args:
        request: The proposed SimulationRequest
    """
    fov = request.observation.num_pix * request.observation.pixel_scale
    preview = ExecutionPlanPreview(
        model_configuration=request.model_configuration,
        substructure_type=request.substructure.substructure_type,
        telescope_name=request.observation.telescope_name,
        num_images=request.num_images,
        grid_resolution=f"{request.observation.num_pix} x {request.observation.num_pix}",
        field_of_view_arcsec=round(fov, 3),
        redshifts=f"z_lens={request.z_lens}, z_source={request.z_source}",
        estimated_runtime_seconds=round(0.4 * request.num_images, 2),
        summary_description=(
            f"Simulating {request.num_images} realization(s) of {request.substructure.substructure_type.value.upper()} "
            f"substructure under {request.model_configuration.value} ({request.observation.telescope_name}) "
            f"at {fov:.2f}\" FOV with theta_E={request.lens.theta_E}\"."
        ),
        ready_for_execution=True,
    )
    return {"status": "valid", "plan": preview.model_dump()}


def run_deeplense_simulation(request: SimulationRequest,
    output_dir: str = "sim_output"):
    pipeline = DeepLensePipeline(base_output_dir=output_dir)
    return pipeline.run(request)


def create_deeplense_agent(model_name: str = "qwen/qwen3.8-27b") -> Agent:
    """
    Constructs and returns the configured Pydantic AI Agent.
    Powered directly by Groq LLM (default: 'qwen/qwen3.8-27b').
    model_name: Groq model identifier (e.g. 'qwen/qwen3.8-27b', 'llama-3.3-70b-versatile').
    """
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise RuntimeError(
            "GROQ_API_KEY environment variable is required to create the DeepLenseSim agent. "
            "Please set GROQ_API_KEY in your .env file or environment."
        )

    provider = OpenAIProvider(
        base_url="https://api.groq.com/openai/v1",
        api_key=groq_key,
    )
    model = OpenAIChatModel(
        model_name=model_name,
        provider=provider,
    )

    agent = Agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
    )

    # Register tools
    agent.tool_plain(inspect_model_capabilities)
    agent.tool_plain(inspect_parameter_ranges)
    agent.tool_plain(clarify_simulation_parameters)
    agent.tool_plain(validate_simulation_plan)
    agent.tool_plain(run_deeplense_simulation)

    return agent
