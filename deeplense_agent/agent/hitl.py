"""
Human-in-the-Loop (HITL) Interactive Controller
===============================================
Manages interactive dialogues between the human researcher and the Pydantic AI agent:
1. Clarification Phase: Identifies missing parameters and asks structured follow-up questions
2. Review & Approval Phase: Displays a formatted Execution Plan Preview for user confirmation
3. Execution Phase: Triggers the validated DeepLenseSim ray-tracing pipeline
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional
from deeplense_agent.utils.schemas import (
    ModelConfiguration,
    SubstructureType,
    SimulationRequest,
    SimulationMetadata,
    ExecutionPlanPreview,
    ClarificationQuestion,
    MODEL_PRESETS,
    create_default_request,
)
from deeplense_agent.core.pipeline import DeepLensePipeline


def clarify_simulation_parameters(missing_fields: List[str]) -> List[ClarificationQuestion]:
    """
    Formulates targeted follow-up questions to ask the human user when simulation parameters
    are missing, ambiguous, or need refinement (Human-in-the-Loop).

    Args:
        missing_fields: List of parameters that need clarification (e.g. ['substructure', 'model_tier'])
    """
    questions: List[ClarificationQuestion] = []
    
    field_handlers = {
        "substructure": ClarificationQuestion(
            parameter_name="substructure_type",
            question_text="Which dark matter substructure class would you like to simulate?",
            suggested_options=[
                "cdm (Cold Dark Matter subhalos ~ 10^7-10^9 M_sun)",
                "axion (Ultra-Light Axion wave-interference vortex defects)",
                "no_sub (Smooth macro-model control sample)",
            ],
            default_value="cdm",
        ),
        "model_tier": ClarificationQuestion(
            parameter_name="model_configuration",
            question_text="Which telescope / model tier specification should we model?",
            suggested_options=[
                "Model_II (Euclid Space Telescope VIS band, 0.10\"/pix)",
                "Model_III (Hubble Space Telescope WFC3/ACS high-res, 0.04\"/pix)",
                "Model_I (Synthetic Baseline, fast & idealized, 0.05\"/pix)",
                "Model_IV (Complex Multi-Component Source Galaxy)",
            ],
            default_value="Model_II",
        ),
        "resolution": ClarificationQuestion(
            parameter_name="num_pix",
            question_text="What image pixel resolution do you need?",
            suggested_options=["150x150 (Standard benchmark)", "64x64 (Fast test)"],
            default_value=150,
        ),
        "num_images": ClarificationQuestion(
            parameter_name="num_images",
            question_text="How many randomized simulation realizations should be generated?",
            suggested_options=["1 (Single sample)", "3 (Small batch)", "10 (Training mini-batch)"],
            default_value=1,
        ),
    }

    for f in missing_fields:
        f_norm = f.lower().strip()
        if f_norm in field_handlers:
            questions.append(field_handlers[f_norm])
        else:
            questions.append(
                ClarificationQuestion(
                    parameter_name=f,
                    question_text=f"Please specify your preferred setting for '{f}':",
                    suggested_options=[],
                    default_value=None,
                )
            )

    return questions


class HITLController:
    """Coordinates Human-in-the-Loop interactive clarification and approval."""

    @staticmethod
    def clarify_missing_parameters(user_prompt: str) -> Dict[str, Any]:
        """
        Detects missing parameters from the prompt and asks clarifying follow-up questions.
        """
        prompt_lower = user_prompt.lower()
        missing: List[str] = []

        # Check for substructure
        if not any(s in prompt_lower for s in ["cdm", "axion", "no_sub", "subhalo", "vortex", "control", "smooth"]):
            missing.append("substructure")

        # Check for model tier
        if not any(m in prompt_lower for m in ["model_i", "model_ii", "model_iii", "model_iv", "euclid", "hst", "hubble", "complex", "synthetic"]):
            missing.append("model_tier")

        if not missing:
            return {}

        questions = clarify_simulation_parameters(missing)
        clarified: Dict[str, Any] = {}

        print("\n" + "=" * 70)
        print("   Parameter Clarification Needed")
        print("=" * 70)
        print("The agent needs a few details to configure the simulation accurately:\n")

        for q in questions:
            print(f"• {q.question_text}")
            if q.suggested_options:
                for idx, opt in enumerate(q.suggested_options, 1):
                    print(f"    [{idx}] {opt}")
                choice = input(f"  Your choice [1-{len(q.suggested_options)}, default: 1]: ").strip()
                if choice.isdigit() and 1 <= int(choice) <= len(q.suggested_options):
                    val = q.suggested_options[int(choice) - 1].split()[0]
                else:
                    val = q.default_value
            else:
                user_val = input(f"  Value [default: {q.default_value}]: ").strip()
                val = user_val if user_val else q.default_value

            clarified[q.parameter_name] = val
            print()

        return clarified

    @staticmethod
    def request_execution_approval(plan: Dict[str, Any]) -> bool:
        """
        Presents the Execution Plan Preview to the user and awaits confirmation before ray-tracing.
        """
        print("\n" + "=" * 70)
        print("  Execution Plan Review")
        print("=" * 70)
        for k, v in plan.items():
            if k != "ready_for_execution":
                label = k.replace("_", " ").title()
                print(f"  • {label:24s}: {v}")
        print("=" * 70)
        approval = input("Proceed with numerical simulation? [Y/n]: ").strip().lower()
        return approval in ["", "y", "yes"]

    @classmethod
    def execute_with_hitl(
        cls,
        user_prompt: str,
        auto_approve: bool = False,
        output_dir: str = "sim_output",
    ):
        """
        Full end-to-end Human-in-the-Loop workflow:
        1. Analyzes prompt & clarifies missing fields
        2. Validates plan
        3. Requests human confirmation
        4. Runs simulation
        """
        print(f"\n[AGENT] Analyzing prompt: \"{user_prompt}\"")
        clarifications = cls.clarify_missing_parameters(user_prompt)

        # Merge extracted intent with clarifications
        prompt_lower = (user_prompt + " " + " ".join(str(v) for v in clarifications.values())).lower()

        tier = ModelConfiguration.MODEL_I
        if "model_iv" in prompt_lower or "complex" in prompt_lower:
            tier = ModelConfiguration.MODEL_IV
        elif "model_iii" in prompt_lower or "hst" in prompt_lower or "hubble" in prompt_lower:
            tier = ModelConfiguration.MODEL_III
        elif "model_ii" in prompt_lower or "euclid" in prompt_lower:
            tier = ModelConfiguration.MODEL_II

        sub_type = SubstructureType.NO_SUB
        if "cdm" in prompt_lower or "subhalo" in prompt_lower:
            sub_type = SubstructureType.CDM
        elif "axion" in prompt_lower or "vortex" in prompt_lower or "wave" in prompt_lower:
            sub_type = SubstructureType.AXION

        req = create_default_request(
            model_cfg=tier,
            substructure_type=sub_type,
            num_images=1,
            num_pix=150,
        )

        # Validate and build preview
        fov = req.observation.num_pix * req.observation.pixel_scale
        preview = ExecutionPlanPreview(
            model_configuration=req.model_configuration,
            substructure_type=req.substructure.substructure_type,
            telescope_name=req.observation.telescope_name,
            num_images=req.num_images,
            grid_resolution=f"{req.observation.num_pix} x {req.observation.num_pix}",
            field_of_view_arcsec=round(fov, 3),
            redshifts=f"z_lens={req.z_lens}, z_source={req.z_source}",
            estimated_runtime_seconds=round(0.4 * req.num_images, 2),
            summary_description=(
                f"Simulating {req.num_images} realization(s) of {req.substructure.substructure_type.value.upper()} "
                f"substructure under {req.model_configuration.value} ({req.observation.telescope_name}) "
                f"at {fov:.2f}\" FOV with theta_E={req.lens.theta_E}\"."
            ),
            ready_for_execution=True,
        )
        plan = preview.model_dump()

        if not auto_approve:
            approved = cls.request_execution_approval(plan)
            if not approved:
                print("\n[AGENT] Simulation cancelled by user.")
                return None

        print("\n[AGENT] Executing DeepLenseSim ray-tracing pipeline...")
        pipeline = DeepLensePipeline(base_output_dir=output_dir)
        results = pipeline.run(req)
        print(f"[AGENT] Successfully generated {len(results)} simulation image(s)!")
        for idx, r in enumerate(results, 1):
            print(f"\n--- Realization #{idx} ---")
            print(f"  • Simulation ID: {r.simulation_id}")
            print(f"  • Magnification: {r.observables.estimated_magnification}x")
            print(f"  • Total Flux: {r.observables.total_flux}")
            print(f"  • FITS Image: {r.image_paths['fits']}")
            print(f"  • PNG Preview: {r.image_paths['png']}")
            print(f"  • Diagnostic Plot: {r.image_paths['diagnostic']}")
            print(f"  • Metadata JSON: {r.simulation_id}_metadata.json")

        return results

