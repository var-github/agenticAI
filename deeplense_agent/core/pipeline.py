"""
DeepLenseSim Orchestration Pipeline
===================================
Manages the batch execution of strong gravitational lensing simulations:
- Runs single or multi-image realizations
- Generates reproducible random seeds per image
- Coordinates ray-tracing, observables computation, and artifact persistence
- Returns structured SimulationMetadata for downstream ML benchmarks
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Any, Dict, List
from deeplense_agent.utils.schemas import SimulationRequest, SimulationMetadata
from deeplense_agent.core.physics import simulate_lensing_realization
from deeplense_agent.utils.storage import export_simulation_artifacts


class DeepLensePipeline:
    """Orchestration pipeline for executing DeepLense strong lensing simulations."""

    def __init__(self, base_output_dir: str = "sim_output"):
        self.base_output_dir = os.path.abspath(base_output_dir)
        os.makedirs(self.base_output_dir, exist_ok=True)

    def run(self, request: SimulationRequest) -> List[SimulationMetadata]:
        """
        Executes the simulation request and generates all artifacts.
        
        Args:
            request: Validated SimulationRequest specifying all physics and instrument options.

        Returns:
            List of SimulationMetadata records (one per generated image realization).
        """
        timestamp_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        sub_type = request.substructure.substructure_type.value
        model_str = request.model_configuration.value
        run_folder = f"{model_str}_{sub_type}_{timestamp_str}"
        output_dir = os.path.join(self.base_output_dir, run_folder)
        os.makedirs(output_dir, exist_ok=True)

        results: List[SimulationMetadata] = []
        base_seed = request.random_seed if request.random_seed is not None else int(time.time())

        for idx in range(request.num_images):
            sim_id = f"{model_str}_{sub_type}_{timestamp_str}_{idx + 1:03d}"
            current_seed = base_seed + (idx * 1000)

            # 1. Run physical ray-tracing simulation
            sim_output = simulate_lensing_realization(request=request, seed=current_seed)

            # 2. Export FITS, NPY, PNG, and JSON metadata
            metadata = export_simulation_artifacts(
                simulation_id=sim_id,
                output_dir=output_dir,
                request=request,
                sim_results=sim_output,
            )
            results.append(metadata)

        return results

