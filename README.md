# DeepLenseSim Pydantic AI Agentic Workflow

An intelligent agentic workflow powered by **[Pydantic AI](https://ai.pydantic.dev/)** and **[lenstronomy](https://github.com/lenstronomy/lenstronomy)** that wraps the **[DeepLenseSim](https://github.com/mwt5345/DeepLenseSim)** simulation pipeline. It enables researchers to simulate strong gravitational lensing images through natural language interaction, strict Pydantic parameter validation, and Human-in-the-Loop (HITL) clarification.

---

## Architecture Overview

```
                                [ Natural Language User Prompt ]
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │     Pydantic AI Agent (Groq)     │
                              │  • Physics-infused system prompt │
                              │  • Semantic parameter extraction │
                              │  • Groq LLM (qwen/qwen3.8-27b)   │
                              └────────────────┬─────────────────┘
                                               │
                      ┌────────────────────────┼────────────────────────┐
                      ▼                        ▼                        ▼
           [ Tool: clarify_parameters ] [ Tool: validate_plan ]  [ Tool: run_simulation ]
           (Targeted HITL questions)    (Physical boundaries)    (lenstronomy ray-tracer)
                      │                        │                        │
                      ▼                        ▼                        ▼
           [ Human Clarification ]   [ Execution Plan Preview ]  ┌───────────────────────┐
                      │                        │                 │  DeepLenseSim Engine  │
                      └────────────────────────┼────────────────►│  • Macro-lens (SIE)   │
                                               │                 │  • Substructure       │
                                               ▼                 │  • PSF & Noise        │
                                      [ Human Approval Gate ]    └───────────┬───────────┘
                                                                             │
                                                                             ▼
                                                                [ Structured Artifacts ]
                                                                • FITS (astronomy data)
                                                                • NPY (ML arrays)
                                                                • PNG (preview & ΔI map)
                                                                • JSON (ground truth)
```

---

https://github.com/user-attachments/assets/3d457099-264f-4eec-a425-b6fc58bf8074

Hosted at [https://deep-lense-agent.streamlit.app/](https://deep-lense-agent.streamlit.app/)

---

## Supported Model Configurations

The agent supports all 4 benchmark model tiers from DeepLenseSim:

| Model Tier | Simulated Instrument / Setup | Pixel Scale | PSF FWHM | Resolution | Astrophysical Characteristics |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Model_I** | Synthetic Baseline | $0.05''/\text{pix}$ | $0.10''$ | $64\times 64$ / $150\times 150$ | Idealized baseline with controlled Gaussian background noise; representation learning benchmark. |
| **Model_II** | Euclid Space Telescope (VIS Band) | $0.10''/\text{pix}$ | $0.16''$ | $150\times 150$ | Wide-field space survey observation with Euclid VIS optics and 1600s survey noise. |
| **Model_III** | Hubble Space Telescope (HST WFC3/ACS)| $0.04''/\text{pix}$ | $0.08''$ | $150\times 150$ | High angular resolution space observation with sharp diffraction PSF and deep integration. |
| **Model_IV** | Complex Source Morphology | $0.05''/\text{pix}$ | $0.09''$ | $150\times 150$ | Multi-component source galaxy with clumpy star-forming knots testing domain adaptation. |

---

## Dark Matter Substructure Physics

The agent models the three primary physical hypotheses studied in the DeepLense project:

1. **`no_sub` (Control Sample)**:
   * Smooth Singular Isothermal Ellipsoid (`SIE` / `EPL`) macro-model + External Shear (`SHEAR`).
   * No dark matter perturbations.
2. **`cdm` (Cold Dark Matter)**:
   * Population of discrete subhalos drawn from a power-law subhalo mass function ($\mathrm{d}N/\mathrm{d}M \propto M^{-1.9}$) in the range $[10^7, 10^9] M_\odot$.
   * Clustered near the Einstein radius ($\theta_E$) where lensing magnification is maximal.
3. **`axion` (Ultra-Light Axion / Fuzzy Dark Matter)**:
   * Wave interference vortex defects and localized potential fluctuations perturbing the critical curve.
   * Parameterized by vortex core radius $\sigma_{\text{vortex}}$ ($0.15''$) and perturbation amplitude.

---

## Human-in-the-Loop (HITL) Workflow

A common failure mode in scientific automation is silently guessing missing parameters or running expensive numerical simulations on malformed inputs. Our agent enforces a **two-tier HITL protocol**:

1. **Phase 1: Ambiguity Detection & Stateful Multi-Turn Clarification**:
   * **Ambiguity Detection**: If a user prompt lacks essential simulation parameters (e.g. specifying `cdm` substructure without a telescope tier, or specifying `euclid` optics without a dark matter hypothesis), the agent triggers targeted clarification questions detailing available instruments (`Model_I` to `Model_IV`) and dark matter hypotheses (`cdm`, `axion`, `no_sub`).
   * **Stateful Context Preservation**: The agent maintains conversational state across turns (`known_tier`, `known_sub`, `initial_prompt`). When the user replies with the missing piece (e.g. simply typing `euclid` or `cdm`), the state machine automatically merges the follow-up with prior specifications, applies any parameter modifications, and advances directly to the execution plan without redundant looping.
2. **Phase 2: Execution Plan Review & Human Gate**:
   * The agent constructs a validated `SimulationRequest` and displays an `ExecutionPlanPreview` (showing deflector mass, source profile, instrument resolution, and estimated runtime).
   * It awaits explicit user confirmation (`✅ Approve & Ray-Trace` button or affirmative chat like `"yes"`, `"proceed"`, `"run"`) before launching numerical ray-tracing with `lenstronomy`.

---

## Registered Agent Tools

The Pydantic AI agent has five domain-specific tools registered via `@agent.tool_plain`:

| Tool Function | Purpose | Inputs & Operation |
| :--- | :--- | :--- |
| **`inspect_model_capabilities`** | Inspects instrument and observation benchmark specs for `Model_I` through `Model_IV`. | Takes `model_tier` (`'Model_I'`, `'Model_II'`, `'Model_III'`, `'Model_IV'` or 1–4). Returns pixel scale, PSF FWHM, survey exposure time, default redshifts, and detector grid resolution. |
| **`inspect_parameter_ranges`** | Inspects valid physical limits, units, and boundary constraints across all 23 simulation parameters. | Takes `parameter_name` (e.g. `'theta_E'`, `'amp'`, `'z_lens'`, `'num_pix'`) or `'all'`. Returns exact allowed numerical bounds, units, and astrophysical descriptions. |
| **`clarify_simulation_parameters`** | Generates targeted multiple-choice follow-up questions when a user prompt lacks essential parameters. | Invoked during Phase 1 HITL when requests are underspecified; provides structured options with recommended scientific defaults. |
| **`validate_simulation_plan`** | Verifies physical feasibility and builds an execution plan preview before any ray-tracing occurs. | Validates cross-parameter constraints (FOV geometric containment, redshift ordering, grid parity). Returns an `ExecutionPlanPreview` with estimated runtime and FOV arcseconds. |
| **`run_deeplense_simulation`** | Executes the full numerical ray-tracing pipeline via `lenstronomy`. | Runs deflector ray deflection, perturber injection, PSF convolution, and noise synthesis. Generates FITS, NPY, PNG, and JSON metadata artifacts in `sim_output/`. |

---

## Multi-Parameter Handling & Validation Pipeline

Strong gravitational lensing simulations require coordinating over 20 interdependent astrophysical, geometrical, and optical parameters. The agent handles multi-parameter complexity through a four-stage pipeline:

### 1. Simultaneous Natural Language Extraction
Users can specify any combination of parameters in a single conversational prompt:
```text
"Simulate Model_IV starburst with no substructure, external shear y2 = 0.05,
 half-light radius 0.35, theta_E = 1.35, source intensity 250, z_lens = 0.4 and z_source = 2.1"
```
* **Parallel Semantic Mapping**: The Groq LLM parses multiple parameters simultaneously into the strongly-typed `SimulationRequest` JSON schema, resolving aliases and shorthand (e.g. `y2` $\rightarrow$ `lens.gamma2`, `half light radius` $\rightarrow$ `source.R_sersic`, `intensity` $\rightarrow$ `source.amp`, `center` $\rightarrow$ `lens.center_x, lens.center_y`).
* **Instrument Preset Inheritance**: Parameters omitted by the user are automatically backfilled with standardized, physically consistent benchmarks for the selected instrument tier (`Model_I` to `Model_IV`).

### 2. Conversational Multi-Parameter Plan Revisions
When a simulation plan is pending user approval, researchers can adjust multiple parameters simultaneously via chat:
```text
"Set theta_E to 1.4, source intensity to 300, and center deflector to (0.05, -0.02)"
```
* **Differential Tracking (`diff_simulation_requests`)**: Computes an exact attribute-level diff between the active plan and proposed updates, displaying changes clearly in the execution card:
  * `• Lens theta_E: 1.25" → 1.40"`
  * `• Source Intensity (amp): 100.0 → 300.0`
  * `• Lens Center (x, y): (0.0, 0.0)" → (0.05, -0.02)"`
* **Non-Destructive State Updates**: Unmodified parameters retain their values from the previous plan without unintended resets.

### 3. Cross-Parameter Physical Constraints
DeepLenseSim validates relational physical dependencies between parameters using Pydantic v2 model validators:
* **Redshift Causality**: $z_{\text{lens}} < z_{\text{source}}$ strictly enforced; deflector galaxies must be closer to the observer than background sources.
* **Geometrical Field-of-View Containment**: $\theta_E < \frac{\mathrm{FOV}}{2}$ (enforcing `theta_E < (num_pix * pixel_scale) / 2`); ensures the Einstein ring fits comfortably within the camera sensor field of view.
* **Mass Function Bounds**: Subhalo mass limits must satisfy $M_{\min} < M_{\max}$ (enforcing `min_mass_solar < max_mass_solar` in $[10^7, 10^9]\,M_\odot$).
* **FFT Grid Parity**: `num_pix` must be an even integer (e.g. 64, 128, 150) for fast Fourier transform PSF convolutions.
* **Informative Physical Range Feedback**: When any parameter violates physical limits, `format_validation_error_with_ranges` catches the error and reports the exact allowed range, units, and astrophysical explanation.

### 4. Unrecognized Parameter Guard
If a prompt includes parameters not modeled in DeepLenseSim ray-tracing (e.g. `temperature`, `metallicity`, `spin`, `dark energy w`, `color`):
* Any property not found in the recognized parameter set is detected upfront.
* The system informs the user that the parameter is not an input to the simulator, presents the complete list of valid modifiable parameters, and preserves the pending plan without corrupting state.

---

## Project Structure

```text
agenticAI/
├── deeplense_agent/
│   ├── core/
│   │   ├── physics.py          # lenstronomy ray-tracing, PSF convolution, observables
│   │   ├── substructure.py     # CDM subhalo population & Axion vortex engine
│   │   └── pipeline.py         # DeepLenseSim batch orchestration pipeline
│   ├── agent/
│   │   ├── prompt.py           # Astrophysical system prompt & domain instructions
│   │   ├── workflow.py         # Pydantic AI agent instance & registered tool functions
│   │   └── hitl.py             # Human-in-the-loop interactive controller & clarification
│   └── utils/
│       ├── schemas.py          # Strongly-typed Pydantic v2 schemas & Model_I-IV presets
│       └── storage.py          # FITS, NPY, PNG, and JSON metadata exporter
├── app.py                      # DeepLenseSim Streamlit AI Studio Front-End
└── README.md                   # Project documentation
```

---

## DeepLenseSim AI Studio Web Dashboard

The entire interactive application runs through Streamlit:

```bash
streamlit run app.py
```

The web dashboard provides:
* **🤖 Conversational Chat Agent**: Enter natural language simulation requests, modify physical parameters directly via chat, review the execution plan, and approve ray-tracing.
* **🎛️ Manual Physics Playground**: Adjust deflector geometry, source light profiles, redshifts, and dark matter substructures with real-time Pydantic validation.
* **📂 Simulation Gallery**: Inspect, visualize, and download generated FITS images, raw NPY arrays, publication-quality PNG previews, 4-panel diagnostic plots, and ground-truth JSON metadata.
* **📖 Parameter Reference**: A sticky reference modal dialog displaying allowed physical ranges, units, and astrophysical descriptions for all 23 simulation parameters.

---

## Output Artifacts & Metadata

For every simulation realization, the pipeline generates:
* **`.fits`**: Standard astronomy data file with full FITS header keywords (`TELESCOP`, `Z_LENS`, `Z_SRC`, `THETA_E`, `SUB_TYPE`, `MAGNIF`).
* **`.npy`**: Raw NumPy 2D array ready for machine learning pipelines.
* **`.png`**: High-contrast, publication-quality image preview with arcsecond coordinates.
* **`_diagnostic.png`**: 4-panel diagnostic figure showing:
  1. Observed Mock Image (with optical blur & noise)
  2. Clean Ray-Traced Light
  3. Surface Mass Density (Convergence $\kappa$)
  4. Substructure Residual Map ($\Delta I = I_{\text{sub}} - I_{\text{smooth}}$)
* **`_metadata.json`**: Ground-truth JSON metadata capturing physical parameters, subhalo catalog, and observables.

