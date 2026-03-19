"""
AutoML Engine for MetaOpticsAI — Automated Metalens Design
===========================================================

Implements a Self-RAG + LangGraph pipeline for automating metalens design:

  ┌──────────────────────────────────────────────────────────────────┐
  │  Natural Language Requirements                                   │
  │         ↓                                                        │
  │  [parse_requirements] → [rag_retrieve] → [grade_documents]      │
  │         ↓ (relevant)           ↓ (irrelevant → re-query)        │
  │  [generate_params] → [simulate_rcwa] → [evaluate_fom]           │
  │         ↓                                                        │
  │  [self_reflect] → [optimize_params] → loop / DONE               │
  └──────────────────────────────────────────────────────────────────┘

Compatible with Python 3.11.9+ | Ollama: deepseek-r1:7b | LangGraph 0.2+
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback
import dataclasses
from dataclasses import field
from typing import Any, Callable, Literal, Optional

import numpy as np

# ── LangChain / LangGraph ─────────────────────────────────────────────────────
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MetaOpticsAI.AutoML")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

OLLAMA_MODEL       = "deepseek-r1:7b"    # Best for physics / math reasoning
OLLAMA_EMBED_MODEL = "nomic-embed-text"  # Lightweight embedding model
OLLAMA_BASE_URL    = "http://localhost:11434"

MAX_RETRIEVAL_RETRIES  = 3
MAX_OPTIMIZE_ITERS     = 10
FOM_TARGET_THRESHOLD   = 0.85           # Strehl ratio target
DOCUMENT_GRADE_CUTOFF  = 0.6            # Relevance score threshold


# ─────────────────────────────────────────────────────────────────────────────
# KNOWLEDGE BASE  (embedded metalens / photonics design corpus)
# ─────────────────────────────────────────────────────────────────────────────

METALENS_KNOWLEDGE_CORPUS: list[str] = [
    # ── Design Principles ──────────────────────────────────────────────────
    """
    Metalens Design Principles:
    A metalens is a flat optical element composed of sub-wavelength nanostructures
    (meta-atoms) arranged on a surface to manipulate the phase, amplitude, or
    polarization of incident light. The key design equation is the hyperbolic phase
    profile: φ(r) = -(2π/λ) * (√(r² + f²) - f), where r is the radial distance,
    λ is the design wavelength, and f is the focal length. For broadband operation,
    the phase profile must be achromatic, requiring dispersion-engineered meta-atoms.
    Numerical Aperture (NA) = sin(arctan(D/2f)), where D is lens diameter.
    """,
    """
    Meta-atom (Unit Cell) Selection:
    Cylindrical pillars (TiO2, Si, GaN) on a substrate (SiO2) are the most common.
    The pillar diameter controls the phase shift (0 to 2π) at a given wavelength.
    Key parameters: pillar diameter D (50–400 nm), height H (300–900 nm),
    periodicity P (300–600 nm). For full 2π phase coverage: pillar height should
    be ≥ λ/2n (where n is refractive index). TiO2 works well for visible
    wavelengths (400–700 nm). Silicon is preferred for near-infrared (900–1600 nm).
    GaN is suitable for UV and visible. The aspect ratio H/D should remain below 15.
    """,
    """
    RCWA (Rigorous Coupled-Wave Analysis) Simulation:
    RCWA is the gold-standard method for simulating periodic nanostructures.
    Key parameters: number of Fourier harmonics (xy_harmonics), spatial resolution
    (resolution), wavelength range, angle of incidence (theta, phi).
    Transmission efficiency (t_power) is the primary figure of merit.
    The complex transmission coefficient t encodes both amplitude and phase.
    For a single-wavelength metalens, optimize t_amplitude ≈ 1 and t_phase = target.
    For multi-wavelength: use multiple RCWA simulations at each wavelength.
    """,
    """
    Phase Profile Optimization Strategy:
    1. Start with geometric phase (Pancharatnam-Berry) for broadband designs.
    2. Use propagation phase for single-wavelength high-efficiency designs.
    3. Target phase error < 0.1 rad RMS across the aperture.
    4. Discretize with ≥8 phase levels for adequate approximation.
    5. Use circular symmetry (radial expansion) to reduce computation.
    6. The Strehl ratio metric captures both phase accuracy and transmission efficiency.
    Recommended FOM: Strehl Ratio > 0.8 indicates a high-quality metalens design.
    """,
    """
    Wavelength-Dependent Design Considerations:
    Visible (450–700 nm): Use TiO2 pillars, period 300–400 nm, height 400–600 nm.
    Near-IR (900–1600 nm): Use Si or a-Si pillars, period 600–900 nm, height 500–800 nm.
    Mid-IR (2–5 μm): Use Ge or chalcogenide glass, period 1–3 μm.
    The phase dispersion (dφ/dλ) must be controlled for achromatic designs.
    For achromatic metalens: use paired resonances with opposite dispersions.
    """,
    """
    Focal Length and NA Trade-offs:
    Short focal length (high NA): requires high phase gradient near the edge.
    This demands small periodicity and tight fabrication tolerances.
    NA > 0.8: extremely challenging, requires sub-100 nm features.
    NA 0.1–0.5: achievable with standard e-beam lithography.
    For a 1 mm diameter lens at 532 nm, f = 1 mm gives NA ≈ 0.45.
    The Abbe diffraction limit: resolution = 0.61λ/NA.
    Depth of focus (DOF) = λ/(2·NA²).
    """,
    """
    Optimization Algorithm Recommendations:
    1. Adam optimizer: best for differentiable RCWA with TensorFlow gradient tape.
    2. Learning rate: 1e-3 to 1e-4 for stable convergence.
    3. Batch size for RCWA: 50–200 unit cells per minibatch.
    4. Loss function: -Strehl_ratio + α * phase_error_penalty + β * fabrication_penalty.
    5. Use circular (radial) expansion to reduce from 2D to 1D optimization.
    6. Convergence criterion: FOM change < 1e-4 over 50 iterations.
    7. Early stopping with best-model tracking recommended.
    """,
    """
    Fabrication Constraints and Penalties:
    Minimum feature size: typically 80–150 nm (e-beam), 200–500 nm (DUV lithography).
    Aspect ratio limit: H/D < 15 to avoid pillar collapse during fabrication.
    Proximity effect: pillars closer than 50 nm interact electromagnetically.
    Post-fabrication correction: add 10–20 nm to all feature sizes.
    Penalty function for over-constraint: P = Σ max(0, D_min - D_i)² 
    Binary constraint for grayscale lithography: use continuous relaxation first,
    then binarize with threshold at 0.5.
    """,
    """
    GDS (GDSII Layout) Export:
    After optimization, export to GDS for e-beam or photolithography.
    Use gdsfactory for layout generation. Assign layers: (1,0) for meta-atoms,
    (2,0) for alignment marks, (3,0) for write-field boundaries.
    Minimum polygon vertices: 4 (rectangles), circles approximated with 64 vertices.
    Cell hierarchy: top-cell → write_field_cell → unit_cell.
    DRC (Design Rule Check): verify minimum feature and spacing before tape-out.
    """,
    """
    Performance Metrics for Metalenses:
    1. Strehl Ratio: I_measured / I_ideal (range 0–1, target > 0.8)
    2. Focusing Efficiency: power within 3× Airy disk / total transmitted power
    3. Chromatic Aberration: Δf/f per 100 nm bandwidth (target < 1% for broadband)
    4. Strehl Bandwidth: wavelength range over which Strehl > 0.8
    5. MTF (Modulation Transfer Function): spatial frequency response
    6. PSF (Point Spread Function): intensity at focal plane
    7. Transmission efficiency: fraction of incident power transmitted
    """,
    """
    Self-RAG Design Iteration Protocol:
    Round 1: Generate initial design from NL requirements + knowledge base.
    Simulate → Evaluate FOM → If FOM < threshold, self-reflect on failure modes.
    Round 2: Retrieve targeted knowledge for identified failure (phase error / efficiency).
    Adjust design parameters based on reflection and new knowledge.
    Round 3+: Fine-tune with gradient-based optimizer, use RCWA feedback.
    Terminate: FOM ≥ target OR max_iterations reached OR convergence detected.
    Document: Log all iterations, parameter history, and final design summary.
    """,
    """
    Material Properties for Metalens Design:
    TiO2: n ≈ 2.35 @ 532nm, 2.31 @ 633nm, 2.25 @ 780nm. Low loss in visible.
    Si (amorphous): n ≈ 3.5 @ 1550nm, high absorption in visible (k > 0.1).
    Si (crystalline): n ≈ 3.48 @ 1550nm, preferred for telecom wavelengths.
    GaN: n ≈ 2.32 @ 405nm, excellent for UV/blue, wide bandgap.
    SiN: n ≈ 2.0 @ 532nm, CMOS-compatible, good for visible.
    Substrate (SiO2/glass): n ≈ 1.46, use as low-index background.
    """,
    """
    Metamodel (Neural Surrogate) for Fast Design:
    Instead of running RCWA for every candidate, train a neural network surrogate.
    Input: [wavelength, pillar_diameter, height, ...other_features]
    Output: [tx_real, tx_imag, ty_real, ty_imag] (complex transmission)
    Architecture: 4-layer FCC [64, 128, 256, 64] with ReLU activations.
    Training: sample 50–200 parameter combinations per feature dimension.
    Accuracy target: |t_predicted - t_RCWA| < 0.02 (amplitude), < 0.05 rad (phase).
    Use metamodel for gradient-based optimization, validate with RCWA at the end.
    """,
    """
    Numerical Aperture and Beam Focusing:
    For a metalens with diameter D and focal length f:
      NA = D / (2 * sqrt(D²/4 + f²))
    The focal spot size (FWHM) at diffraction limit: w = 0.51λ/NA
    Depth of focus (DOF): DOF = λ / (2 * NA²)
    F-number: F/# = f/D
    For imaging applications, NA > 0.3 is typically required.
    For collimation/beam-shaping: lower NA (0.05–0.2) sufficient.
    """,
    """
    LangGraph AutoML Workflow States:
    INIT → PARSE → RETRIEVE → GRADE → GENERATE → SIMULATE → EVALUATE → REFLECT
    REFLECT conditions:
      - If Strehl < 0.5: major redesign needed (regenerate params)
      - If 0.5 ≤ Strehl < target: fine-tune with optimizer
      - If Strehl ≥ target: success, export design
    Re-retrieval triggers: phase_error > 0.3 rad, transmission < 0.5, fabrication_violation.
    Max iterations: 10 (configurable). Early termination on convergence.
    Output: optimized design parameters, simulation results, GDS-ready layout.
    """,
]


# ─────────────────────────────────────────────────────────────────────────────
# STATE DEFINITION
# ─────────────────────────────────────────────────────────────────────────────

class AutoMLState(TypedDict):
    """LangGraph state for the AutoML metalens design pipeline."""
    # ── Input ──────────────────────────────────────────────────────────────
    user_requirements:      str             # Natural language design requirements
    design_constraints:     dict            # Parsed hard constraints

    # ── RAG ────────────────────────────────────────────────────────────────
    query:                  str             # Current retrieval query
    retrieved_documents:    list[str]       # Retrieved document texts
    document_grades:        list[float]     # Relevance scores [0,1]
    retrieval_retries:      int             # How many times we've re-retrieved

    # ── Design Parameters ──────────────────────────────────────────────────
    design_params:          dict            # Current metalens design parameters
    param_history:          list[dict]      # History of all parameter dicts tried

    # ── Simulation ─────────────────────────────────────────────────────────
    simulation_results:     dict            # Latest simulation output
    fom_history:            list[float]     # FOM (Strehl) over iterations

    # ── Evaluation & Reflection ────────────────────────────────────────────
    current_fom:            float           # Current figure of merit
    failure_modes:          list[str]       # Identified failure modes
    reflection_notes:       str             # LLM self-reflection text
    optimize_iter:          int             # Current optimization iteration

    # ── Control Flow ───────────────────────────────────────────────────────
    status:                 str             # "running" | "success" | "failed" | "converged"
    error_message:          str             # Any error message

    # ── Output ─────────────────────────────────────────────────────────────
    final_design:           dict            # Final accepted design
    design_summary:         str             # Human-readable design summary
    messages:               list[str]       # Progress log for GUI


# ─────────────────────────────────────────────────────────────────────────────
# METALENS DESIGN PARAMETERS (typed)
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class MetalensDesignParams:
    """Structured metalens design parameters."""
    wavelength_nm:       float   = 532.0    # Design wavelength [nm]
    focal_length_mm:     float   = 1.0      # Focal length [mm]
    diameter_mm:         float   = 0.5      # Lens diameter [mm]
    pillar_material:     str     = "TiO2"   # Meta-atom material
    pillar_height_nm:    float   = 500.0    # Pillar height [nm]
    min_diameter_nm:     float   = 80.0     # Min pillar diameter [nm]
    max_diameter_nm:     float   = 300.0    # Max pillar diameter [nm]
    periodicity_nm:      float   = 350.0    # Unit cell period [nm]
    xy_harmonics:        tuple   = (3, 3)   # RCWA harmonics
    n_radial_pixels:     int     = 50       # Number of radial pixels
    substrate_index:     float   = 1.46     # Substrate refractive index
    jones_vector:        tuple   = (1, 0)   # Polarization (x-pol default)

    def to_dict(self) -> dict:
        return {
            "wavelength_nm":    self.wavelength_nm,
            "focal_length_mm":  self.focal_length_mm,
            "diameter_mm":      self.diameter_mm,
            "pillar_material":  self.pillar_material,
            "pillar_height_nm": self.pillar_height_nm,
            "min_diameter_nm":  self.min_diameter_nm,
            "max_diameter_nm":  self.max_diameter_nm,
            "periodicity_nm":   self.periodicity_nm,
            "xy_harmonics":     list(self.xy_harmonics),
            "n_radial_pixels":  self.n_radial_pixels,
            "substrate_index":  self.substrate_index,
            "jones_vector":     list(self.jones_vector),
            "numerical_aperture": self.numerical_aperture,
            "f_number":         self.f_number,
        }

    @property
    def numerical_aperture(self) -> float:
        D = self.diameter_mm
        f = self.focal_length_mm
        return D / (2 * np.sqrt((D/2)**2 + f**2))

    @property
    def f_number(self) -> float:
        return self.focal_length_mm / self.diameter_mm

    @classmethod
    def from_dict(cls, d: dict) -> "MetalensDesignParams":
        return cls(
            wavelength_nm       = d.get("wavelength_nm",    532.0),
            focal_length_mm     = d.get("focal_length_mm",  1.0),
            diameter_mm         = d.get("diameter_mm",      0.5),
            pillar_material     = d.get("pillar_material",  "TiO2"),
            pillar_height_nm    = d.get("pillar_height_nm", 500.0),
            min_diameter_nm     = d.get("min_diameter_nm",  80.0),
            max_diameter_nm     = d.get("max_diameter_nm",  300.0),
            periodicity_nm      = d.get("periodicity_nm",   350.0),
            xy_harmonics        = tuple(d.get("xy_harmonics", [3, 3])),
            n_radial_pixels     = d.get("n_radial_pixels",  50),
            substrate_index     = d.get("substrate_index",  1.46),
            jones_vector        = tuple(d.get("jones_vector", [1, 0])),
        )


# ─────────────────────────────────────────────────────────────────────────────
# KNOWLEDGE BASE
# ─────────────────────────────────────────────────────────────────────────────

class MetalensKnowledgeBase:
    """
    In-memory FAISS vector store populated with the metalens design corpus.
    Supports semantic retrieval with relevance grading (Self-RAG step 1).
    """

    def __init__(self, embed_model: str = OLLAMA_EMBED_MODEL):
        self.embed_model = embed_model
        self._vectorstore: Optional[FAISS] = None
        self._embeddings  = None
        self._initialized = False

    def initialize(self) -> None:
        """Build the vector store from the corpus."""
        if self._initialized:
            return

        logger.info("Initializing MetalensKnowledgeBase…")
        try:
            self._embeddings = OllamaEmbeddings(
                model=self.embed_model,
                base_url=OLLAMA_BASE_URL,
            )
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800, chunk_overlap=80
            )
            docs = []
            for i, text in enumerate(METALENS_KNOWLEDGE_CORPUS):
                chunks = splitter.create_documents(
                    [text],
                    metadatas=[{"source": f"corpus_{i}"}]
                )
                docs.extend(chunks)

            self._vectorstore = FAISS.from_documents(docs, self._embeddings)
            self._initialized = True
            logger.info(f"Knowledge base ready — {len(docs)} chunks indexed.")
        except Exception as e:
            logger.warning(f"Ollama embeddings unavailable ({e}). Using keyword fallback.")
            self._vectorstore = None
            self._initialized = True  # mark ready even without embeddings

    def retrieve(self, query: str, k: int = 4) -> list[Document]:
        """Retrieve top-k relevant documents for a query."""
        self.initialize()
        if self._vectorstore is not None:
            try:
                return self._vectorstore.similarity_search(query, k=k)
            except Exception as e:
                logger.warning(f"FAISS search failed: {e}")

        # Keyword-based fallback
        query_tokens = set(query.lower().split())
        scored = []
        for i, text in enumerate(METALENS_KNOWLEDGE_CORPUS):
            text_tokens = set(text.lower().split())
            score = len(query_tokens & text_tokens) / max(len(query_tokens), 1)
            scored.append((score, i, text))
        scored.sort(reverse=True)
        return [
            Document(page_content=text, metadata={"source": f"corpus_{i}", "score": s})
            for s, i, text in scored[:k]
        ]

    def retrieve_texts(self, query: str, k: int = 4) -> list[str]:
        """Return document texts only."""
        docs = self.retrieve(query, k=k)
        return [d.page_content for d in docs]


# ─────────────────────────────────────────────────────────────────────────────
# METALENS SIMULATOR  (wraps metabox / uses analytical fallback)
# ─────────────────────────────────────────────────────────────────────────────

class MetalensSimulator:
    """
    Bridges AutoML with the metabox RCWA simulation backend.
    Falls back to an analytical Strehl estimator when metabox is unavailable.
    """

    def __init__(self):
        self._metabox_available = self._check_metabox()

    @staticmethod
    def _check_metabox() -> bool:
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
            from metabox import rcwa, assembly, utils  # noqa: F401
            return True
        except ImportError:
            return False

    def simulate(self, params: MetalensDesignParams) -> dict:
        """Run simulation and return metrics dict."""
        if self._metabox_available:
            return self._simulate_metabox(params)
        return self._simulate_analytical(params)

    def _simulate_metabox(self, params: MetalensDesignParams) -> dict:
        """Full RCWA simulation via metabox."""
        try:
            from metabox import rcwa, utils, assembly

            wl_m = params.wavelength_nm * 1e-9
            period_m = params.periodicity_nm * 1e-9
            height_m = params.pillar_height_nm * 1e-9
            d_min_m  = params.min_diameter_nm  * 1e-9
            d_max_m  = params.max_diameter_nm  * 1e-9

            # Build ProtoUnitCell
            diam_feature = utils.Feature(
                vmin=d_min_m, vmax=d_max_m, name="diameter", sampling=20
            )
            pillar = rcwa.Circle(material=2.35, radius=diam_feature)
            layer  = rcwa.Layer(material=1.0, thickness=height_m, shapes=(pillar,))
            uc     = rcwa.UnitCell(
                layers=[layer],
                periodicity=(period_m, period_m),
                refl_index=1.0,
                tran_index=params.substrate_index,
            )
            proto  = rcwa.ProtoUnitCell(uc)

            incidence = utils.Incidence(wavelength=(wl_m,))
            sim_cfg   = rcwa.SimConfig(
                xy_harmonics=params.xy_harmonics,
                resolution=64,
                return_tensor=True,
                return_zeroth_order=True,
                use_transmission=True,
            )

            # Sample phase coverage
            n_samples = min(params.n_radial_pixels, 30)
            diameters  = np.linspace(d_min_m, d_max_m, n_samples)
            import tensorflow as tf
            param_tensor = tf.constant(diameters.reshape(1, -1), dtype=tf.float32)

            result = rcwa.simulate_parameterized_unit_cells(
                param_tensor, proto, incidence, sim_cfg
            )
            tx = result[:, :, 0].numpy()
            amplitude = np.abs(tx).mean()
            phase_vals = np.angle(tx[0])
            phase_range = float(np.max(phase_vals) - np.min(phase_vals))
            phase_uniformity = float(1.0 - np.std(np.diff(phase_vals)) / np.pi)

            # Strehl estimate
            strehl = min(1.0, amplitude**2 * max(0, phase_uniformity) *
                         min(1.0, phase_range / (2 * np.pi)))

            return {
                "strehl_ratio":      round(strehl, 4),
                "transmission":      round(float(amplitude), 4),
                "phase_range_rad":   round(phase_range, 4),
                "phase_uniformity":  round(phase_uniformity, 4),
                "na":                round(params.numerical_aperture, 4),
                "f_number":          round(params.f_number, 4),
                "engine":            "metabox_rcwa",
            }
        except Exception as e:
            logger.warning(f"metabox simulation failed: {e}\nFalling back to analytical.")
            return self._simulate_analytical(params)

    def _simulate_analytical(self, params: MetalensDesignParams) -> dict:
        """
        Analytical figure-of-merit estimator for use when metabox is unavailable.
        Scores design parameters against known heuristics.
        """
        score = 0.0
        notes = []

        # ── Phase coverage heuristic ──────────────────────────────────────
        wl_nm     = params.wavelength_nm
        height_nm = params.pillar_height_nm
        n_mat     = {"TiO2": 2.35, "Si": 3.48, "GaN": 2.32, "SiN": 2.0}.get(
                      params.pillar_material, 2.0)
        phase_range = (2 * np.pi * (n_mat - 1) * height_nm) / wl_nm
        phase_score = min(1.0, phase_range / (2 * np.pi))
        score += 0.35 * phase_score

        # ── Periodicity heuristic ─────────────────────────────────────────
        period_to_wl = params.periodicity_nm / wl_nm
        if 0.6 <= period_to_wl <= 1.1:
            period_score = 1.0
        elif period_to_wl < 0.6:
            period_score = period_to_wl / 0.6
        else:
            period_score = max(0, 1.0 - (period_to_wl - 1.1) * 2)
        score += 0.25 * period_score
        notes.append(f"period_ratio={period_to_wl:.2f}")

        # ── Aspect-ratio constraint ────────────────────────────────────────
        mean_d_nm   = (params.min_diameter_nm + params.max_diameter_nm) / 2
        aspect_ratio = params.pillar_height_nm / max(mean_d_nm, 1)
        if aspect_ratio <= 10:
            ar_score = 1.0
        elif aspect_ratio <= 15:
            ar_score = 1.0 - (aspect_ratio - 10) / 5
        else:
            ar_score = 0.0
        score += 0.20 * ar_score

        # ── NA feasibility ────────────────────────────────────────────────
        na = params.numerical_aperture
        na_score = 1.0 if na <= 0.6 else max(0, 1.0 - (na - 0.6) / 0.4)
        score += 0.20 * na_score

        # ── Transmission estimate ──────────────────────────────────────────
        transmission = 0.5 + 0.5 * phase_score * ar_score

        return {
            "strehl_ratio":     round(min(score, 0.98), 4),
            "transmission":     round(float(transmission), 4),
            "phase_range_rad":  round(float(phase_range), 4),
            "phase_uniformity": round(float(phase_score), 4),
            "na":               round(params.numerical_aperture, 4),
            "f_number":         round(params.f_number, 4),
            "notes":            ", ".join(notes),
            "engine":           "analytical_heuristic",
        }


# ─────────────────────────────────────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

PARSE_REQUIREMENTS_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     """You are an expert nanophotonics engineer specializing in metalens design.
Extract design requirements from the user's request and return ONLY a valid JSON object.

JSON Schema:
{{
  "wavelength_nm":    <float, design wavelength in nm>,
  "focal_length_mm":  <float, focal length in mm>,
  "diameter_mm":      <float, lens diameter in mm>,
  "pillar_material":  <"TiO2"|"Si"|"GaN"|"SiN">,
  "pillar_height_nm": <float, pillar height in nm>,
  "min_diameter_nm":  <float, minimum pillar diameter in nm>,
  "max_diameter_nm":  <float, maximum pillar diameter in nm>,
  "periodicity_nm":   <float, unit cell period in nm>,
  "n_radial_pixels":  <int, number of radial design points>,
  "target_strehl":    <float, 0.0-1.0>,
  "application":      <str, brief description>
}}

If not specified, use sensible defaults for the given wavelength.
Respond with ONLY the JSON object, no extra text."""),
    ("human", "User Requirements:\n{requirements}\n\nKnowledge Context:\n{context}")
])

GRADE_DOCUMENTS_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     """You are grading retrieved documents for relevance to a metalens design query.
Return ONLY a JSON object: {{"score": <0.0 to 1.0>, "reason": "<brief reason>"}}
Score > 0.6 means the document is relevant and useful."""),
    ("human", "Query: {query}\n\nDocument: {document}")
])

GENERATE_PARAMS_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     """You are an expert metalens designer. Given design constraints and retrieved knowledge,
generate optimal metalens design parameters. Return ONLY a valid JSON object matching this schema:

{{
  "wavelength_nm":    <float>,
  "focal_length_mm":  <float>,
  "diameter_mm":      <float>,
  "pillar_material":  <"TiO2"|"Si"|"GaN"|"SiN">,
  "pillar_height_nm": <float, optimize for 2π phase range>,
  "min_diameter_nm":  <float, ≥80 nm for fabricability>,
  "max_diameter_nm":  <float, ≤periodicity*0.9>,
  "periodicity_nm":   <float, 0.6–1.1× wavelength>,
  "n_radial_pixels":  <int, 30–100>,
  "xy_harmonics":     [<int_odd>, <int_odd>],
  "substrate_index":  <float>
}}

Apply physics-based reasoning. Ensure full 2π phase coverage and fabrication constraints."""),
    ("human",
     "Design Constraints:\n{constraints}\n\nKnowledge:\n{knowledge}\n\nReflection Notes:\n{reflection}")
])

REFLECT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     """You are a metalens design critic performing self-reflection.
Analyze the simulation results and identify failure modes and improvements.
Return ONLY a JSON object:
{{
  "failure_modes": [<list of str>],
  "improvement_actions": [<list of str>],
  "updated_params": {{<only params to change>}},
  "reflection": "<2-3 sentence technical summary>",
  "should_redesign": <true|false>,
  "confidence": <0.0-1.0>
}}"""),
    ("human",
     "Current Params:\n{params}\n\nSimulation Results:\n{results}\n\nFOM History:\n{fom_history}\n\nKnowledge:\n{knowledge}\n\nIteration: {iteration}")
])

SUMMARIZE_DESIGN_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     """You are a photonics engineer writing a concise technical design summary.
Summarize the final metalens design in clear, technical prose (3–5 sentences)."""),
    ("human",
     "Final Design Parameters:\n{params}\n\nSimulation Results:\n{results}\n\nIterations: {iterations}")
])


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH NODES
# ─────────────────────────────────────────────────────────────────────────────

class AutoMLNodes:
    """All LangGraph node functions for the AutoML metalens pipeline."""

    def __init__(
        self,
        llm:        ChatOllama,
        kb:         MetalensKnowledgeBase,
        simulator:  MetalensSimulator,
        callback:   Optional[Callable[[str], None]] = None,
    ):
        self.llm       = llm
        self.kb        = kb
        self.simulator = simulator
        self.callback  = callback or (lambda msg: None)

    def _log(self, msg: str) -> None:
        logger.info(msg)
        self.callback(msg)

    def _safe_llm_json(self, chain, inputs: dict, fallback: dict) -> dict:
        """Run LLM chain and safely parse JSON output."""
        try:
            raw = chain.invoke(inputs)
            text = raw.content if hasattr(raw, "content") else str(raw)
            # Strip markdown fences
            text = text.strip()
            for fence in ["```json", "```"]:
                text = text.replace(fence, "")
            # Extract JSON
            start = text.find("{")
            end   = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except Exception as e:
            logger.warning(f"JSON parse failed: {e}")
        return fallback

    # ── Node 1: Parse Requirements ─────────────────────────────────────────
    def parse_requirements(self, state: AutoMLState) -> AutoMLState:
        self._log("📋  Parsing design requirements…")
        req = state["user_requirements"]

        # Quick retrieval for context
        context_docs = self.kb.retrieve_texts("metalens design parameters wavelength", k=2)
        context = "\n---\n".join(context_docs[:2])

        chain   = PARSE_REQUIREMENTS_PROMPT | self.llm
        default = MetalensDesignParams().to_dict()
        default["target_strehl"] = 0.85
        default["application"]   = "general purpose metalens"

        parsed = self._safe_llm_json(chain, {"requirements": req, "context": context}, default)

        # Merge with defaults
        default.update({k: v for k, v in parsed.items() if v is not None})

        state["design_constraints"] = default
        state["query"] = (
            f"metalens design {default.get('wavelength_nm',532)}nm wavelength "
            f"{default.get('pillar_material','TiO2')} pillars focal length design"
        )
        state["messages"] = state.get("messages", []) + [
            f"✅ Requirements parsed — λ={default.get('wavelength_nm')}nm, "
            f"f={default.get('focal_length_mm')}mm, Ø={default.get('diameter_mm')}mm"
        ]
        return state

    # ── Node 2: RAG Retrieve ───────────────────────────────────────────────
    def rag_retrieve(self, state: AutoMLState) -> AutoMLState:
        query   = state.get("query", "metalens design")
        retries = state.get("retrieval_retries", 0)
        self._log(f"🔍  Retrieving knowledge (attempt {retries+1})…")

        docs = self.kb.retrieve_texts(query, k=5)
        state["retrieved_documents"] = docs
        state["messages"] = state.get("messages", []) + [
            f"📚 Retrieved {len(docs)} knowledge chunks for: '{query[:60]}…'"
        ]
        return state

    # ── Node 3: Grade Documents (Self-RAG ─────────────────────────────────
    def grade_documents(self, state: AutoMLState) -> AutoMLState:
        self._log("📊  Grading document relevance (Self-RAG)…")
        query = state.get("query", "")
        docs  = state.get("retrieved_documents", [])
        chain = GRADE_DOCUMENTS_PROMPT | self.llm

        grades = []
        for doc in docs[:4]:   # grade top 4 only for speed
            result = self._safe_llm_json(
                chain,
                {"query": query, "document": doc[:500]},
                {"score": 0.5, "reason": "default"}
            )
            grades.append(float(result.get("score", 0.5)))

        state["document_grades"] = grades
        avg_grade = np.mean(grades) if grades else 0.0
        self._log(f"   Avg relevance score: {avg_grade:.2f}")

        state["messages"] = state.get("messages", []) + [
            f"📊 Document grades: {[f'{g:.2f}' for g in grades]} — avg={avg_grade:.2f}"
        ]
        return state

    # ── Node 4: Generate Design Parameters ────────────────────────────────
    def generate_params(self, state: AutoMLState) -> AutoMLState:
        self._log("⚙️   Generating metalens design parameters…")
        constraints = state.get("design_constraints", {})
        docs        = state.get("retrieved_documents", [])
        reflection  = state.get("reflection_notes", "No prior reflection.")

        knowledge   = "\n\n".join(docs[:4])
        chain       = GENERATE_PARAMS_PROMPT | self.llm

        raw_params = self._safe_llm_json(
            chain,
            {
                "constraints": json.dumps(constraints, indent=2),
                "knowledge":   knowledge[:2000],
                "reflection":  reflection,
            },
            {}
        )

        # Merge constraints with generated params
        base = {k: v for k, v in constraints.items() if k in MetalensDesignParams.__dataclass_fields__}
        base.update({k: v for k, v in raw_params.items() if v is not None})

        # Apply safety bounds
        base["min_diameter_nm"]  = max(60.0,  float(base.get("min_diameter_nm",  80.0)))
        base["max_diameter_nm"]  = min(
            float(base.get("periodicity_nm", 350.0)) * 0.85,
            float(base.get("max_diameter_nm", 300.0))
        )
        base["pillar_height_nm"] = max(100.0, min(2000.0, float(base.get("pillar_height_nm", 500.0))))

        # Store
        history = state.get("param_history", [])
        history.append(base)
        state["design_params"]  = base
        state["param_history"]  = history

        params_obj = MetalensDesignParams.from_dict(base)
        state["messages"] = state.get("messages", []) + [
            f"⚙️  Design params generated — NA={params_obj.numerical_aperture:.3f}, "
            f"F/#{params_obj.f_number:.1f}, "
            f"h={base.get('pillar_height_nm')}nm, P={base.get('periodicity_nm')}nm"
        ]
        return state

    # ── Node 5: Simulate ───────────────────────────────────────────────────
    def simulate(self, state: AutoMLState) -> AutoMLState:
        self._log("🔬  Running RCWA simulation…")
        params = MetalensDesignParams.from_dict(state.get("design_params", {}))
        t0     = time.perf_counter()

        results = self.simulator.simulate(params)
        dt      = time.perf_counter() - t0

        state["simulation_results"] = results
        strehl = float(results.get("strehl_ratio", 0.0))
        state["current_fom"] = strehl

        fom_hist = state.get("fom_history", [])
        fom_hist.append(strehl)
        state["fom_history"] = fom_hist

        state["messages"] = state.get("messages", []) + [
            f"🔬 Simulation done ({dt:.2f}s) — "
            f"Strehl={strehl:.4f}, T={results.get('transmission',0):.3f}, "
            f"φ_range={results.get('phase_range_rad',0):.2f} rad  [{results.get('engine','')}]"
        ]
        return state

    # ── Node 6: Evaluate ───────────────────────────────────────────────────
    def evaluate(self, state: AutoMLState) -> AutoMLState:
        strehl     = state.get("current_fom", 0.0)
        target     = state.get("design_constraints", {}).get("target_strehl", FOM_TARGET_THRESHOLD)
        iteration  = state.get("optimize_iter", 0)
        max_iters  = MAX_OPTIMIZE_ITERS

        self._log(f"📈  Evaluate iter={iteration}: Strehl={strehl:.4f} / target={target:.2f}")

        # Detect convergence
        fom_hist = state.get("fom_history", [strehl])
        converged = (
            len(fom_hist) >= 3 and
            max(abs(fom_hist[-1] - fom_hist[-2]),
                abs(fom_hist[-2] - fom_hist[-3])) < 1e-4
        )

        if strehl >= target:
            state["status"] = "success"
            self._log(f"✅  Target Strehl reached: {strehl:.4f} ≥ {target:.2f}")
        elif iteration >= max_iters:
            state["status"] = "max_iters"
            self._log(f"⚠️   Max iterations ({max_iters}) reached. Best Strehl: {max(fom_hist):.4f}")
        elif converged:
            state["status"] = "converged"
            self._log(f"📐  Converged (no improvement). Best Strehl: {max(fom_hist):.4f}")
        else:
            state["status"] = "running"

        state["optimize_iter"] = iteration + 1
        return state

    # ── Node 7: Self-Reflect ───────────────────────────────────────────────
    def self_reflect(self, state: AutoMLState) -> AutoMLState:
        self._log("🧠  Self-reflecting on results…")
        params  = state.get("design_params", {})
        results = state.get("simulation_results", {})
        fom_hist= state.get("fom_history", [])
        docs    = state.get("retrieved_documents", [])
        knowledge = "\n\n".join(docs[:3])

        chain  = REFLECT_PROMPT | self.llm
        output = self._safe_llm_json(
            chain,
            {
                "params":      json.dumps(params, indent=2),
                "results":     json.dumps(results, indent=2),
                "fom_history": str(fom_hist),
                "knowledge":   knowledge[:1500],
                "iteration":   state.get("optimize_iter", 1),
            },
            {
                "failure_modes":       ["insufficient phase coverage"],
                "improvement_actions": ["increase pillar height"],
                "updated_params":      {},
                "reflection":          "Phase coverage is insufficient. Increase height.",
                "should_redesign":     True,
                "confidence":          0.5,
            }
        )

        state["failure_modes"]    = output.get("failure_modes", [])
        state["reflection_notes"] = output.get("reflection", "")

        # Update query for re-retrieval
        modes = output.get("failure_modes", [])
        if modes:
            state["query"] = f"metalens design optimization {' '.join(modes[:3])}"

        # Apply suggested parameter updates
        updated = output.get("updated_params", {})
        if updated and isinstance(updated, dict):
            current = state.get("design_params", {}).copy()
            current.update(updated)
            state["design_params"] = current

        state["messages"] = state.get("messages", []) + [
            f"🧠 Reflection: {output.get('reflection','')[:120]}",
            f"   Failures: {', '.join(output.get('failure_modes',[])[:3])}",
            f"   Should redesign: {output.get('should_redesign', False)}",
        ]
        return state

    # ── Node 8: Finalize ───────────────────────────────────────────────────
    def finalize(self, state: AutoMLState) -> AutoMLState:
        self._log("🏁  Finalizing design…")

        # Pick best design from history
        fom_hist   = state.get("fom_history", [0.0])
        best_idx   = int(np.argmax(fom_hist))
        param_hist = state.get("param_history", [state.get("design_params", {})])
        best_params= param_hist[min(best_idx, len(param_hist) - 1)]

        state["final_design"] = best_params

        # Generate summary
        chain  = SUMMARIZE_DESIGN_PROMPT | self.llm
        try:
            raw = chain.invoke({
                "params":     json.dumps(best_params, indent=2),
                "results":    json.dumps(state.get("simulation_results", {}), indent=2),
                "iterations": state.get("optimize_iter", 1),
            })
            summary = raw.content if hasattr(raw, "content") else str(raw)
        except Exception:
            p = MetalensDesignParams.from_dict(best_params)
            summary = (
                f"Metalens designed for λ={best_params.get('wavelength_nm')}nm with "
                f"f={best_params.get('focal_length_mm')}mm focal length and "
                f"NA={p.numerical_aperture:.3f}. "
                f"Achieved Strehl ratio {max(fom_hist):.4f} after {len(fom_hist)} iterations."
            )
        state["design_summary"] = summary
        state["messages"] = state.get("messages", []) + [
            f"🏁 Design complete. Best Strehl={max(fom_hist):.4f}",
            f"📝 {summary[:200]}",
        ]
        return state


# ─────────────────────────────────────────────────────────────────────────────
# CONDITIONAL EDGES
# ─────────────────────────────────────────────────────────────────────────────

def should_re_retrieve(state: AutoMLState) -> Literal["generate_params", "rag_retrieve"]:
    """Re-retrieve if average document grade is below cutoff."""
    grades  = state.get("document_grades", [1.0])
    retries = state.get("retrieval_retries", 0)
    avg     = np.mean(grades) if grades else 0.0
    if avg < DOCUMENT_GRADE_CUTOFF and retries < MAX_RETRIEVAL_RETRIES:
        state["retrieval_retries"] = retries + 1
        return "rag_retrieve"
    return "generate_params"


def should_continue_optimizing(
    state: AutoMLState
) -> Literal["self_reflect", "finalize"]:
    """Route to reflection/optimization or finalize based on status."""
    status = state.get("status", "running")
    if status in ("success", "max_iters", "converged"):
        return "finalize"
    return "self_reflect"


def should_redesign(state: AutoMLState) -> Literal["rag_retrieve", "simulate"]:
    """After reflection: re-retrieve+redesign or directly re-simulate."""
    modes = state.get("failure_modes", [])
    major = any(m in " ".join(modes).lower()
                for m in ["phase coverage", "material", "periodicity", "major"])
    if major:
        return "rag_retrieve"
    return "simulate"


# ─────────────────────────────────────────────────────────────────────────────
# MAIN AutoML CLASS
# ─────────────────────────────────────────────────────────────────────────────

class MetalensAutoML:
    """
    Top-level AutoML controller for metalens design.
    Builds and executes the LangGraph Self-RAG pipeline.

    Usage::

        automl = MetalensAutoML()
        result = automl.run(
            "Design a metalens for 532nm, 1mm focal length, 0.5mm diameter"
        )
    """

    def __init__(
        self,
        model:       str                        = OLLAMA_MODEL,
        base_url:    str                        = OLLAMA_BASE_URL,
        temperature: float                      = 0.2,
        callback:    Optional[Callable]         = None,
    ):
        self.callback = callback or (lambda msg: None)

        # ── LLM ───────────────────────────────────────────────────────────
        self.llm = ChatOllama(
            model=model,
            base_url=base_url,
            temperature=temperature,
            num_predict=1024,
        )
        # ── Knowledge Base ────────────────────────────────────────────────
        self.kb = MetalensKnowledgeBase()
        self.kb.initialize()

        # ── Simulator ─────────────────────────────────────────────────────
        self.simulator = MetalensSimulator()

        # ── Nodes ─────────────────────────────────────────────────────────
        self.nodes = AutoMLNodes(self.llm, self.kb, self.simulator, callback)

        # ── Build Graph ───────────────────────────────────────────────────
        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        """Construct the LangGraph StateGraph."""
        g = StateGraph(AutoMLState)
        n = self.nodes

        # Register nodes
        g.add_node("parse_requirements", n.parse_requirements)
        g.add_node("rag_retrieve",        n.rag_retrieve)
        g.add_node("grade_documents",     n.grade_documents)
        g.add_node("generate_params",     n.generate_params)
        g.add_node("simulate",            n.simulate)
        g.add_node("evaluate",            n.evaluate)
        g.add_node("self_reflect",        n.self_reflect)
        g.add_node("finalize",            n.finalize)

        # Edges
        g.add_edge(START, "parse_requirements")
        g.add_edge("parse_requirements", "rag_retrieve")
        g.add_edge("rag_retrieve",       "grade_documents")

        # Self-RAG conditional: re-retrieve or proceed
        g.add_conditional_edges(
            "grade_documents",
            should_re_retrieve,
            {"rag_retrieve": "rag_retrieve", "generate_params": "generate_params"},
        )

        g.add_edge("generate_params", "simulate")
        g.add_edge("simulate",        "evaluate")

        # Route after evaluation
        g.add_conditional_edges(
            "evaluate",
            should_continue_optimizing,
            {"self_reflect": "self_reflect", "finalize": "finalize"},
        )

        # After reflection: redesign or just re-simulate
        g.add_conditional_edges(
            "self_reflect",
            should_redesign,
            {"rag_retrieve": "rag_retrieve", "simulate": "simulate"},
        )

        g.add_edge("finalize", END)

        return g.compile()

    def run(self, requirements: str) -> dict:
        """
        Execute the AutoML pipeline.

        Args:
            requirements: Natural language design requirements.

        Returns:
            dict with keys: final_design, design_summary, fom_history,
                            simulation_results, messages, status.
        """
        initial_state: AutoMLState = {
            "user_requirements":   requirements,
            "design_constraints":  {},
            "query":               "",
            "retrieved_documents": [],
            "document_grades":     [],
            "retrieval_retries":   0,
            "design_params":       {},
            "param_history":       [],
            "simulation_results":  {},
            "fom_history":         [],
            "current_fom":         0.0,
            "failure_modes":       [],
            "reflection_notes":    "",
            "optimize_iter":       0,
            "status":              "running",
            "error_message":       "",
            "final_design":        {},
            "design_summary":      "",
            "messages":            [],
        }

        self.callback("🚀  MetalensAutoML pipeline starting…")

        try:
            final_state = self.graph.invoke(initial_state)
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"AutoML pipeline error: {tb}")
            initial_state["status"]        = "failed"
            initial_state["error_message"] = str(e)
            initial_state["messages"].append(f"❌ Pipeline error: {e}")
            return initial_state

        return {
            "final_design":        final_state.get("final_design", {}),
            "design_summary":      final_state.get("design_summary", ""),
            "fom_history":         final_state.get("fom_history", []),
            "simulation_results":  final_state.get("simulation_results", {}),
            "messages":            final_state.get("messages", []),
            "status":              final_state.get("status", "unknown"),
            "param_history":       final_state.get("param_history", []),
        }

    def run_stream(self, requirements: str):
        """
        Generator that yields state updates as the pipeline progresses.
        Suitable for streaming progress to a GUI.
        """
        initial_state: AutoMLState = {
            "user_requirements":   requirements,
            "design_constraints":  {},
            "query":               "",
            "retrieved_documents": [],
            "document_grades":     [],
            "retrieval_retries":   0,
            "design_params":       {},
            "param_history":       [],
            "simulation_results":  {},
            "fom_history":         [],
            "current_fom":         0.0,
            "failure_modes":       [],
            "reflection_notes":    "",
            "optimize_iter":       0,
            "status":              "running",
            "error_message":       "",
            "final_design":        {},
            "design_summary":      "",
            "messages":            [],
        }

        try:
            for step in self.graph.stream(initial_state):
                yield step
        except Exception as e:
            yield {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# QUICK-TEST  (run: python core/automl.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    def printer(msg):
        print(f"  {msg}")

    print("=" * 70)
    print("  MetaOpticsAI — AutoML Self-RAG + LangGraph Test")
    print("=" * 70)

    automl = MetalensAutoML(callback=printer)
    result = automl.run(
        "Design a TiO2 metalens for 532 nm green laser, focal length 2 mm, "
        "diameter 1 mm. Optimize for maximum Strehl ratio."
    )

    print("\n── FINAL DESIGN ──────────────────────────────────────")
    print(json.dumps(result["final_design"], indent=2))
    print("\n── SUMMARY ───────────────────────────────────────────")
    print(result["design_summary"])
    print(f"\n── FOM History: {result['fom_history']}")
    print(f"── Status: {result['status']}")