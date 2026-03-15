"""FastAPI server for MyeloMemory inference.

Endpoints:
    POST /predict       — Single sample prediction
    POST /predict/batch — Batch prediction (up to max_batch_size)
    GET  /health        — Health check
    GET  /config        — Current model configuration
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from myelomemory.config import MyeloMemoryConfig
from myelomemory.inference.pipeline import MyeloMemoryPipeline
from myelomemory.utils.checkpoint import CheckpointManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """Request body for single-sample prediction."""

    protein_abundances: dict[str, float] = Field(
        ...,
        description="Mapping of protein/gene name → abundance value.",
        examples=[{"EZH2": 1.5, "DNMT1": 2.3, "TET2": 0.8}],
    )


class DrugPrediction(BaseModel):
    """Prediction for a single drug."""

    drug_name: str
    predicted_ic50: float = Field(description="Predicted IC50 (log scale)")
    resistance_probability: float = Field(
        description="P(resistant), derived from IC50"
    )
    reversibility_probability: float = Field(
        description="P(reversible), 0=locked-in, 1=easily reversible"
    )


class PredictResponse(BaseModel):
    """Response body for prediction endpoint."""

    stability_score: float = Field(
        description="Memory stability: 0=transient adaptation, 1=locked-in memory"
    )
    memory_state: list[float] = Field(
        description="64-dim latent epigenetic memory state embedding"
    )
    drug_predictions: list[DrugPrediction]
    interpretation: str = Field(
        description="Human-readable interpretation of the results"
    )


class BatchPredictRequest(BaseModel):
    """Request body for batch prediction."""

    samples: list[PredictRequest]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    device: str
    version: str


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(
    ckpt_mgr: CheckpointManager,
    config: MyeloMemoryConfig,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        ckpt_mgr: Checkpoint manager with trained model weights.
        config: Full pipeline configuration.

    Returns:
        Configured FastAPI app.
    """
    app = FastAPI(
        title="MyeloMemory API",
        description=(
            "AI pipeline for predicting epigenetic drug resistance memory "
            "states in multiple myeloma."
        ),
        version="0.1.0",
    )

    # Load pipeline at startup
    pipeline: MyeloMemoryPipeline | None = None

    @app.on_event("startup")
    async def load_pipeline() -> None:
        nonlocal pipeline
        try:
            pipeline = MyeloMemoryPipeline.from_checkpoints(ckpt_mgr, config)
            logger.info("Pipeline loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load pipeline: {e}")
            raise

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok" if pipeline is not None else "not_ready",
            model_loaded=pipeline is not None,
            device=str(config.device),
            version="0.1.0",
        )

    @app.get("/config")
    async def get_config() -> dict[str, Any]:
        return {
            "target_drugs": pipeline.drug_names if pipeline else config.data.target_drugs,
            "latent_dim": config.vae.latent_dim,
            "reader_writer_proteins": config.stability.reader_writer_proteins,
            "num_gnn_layers": config.gnn.num_layers,
        }

    @app.post("/predict", response_model=PredictResponse)
    async def predict(request: PredictRequest) -> PredictResponse:
        if pipeline is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        # Convert protein dict to tensor
        protein_names = list(request.protein_abundances.keys())
        values = list(request.protein_abundances.values())
        proteomics = torch.tensor(values, dtype=torch.float32)

        # Pad to expected input dimension if needed
        expected_dim = config.vae.input_dim
        if len(values) < expected_dim:
            padding = torch.zeros(expected_dim - len(values))
            proteomics = torch.cat([proteomics, padding])
            protein_names.extend(
                [f"_pad_{i}" for i in range(expected_dim - len(values))]
            )

        result = pipeline.predict_single(proteomics, protein_names)

        # Build drug predictions — only for drugs the model was trained on
        drug_preds = []
        for drug_name in pipeline.drug_names:
            ic50 = result.drug_resistance.get(drug_name, 0.0)
            rev = result.drug_reversibility.get(drug_name, 0.5)

            # Convert IC50 to resistance probability (sigmoid of normalized IC50)
            resistance_prob = 1.0 / (1.0 + 2.718 ** (-ic50))

            drug_preds.append(DrugPrediction(
                drug_name=drug_name,
                predicted_ic50=round(ic50, 4),
                resistance_probability=round(resistance_prob, 4),
                reversibility_probability=round(rev, 4),
            ))

        # Generate interpretation
        interpretation = _interpret(result.stability_score, drug_preds)

        return PredictResponse(
            stability_score=round(result.stability_score, 4),
            memory_state=result.memory_state.tolist(),
            drug_predictions=drug_preds,
            interpretation=interpretation,
        )

    @app.post("/predict/batch")
    async def predict_batch(request: BatchPredictRequest) -> list[PredictResponse]:
        if pipeline is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        if len(request.samples) > config.api.max_batch_size:
            raise HTTPException(
                status_code=400,
                detail=f"Batch size {len(request.samples)} exceeds max {config.api.max_batch_size}",
            )

        results = []
        for sample in request.samples:
            single_response = await predict(sample)
            results.append(single_response)

        return results

    return app


def _interpret(stability_score: float, drug_preds: list[DrugPrediction]) -> str:
    """Generate a human-readable interpretation of the results.

    Args:
        stability_score: Memory stability score.
        drug_preds: List of per-drug predictions.

    Returns:
        Interpretation string.
    """
    if stability_score > 0.7:
        stability_text = (
            "HIGH memory stability — this epigenetic state appears deeply "
            "locked in. Drug resistance driven by this state is likely "
            "IRREVERSIBLE through drug holidays alone."
        )
    elif stability_score > 0.4:
        stability_text = (
            "MODERATE memory stability — the epigenetic state shows partial "
            "locking. Some resistance may be reversible with sufficient "
            "washout time or epigenetic therapy."
        )
    else:
        stability_text = (
            "LOW memory stability — this appears to be a TRANSIENT adaptation. "
            "Drug resistance may be reversible through treatment breaks or "
            "sequential therapy."
        )

    resistant_drugs = [p.drug_name for p in drug_preds if p.resistance_probability > 0.6]
    reversible_drugs = [
        p.drug_name for p in drug_preds
        if p.resistance_probability > 0.6 and p.reversibility_probability > 0.5
    ]

    drugs_text = ""
    if resistant_drugs:
        drugs_text = f" Predicted resistance to: {', '.join(resistant_drugs)}."
        if reversible_drugs:
            drugs_text += (
                f" Of these, {', '.join(reversible_drugs)} resistance "
                "may be reversible."
            )

    return f"{stability_text}{drugs_text}"
