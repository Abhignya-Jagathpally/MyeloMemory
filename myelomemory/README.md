# MyeloMemory

An AI pipeline that infers epigenetic drug resistance memory states in multiple myeloma by integrating proteomic profiles with epigenomic signatures. MyeloMemory predicts not just **whether** a patient will resist treatment, but **how stable** that resistance is — distinguishing reversible adaptations from locked-in epigenetic memory.

## Key Innovation

Traditional drug resistance predictors use static molecular features. MyeloMemory introduces a **memory stability score** derived from ODE-based chromatin dynamics, quantifying how deeply an epigenetic resistance state is locked in. This enables clinicians to distinguish:

- **Transient adaptations** (low stability) — reversible through drug holidays
- **Locked-in memory** (high stability) — requires epigenetic therapy to overcome

## Architecture

MyeloMemory is a three-module pipeline:

```
Proteomic Profile (7,853 proteins)
         │
         ├──► Module 1: Conditional VAE ──► 64-dim Epigenetic Memory State
         │         (ProteomeToEpigenomeVAE)
         │
         ├──► Module 2: ODE Stability Scorer ──► Memory Stability Score [0, 1]
         │         (Sneppen-Ringrose chromatin ODE + Jacobian eigenvalue analysis)
         │
         └──► Module 3: Graph Attention Network ──► Per-Drug Resistance + Reversibility
                   (ResistanceGNN on STRING PPI network)
```

### Module 1: Proteome-to-Epigenome VAE

Conditional variational autoencoder that maps proteomic profiles to a 64-dimensional epigenetic memory state embedding. Trained on CCLE pan-cancer data (375 cell lines, 8,001 proteins after coverage filtering), then fine-tuned on hematological cell lines.

- Encoder: 8,001 → 2,048 → 1,024 → 512 → 64 (latent)
- Decoder: 64 → 512 → 1,024 → 2,048 → 42 (epigenomic features)
- Cyclical KL annealing, GELU activations, gradient checkpointing for bf16

### Module 2: Memory Stability Scorer

Quantifies how deeply an epigenetic state is locked in using a Sneppen-Ringrose bistability ODE system parameterized by 26 chromatin reader/writer protein levels (EZH2, DNMT1, TET2, HDAC1, etc.).

**Stability estimation via Jacobian eigenvalue analysis:**
1. Extract reader/writer protein levels from full proteome
2. Map to ODE parameters via learned neural network
3. Integrate chromatin ODE to steady state (Euler, step_size=0.1)
4. Compute 2x2 Jacobian at steady state via finite differences (float32)
5. Extract max eigenvalue via batched quadratic formula
6. Score = sigmoid(-lambda_max), centered on training distribution

### Module 3: Resistance Pathway GNN

Graph Attention Network operating on the STRING protein-protein interaction network (7,853 nodes, 460,888 edges at confidence >= 0.7). Each node receives the protein's abundance, the sample's memory state (broadcast), and stability score.

- 3-layer GAT with 4 attention heads, hidden_dim=128
- Global attention pooling
- Predicts per-drug IC50 and reversibility probability

## Data Sources

| Dataset | Source | Size |
|---------|--------|------|
| CCLE Proteomics | DepMap | 375 cell lines, 12,558 proteins |
| CCLE Epigenomics | DepMap | 897 cell lines, 43 chromatin features |
| STRING PPI | STRING v12 | 13.5M edges (460K after filtering) |
| Drug Sensitivity | GDSC/CTRPv2 | IC50 for Bortezomib, Lenalidomide |
| Validation | Kronke et al. 2024 | Held-out myeloma cohort |

## Results

### Pipeline Validation (367 test samples)

| Metric | Value |
|--------|-------|
| Mean stability score | 0.975 |
| Stability std | 0.078 |
| High stability samples | 357 (97.3%) |
| Medium stability samples | 10 (2.7%) |
| GNN best validation loss | 277.78 |
| Stability calibration loss | 0.010 |

### API Test Harness (5 clinical scenarios)

| Profile | Stability | Expected | Biological Order |
|---------|-----------|----------|-----------------|
| Multi-drug resistant MM | 0.9998 | High | -- |
| Bortezomib-resistant MM | 0.9966 | High | -- |
| Post-washout MM | 0.8032 | Medium | -- |
| Lenalidomide-sensitive MM | 0.7716 | Low | -- |
| Treatment-naive MM | 0.4425 | Medium | -- |

- Score spread: 0.56 (range 0.44 - 1.00)
- All 3 biological ordering checks pass
- Inference latency: ~0.35s per sample on H100

### Training Performance

| Stage | Duration (GPU) | Duration (CPU) | Speedup |
|-------|---------------|----------------|---------|
| Stability calibration | 1.8 min | ~30 min | 17x |
| GNN training (100 epochs) | 8.7 min | ~9 hours | 62x |
| Validation (367 samples) | 2.2 min | ~12 min | 5x |

## Quick Start

### One-command setup

```bash
cd myelomemory/
bash scripts/setup_and_run.sh
```

This will create the environment, download all datasets, auto-detect your GPU, and run the full pipeline with checkpointing.

### Manual setup

```bash
# Install dependencies
pip install -r requirements.txt

# Download datasets
python scripts/download_data.py --data-dir data/raw

# Run full pipeline (auto-detects checkpoints, skips completed stages)
python main.py --config configs/gpu_optimized.yaml

# Or run individual stages
python main.py --config configs/gpu_optimized.yaml --stage vae_pretrain
python main.py --config configs/gpu_optimized.yaml --stage gnn_train

# Resume from latest checkpoint
python main.py --config configs/gpu_optimized.yaml --resume-from-latest

# Multi-GPU training (H100 node with 8 GPUs)
torchrun --nproc_per_node=8 main.py --config configs/gpu_optimized.yaml --stage vae_pretrain
```

### Start the API server

```bash
python main.py --config configs/gpu_optimized.yaml --stage serve --port 8001
```

### Run tests

```bash
# Unit tests (21 tests, no external data needed)
pytest tests/ -v --tb=short

# API integration test
python scripts/test_api.py --url http://localhost:8001 --batch
```

## API Reference

Once the server is running, interactive docs are available at `http://localhost:8001/docs`.

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check (status, model_loaded, device) |
| `GET` | `/config` | Current model configuration |
| `POST` | `/predict` | Single sample prediction |
| `POST` | `/predict/batch` | Batch prediction (up to 64 samples) |

### Example request

```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "protein_abundances": {
      "EZH2": 4.5, "SUZ12": 3.8, "EED": 3.5,
      "DNMT1": 4.0, "DNMT3A": 3.2, "DNMT3B": 2.5,
      "TET1": 0.2, "TET2": 0.3, "TET3": 0.1,
      "KDM6A": 0.3, "KDM6B": 0.2,
      "HDAC1": 3.5, "HDAC2": 3.0, "HDAC3": 2.8,
      "EP300": 0.3, "CREBBP": 0.4,
      "UHRF1": 3.5, "SMARCB1": 0.5, "SMARCA4": 0.4
    }
  }'
```

### Example response

```json
{
  "stability_score": 0.9966,
  "memory_state": [0.12, -0.05, ...],
  "drug_predictions": [
    {
      "drug_name": "Bortezomib",
      "predicted_ic50": -0.085,
      "resistance_probability": 0.479,
      "reversibility_probability": 0.008
    },
    {
      "drug_name": "Lenalidomide",
      "predicted_ic50": 35.23,
      "resistance_probability": 1.0,
      "reversibility_probability": 0.009
    }
  ],
  "interpretation": "HIGH memory stability — this epigenetic state appears deeply locked in. Drug resistance driven by this state is likely IRREVERSIBLE through drug holidays alone. Predicted resistance to: Lenalidomide."
}
```

## Project Structure

```
myelomemory/
├── main.py                          # Pipeline orchestration (8 stages)
├── pyproject.toml                   # Package metadata and dependencies
├── requirements.txt                 # Pinned dependencies
├── configs/
│   ├── default.yaml                 # CPU/development config
│   ├── gpu_optimized.yaml           # H100-optimized (bf16, pin_memory)
│   └── h100.yaml                    # Full H100 config (compile, large batch)
├── myelomemory/
│   ├── config.py                    # Configuration dataclasses
│   ├── data/
│   │   ├── loaders.py               # CCLE, GDSC, STRING data loaders
│   │   ├── preprocessors.py         # Normalization, imputation, harmonization
│   │   └── graph_builder.py         # STRING PPI network → PyG graph
│   ├── models/
│   │   ├── vae.py                   # Module 1: Proteome-to-Epigenome cVAE
│   │   ├── stability.py             # Module 2: ODE Memory Stability Scorer
│   │   └── gnn.py                   # Module 3: Resistance Pathway GNN
│   ├── inference/
│   │   ├── pipeline.py              # End-to-end inference (all 3 modules)
│   │   └── api.py                   # FastAPI serving endpoint
│   └── utils/
│       ├── checkpoint.py            # Checkpoint save/load/resume
│       ├── logging.py               # Structured logging + W&B integration
│       └── metrics.py               # AUROC, AUPRC, reconstruction error
├── scripts/
│   ├── download_data.py             # Automated dataset download
│   ├── download_data.sh             # Manual download guide
│   ├── setup_and_run.sh             # One-command setup + full pipeline
│   └── test_api.py                  # API test harness (5 clinical profiles)
├── tests/
│   ├── test_vae.py                  # VAE shape contracts + connectivity
│   ├── test_stability.py            # ODE solver + scoring tests
│   ├── test_gnn.py                  # GNN forward pass + loss tests
│   └── test_pipeline.py             # End-to-end integration test
└── checkpoints/                     # Git-ignored; stored on shared filesystem
```

## Checkpoints

The pipeline saves checkpoints after every major stage. Each contains model weights, optimizer state, epoch, metrics, config, timestamp, and git hash.

| Stage | Checkpoint | Size | Contents |
|-------|-----------|------|----------|
| Data preprocessing | `data_ready.pt` | 25M | Preprocessed tensors, splits, normalization params |
| VAE pretraining | `vae_pretrained.pt` | 247M | Pan-cancer CCLE weights |
| VAE fine-tuning | `vae_finetuned.pt` | 247M | Hematological subset weights |
| Stability calibration | `stability_calibrated.pt` | 21K | ODE parameters, Jacobian normalization |
| GNN training | `gnn_trained.pt` | 1.6M | Best validation loss weights |
| Pipeline validation | `pipeline_validated.pt` | 4.7K | End-to-end metrics |

## Hardware Requirements

**Recommended:** NVIDIA H100 (80GB HBM3) or equivalent

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | Any CUDA GPU (8GB+) | H100 80GB |
| RAM | 16GB | 64GB |
| Storage | 5GB (code + checkpoints) | 50GB (with raw data) |
| Python | 3.10+ | 3.12 |

**GPU optimizations** (enabled in `gpu_optimized.yaml`):
- bf16 mixed precision (native H100 support)
- Gradient checkpointing for VAE encoder
- Pin memory on all DataLoaders
- 8 DataLoader workers with prefetch_factor=4

## Configuration

Two primary configs are provided:

| Config | Device | Batch Size | dtype | Use Case |
|--------|--------|-----------|-------|----------|
| `default.yaml` | CPU | 32-64 | float32 | Development, testing |
| `gpu_optimized.yaml` | CUDA | 32-512 | bfloat16 | Production training on H100 |

All hyperparameters are defined as dataclasses in `myelomemory/config.py` and loaded from YAML. The pipeline auto-overrides dimensions from the actual data at runtime.

## Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| PyTorch | >= 2.2.0 | Core ML framework |
| torchdiffeq | >= 0.2.3 | GPU-accelerated ODE solving |
| torch-geometric | >= 2.5.0 | Graph neural networks |
| FastAPI | >= 0.110.0 | API server |
| pandas | >= 2.0.0 | Data loading |
| scikit-learn | >= 1.3.0 | Preprocessing |
| wandb | >= 0.16.0 | Experiment tracking |

## License

MIT
