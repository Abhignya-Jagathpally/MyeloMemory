# CLAUDE.md — MyeloMemory Development Guide

## Project Overview

MyeloMemory is an AI pipeline that infers epigenetic drug resistance memory states
in multiple myeloma by integrating proteomic profiles with epigenomic signatures.
It predicts not just WHETHER a patient will resist treatment, but HOW STABLE that
resistance is — distinguishing reversible adaptations from locked-in epigenetic memory.

## Architecture

```
myelomemory/
├── main.py                          # Entry point — clean pipeline orchestration
├── CLAUDE.md                        # This file
├── pyproject.toml                   # Package metadata and dependencies
├── requirements.txt                 # Pinned dependencies
├── configs/
│   ├── default.yaml                 # Default hyperparameters
│   └── h100.yaml                    # H100-optimized settings (bf16, flash attention)
├── myelomemory/
│   ├── __init__.py
│   ├── config.py                    # Configuration dataclasses
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loaders.py               # Dataset classes for CCLE, PRIDE, ENCODE
│   │   ├── preprocessors.py         # Normalization, imputation, harmonization
│   │   └── graph_builder.py         # STRING PPI network → PyG graph
│   ├── models/
│   │   ├── __init__.py
│   │   ├── vae.py                   # Module 1: Proteome-to-Epigenome cVAE
│   │   ├── stability.py             # Module 2: Memory Stability Scorer (ODE)
│   │   └── gnn.py                   # Module 3: Resistance Pathway GNN
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── pipeline.py              # Full inference pipeline (all 3 modules)
│   │   └── api.py                   # FastAPI serving endpoint
│   └── utils/
│       ├── __init__.py
│       ├── checkpoint.py            # Save/load/resume checkpoints
│       ├── logging.py               # Structured logging with W&B integration
│       └── metrics.py               # AUROC, AUPRC, reconstruction error, etc.
├── scripts/
│   ├── download_data.py             # Automated download of all public datasets
│   ├── download_data.sh             # Manual download guide with validation
│   └── setup_and_run.sh             # One-command: env + data + full pipeline
├── tests/
│   ├── test_vae.py
│   ├── test_stability.py
│   ├── test_gnn.py
│   └── test_pipeline.py
└── checkpoints/                     # Git-ignored; stored on shared filesystem
```

## Hardware Target: NVIDIA H100 (80 GB HBM3)

All training code is written for H100 GPUs. Key optimizations:

- **bf16 mixed precision** via `torch.amp.autocast("cuda", dtype=torch.bfloat16)`
- **torch.compile()** on all model forward passes (inductor backend)
- **Flash Attention v2** for any attention layers (via `F.scaled_dot_product_attention`)
- **Gradient checkpointing** in the VAE encoder for large batch sizes
- **tf32 enabled** for matmuls: `torch.backends.cuda.matmul.allow_tf32 = True`
- **Pin memory** on all DataLoaders for faster host-to-device transfer
- **Batch sizes**: VAE=512, GNN=256 (fits in 80 GB with bf16)

For multi-GPU: use `torchrun --nproc_per_node=N` with DDP. The code uses
`torch.nn.parallel.DistributedDataParallel` when `WORLD_SIZE > 1`.

## Checkpoints — MANDATORY after every major step

The pipeline saves checkpoints after every critical stage. Each checkpoint
contains the model state, optimizer state, epoch, global step, metrics,
and the full config used to produce it.

### Checkpoint schedule

| Stage | Checkpoint Name | Contents |
|-------|----------------|----------|
| After data preprocessing | `data_ready.pt` | Preprocessed tensors, split indices, normalization params |
| After VAE pretraining (pan-cancer) | `vae_pretrained.pt` | VAE weights from pan-cancer CCLE training |
| After VAE fine-tuning (hematological) | `vae_finetuned.pt` | VAE weights fine-tuned on hematological subset |
| After stability model calibration | `stability_calibrated.pt` | ODE parameters, basin depth estimates |
| After GNN training | `gnn_trained.pt` | GNN weights, best validation AUROC |
| After full pipeline validation | `pipeline_validated.pt` | All 3 module weights + end-to-end metrics |

### Checkpoint format

```python
{
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "epoch": epoch,
    "global_step": global_step,
    "best_metric": best_metric,
    "config": dataclasses.asdict(config),
    "timestamp": datetime.utcnow().isoformat(),
    "git_hash": subprocess.check_output(["git", "rev-parse", "HEAD"]).strip(),
}
```

### Resuming from checkpoints

Every training function accepts `--resume <checkpoint_path>`. The main.py
pipeline automatically detects existing checkpoints and skips completed stages.

```bash
# Resume full pipeline from wherever it left off
python main.py --config configs/h100.yaml --resume-from-latest

# Resume a specific stage
python main.py --config configs/h100.yaml --stage vae_finetune --resume checkpoints/vae_pretrained.pt
```

## Workflow: Subagents and Worktrees

### Why subagents and worktrees?

MyeloMemory has three independent model modules that can be developed, trained,
and tested in parallel. Using Claude Code subagents with git worktrees enables:

1. **Parallel development** — Module 1 (VAE), Module 2 (ODE), and Module 3 (GNN)
   can be built and tested simultaneously in isolated worktrees
2. **Safe experimentation** — Each worktree is a full repo copy; failed experiments
   don't pollute the main branch
3. **Checkpoint isolation** — Each worktree trains independently; only validated
   checkpoints are merged back

### Worktree strategy

```
main branch (myelomemory/)
├── worktree: feature/vae-module        → Subagent 1 builds & trains VAE
├── worktree: feature/stability-module  → Subagent 2 builds & calibrates ODE
├── worktree: feature/gnn-module        → Subagent 3 builds & trains GNN
└── worktree: feature/integration       → Subagent 4 wires modules together
```

### Subagent task definitions

When using Claude Code to develop this project, launch subagents as follows:

**Subagent 1 — VAE Development (isolation: worktree)**
```
Task: Implement and train the conditional VAE in myelomemory/models/vae.py.
- Build encoder (proteomic input → latent space)
- Build decoder (latent space → reconstructed epigenomic profiles)
- Implement KL annealing schedule
- Train on CCLE pan-cancer, then fine-tune on hematological subset
- Save checkpoints: vae_pretrained.pt, vae_finetuned.pt
- Run tests/test_vae.py before merging
```

**Subagent 2 — Stability Scorer (isolation: worktree)**
```
Task: Implement the ODE-based memory stability scorer in myelomemory/models/stability.py.
- Implement Sneppen-Ringrose bistability ODE system
- Parameterize with chromatin reader/writer protein levels
- Compute basin-of-attraction depth as stability score
- Calibrate on CCLE washout time-course data
- Save checkpoint: stability_calibrated.pt
- Run tests/test_stability.py before merging
```

**Subagent 3 — GNN Classifier (isolation: worktree)**
```
Task: Implement the drug resistance GNN in myelomemory/models/gnn.py.
- Build STRING PPI graph (myelomemory/data/graph_builder.py)
- Implement GNN with memory-state node features
- Train on GDSC/CTRPv2 drug sensitivity data
- Evaluate with/without memory features (ablation study)
- Save checkpoint: gnn_trained.pt
- Run tests/test_gnn.py before merging
```

**Subagent 4 — Integration (after 1-3 complete)**
```
Task: Wire all three modules into the full pipeline.
- Load all three checkpoints
- Implement end-to-end inference in myelomemory/inference/pipeline.py
- Validate on held-out Krönke et al. 2024 data
- Save checkpoint: pipeline_validated.pt
- Run tests/test_pipeline.py
- Build FastAPI endpoint in myelomemory/inference/api.py
```

### Merge protocol

1. Each subagent runs its test suite in the worktree
2. If tests pass, create a PR from the worktree branch to main
3. Review diffs — ensure no config conflicts
4. Merge with `--no-ff` to preserve branch history
5. After all modules merged, Subagent 4 runs integration tests on main

## Quick Start — End-to-End on UNT HPC

```bash
# Copy the project to your UNT home directory
cp -r myelomemory/ /home/aj0486@students.ad.unt.edu/pipeline3/myelomemory/
cd /home/aj0486@students.ad.unt.edu/pipeline3/myelomemory/

# One command does everything: env setup → data download → full pipeline
bash scripts/setup_and_run.sh
```

The setup script will:
1. Create a conda environment with all dependencies (PyTorch CUDA 11.8, torch-geometric, torchdiffeq)
2. Download all 6 real datasets from DepMap, STRING, GDSC using `scripts/download_data.py`
3. Auto-detect your GPU and select the right config (h100.yaml or default.yaml)
4. Run the full pipeline through all 8 stages with automatic checkpointing

If any download fails (e.g. DepMap requires free registration), the script tells you
exactly which file to download manually and where to save it. After placing the file,
just re-run `bash scripts/setup_and_run.sh` — it skips completed downloads and stages.

## Commands Reference

```bash
# Full pipeline (auto-detects checkpoints, skips completed stages)
python main.py --config configs/h100.yaml

# Individual stages
python main.py --config configs/h100.yaml --stage data_prep
python main.py --config configs/h100.yaml --stage vae_pretrain
python main.py --config configs/h100.yaml --stage vae_finetune
python main.py --config configs/h100.yaml --stage stability_calibrate
python main.py --config configs/h100.yaml --stage gnn_train
python main.py --config configs/h100.yaml --stage validate
python main.py --config configs/h100.yaml --stage serve

# Download data only (no training)
python scripts/download_data.py --data-dir data/raw

# Multi-GPU training (H100 node with 8 GPUs)
torchrun --nproc_per_node=8 main.py --config configs/h100.yaml --stage vae_pretrain

# Run all tests (21 unit tests)
pytest tests/ -v --tb=short

# Launch API server
python main.py --config configs/h100.yaml --stage serve --port 8000
```

## Code Style

- Type hints on all function signatures
- Docstrings (Google style) on all public functions
- No wildcard imports
- `dataclasses` for all configs (no raw dicts)
- `logging` module (not print statements)
- All tensor operations must specify device explicitly
- Use `pathlib.Path` not string concatenation for file paths

## Testing

- Every model module has a corresponding test file
- Tests use small tensors in the same format as real data (no external downloads)
- Tests verify shape contracts and module connectivity, NOT scientific validity
- The pipeline itself REQUIRES real data — no synthetic fallbacks exist
- `pytest` with `--tb=short` for CI
- Integration test (`test_pipeline.py`) runs a tiny end-to-end pass on unit-test-scale tensors

## Git Conventions

- Branch naming: `feature/<module-name>`, `fix/<issue>`, `experiment/<description>`
- Commit messages: imperative mood, max 72 chars first line
- Tag checkpoints: `v0.1-vae-pretrained`, `v0.2-stability-calibrated`, etc.
- Never commit checkpoint files (`.pt`) — store on shared filesystem, reference by path
