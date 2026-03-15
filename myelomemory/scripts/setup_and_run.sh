#!/usr/bin/env bash
# setup_and_run.sh — One-command deployment for MyeloMemory on UNT HPC
#
# This script:
#   1. Creates a conda environment with all dependencies
#   2. Downloads all required public datasets (real data only)
#   3. Runs the full pipeline end-to-end
#
# Usage:
#   cd /home/aj0486@students.ad.unt.edu/pipeline3/myelomemory
#   bash scripts/setup_and_run.sh
#
# Prerequisites:
#   - CUDA 11.8+ (H100 compatible)
#   - conda or miniconda installed
#   - Internet access (no proxy blocking public repos)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "================================================================"
echo "  MyeloMemory — Full Setup & Execution"
echo "  Project root: $PROJECT_DIR"
echo "================================================================"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Environment setup
# ---------------------------------------------------------------------------
echo "--- [Step 1/4] Environment Setup ---"
echo ""

ENV_NAME="myelomemory"

if command -v conda &> /dev/null; then
    if conda env list | grep -q "^${ENV_NAME} "; then
        echo "  Conda environment '$ENV_NAME' already exists. Activating..."
    else
        echo "  Creating conda environment '$ENV_NAME' with Python 3.10..."
        conda create -n "$ENV_NAME" python=3.10 -y
    fi

    # Activate — works in both bash and zsh
    eval "$(conda shell.bash hook 2>/dev/null)"
    conda activate "$ENV_NAME"
    echo "  Using Python: $(which python) ($(python --version))"
else
    echo "  No conda found. Using system Python."
    echo "  If dependencies fail, install conda first:"
    echo "    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "    bash Miniconda3-latest-Linux-x86_64.sh"
fi

echo ""
echo "  Installing Python dependencies..."
pip install --upgrade pip -q

# PyTorch with CUDA 11.8 (compatible with H100)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -q 2>&1 | tail -1

# PyTorch Geometric
pip install torch-geometric -q 2>&1 | tail -1

# ODE solver, ML, API, data
pip install torchdiffeq scikit-learn scipy pandas openpyxl requests -q 2>&1 | tail -1
pip install fastapi uvicorn pydantic wandb -q 2>&1 | tail -1

# Install the project itself
pip install -e . -q 2>&1 | tail -1

echo "  Dependencies installed."
echo ""

# Quick sanity check
python -c "
import torch
print(f'  PyTorch {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    props = torch.cuda.get_device_properties(0)
    mem = getattr(props, 'total_memory', None) or getattr(props, 'total_mem', 0)
    print(f'  GPU memory: {mem / 1e9:.1f} GB')
from myelomemory.config import MyeloMemoryConfig
print('  myelomemory package: OK')
"

# ---------------------------------------------------------------------------
# Step 2: Download real data
# ---------------------------------------------------------------------------
echo ""
echo "--- [Step 2/4] Downloading Real Datasets ---"
echo ""
echo "  All datasets are from public repositories."
echo "  No synthetic data — no fallbacks — real data only."
echo ""

python scripts/download_data.py --data-dir data/raw

# ---------------------------------------------------------------------------
# Step 3: Validate data files
# ---------------------------------------------------------------------------
echo ""
echo "--- [Step 3/4] Validating Data Files ---"
echo ""

python -c "
from pathlib import Path
import sys

data_dir = Path('data/raw')
required = {
    'ccle_proteomics.csv': 'CCLE Proteomics (DepMap)',
    'sample_info.csv': 'Cell Line Metadata (DepMap)',
    'string_ppi.txt': 'STRING PPI Network',
    'gdsc_drug_sensitivity.csv': 'GDSC Drug Sensitivity',
}

# At least one epigenomic file
epigenomic = [
    'ccle_epigenomics/chromatin_profiling.csv',
    'ccle_epigenomics/atac_seq.csv',
]

all_ok = True
for fname, label in required.items():
    path = data_dir / fname
    if path.exists() and path.stat().st_size > 0:
        mb = path.stat().st_size / 1e6
        print(f'  [OK]      {label} ({mb:.1f} MB)')
    else:
        print(f'  [MISSING] {label} — {path}')
        all_ok = False

epi_ok = False
for fname in epigenomic:
    path = data_dir / fname
    if path.exists() and path.stat().st_size > 0:
        mb = path.stat().st_size / 1e6
        print(f'  [OK]      Epigenomics: {fname} ({mb:.1f} MB)')
        epi_ok = True
        break

if not epi_ok:
    print(f'  [MISSING] Epigenomics — need at least one of: {epigenomic}')
    all_ok = False

if not all_ok:
    print()
    print('  Some files are missing. See download_data.py output above.')
    print('  You may need to register at DepMap/GDSC for some files.')
    print('  Run: bash scripts/download_data.sh  for manual instructions.')
    sys.exit(1)

print()
print('  All required data files present.')
"

# ---------------------------------------------------------------------------
# Step 4: Run the full pipeline
# ---------------------------------------------------------------------------
echo ""
echo "--- [Step 4/4] Running Full Pipeline ---"
echo ""

# Detect GPU and pick config
if python -c "import torch; assert torch.cuda.is_available() and 'H100' in torch.cuda.get_device_name(0)" 2>/dev/null; then
    CONFIG="configs/h100.yaml"
    echo "  H100 detected — using $CONFIG"
elif python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    CONFIG="configs/default.yaml"
    echo "  GPU detected (non-H100) — using $CONFIG"
else
    CONFIG="configs/default.yaml"
    echo "  No GPU — using CPU config: $CONFIG"
fi

echo ""
echo "  Starting pipeline..."
echo "  Checkpoints will be saved to: checkpoints/"
echo "  Logs will be saved to: logs/"
echo ""

python main.py --config "$CONFIG"

echo ""
echo "================================================================"
echo "  MyeloMemory pipeline complete."
echo "  Checkpoints: $(ls checkpoints/*.pt 2>/dev/null | wc -l) saved"
echo "================================================================"