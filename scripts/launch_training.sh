#!/usr/bin/env bash
# launch_training.sh — Launch MyeloMemory training on H100 GPU(s)
#
# Usage:
#   # Single H100
#   bash scripts/launch_training.sh
#
#   # Multi-GPU (8x H100 node)
#   bash scripts/launch_training.sh --multi-gpu
#
#   # SLURM submission
#   bash scripts/launch_training.sh --slurm
#
#   # Specific stage only
#   bash scripts/launch_training.sh --stage vae_pretrain

set -euo pipefail

MODE="single"
STAGE=""
CONFIG="configs/h100.yaml"
NGPUS=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --multi-gpu) MODE="multi"; NGPUS="${2:-8}"; shift ;;
        --slurm) MODE="slurm" ;;
        --stage) STAGE="$2"; shift ;;
        --config) CONFIG="$2"; shift ;;
        --ngpus) NGPUS="$2"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

# Environment setup
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TORCH_CUDNN_V8_API_ENABLED=1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0

echo "=== MyeloMemory Training Launch ==="
echo "Mode: $MODE"
echo "Config: $CONFIG"
echo "GPUs: $NGPUS"
[ -n "$STAGE" ] && echo "Stage: $STAGE"
echo ""

STAGE_ARG=""
[ -n "$STAGE" ] && STAGE_ARG="--stage $STAGE"

case $MODE in
    single)
        echo "Launching single-GPU training..."
        python main.py --config "$CONFIG" $STAGE_ARG
        ;;

    multi)
        echo "Launching multi-GPU training with $NGPUS GPUs..."
        torchrun \
            --nproc_per_node="$NGPUS" \
            --master_port=29500 \
            main.py --config "$CONFIG" $STAGE_ARG
        ;;

    slurm)
        echo "Submitting SLURM job..."
        sbatch <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=myelomemory
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=$NGPUS
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

module load cuda/12.4
module load python/3.11

source venv/bin/activate

torchrun \\
    --nproc_per_node=$NGPUS \\
    --master_port=29500 \\
    main.py --config $CONFIG $STAGE_ARG
SBATCH_EOF
        echo "Job submitted. Check logs/ for output."
        ;;
esac
