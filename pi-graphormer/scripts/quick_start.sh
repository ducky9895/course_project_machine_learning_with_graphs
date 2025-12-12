#!/bin/bash
# Quick start script for running a minimal experiment

set -e

echo "=========================================="
echo "Graphormer-PIGNN v2: Quick Start"
echo "=========================================="

# Configuration
DATA_DIR="data"
CHECKPOINT_DIR="chkpts/quick_start"
SEED=42

# Create directories
mkdir -p "${DATA_DIR}" "${CHECKPOINT_DIR}"

# Step 1: Generate small synthetic dataset
echo ""
echo "Step 1: Generating synthetic dataset..."
python generate_ptmotifs.py \
    --output_dir "${DATA_DIR}/PT-Motifs/raw" \
    --n_train 1000 \
    --n_val 200 \
    --n_test 200 \
    --seed ${SEED} \
    --motifs "cycle,house,star"

# Step 2: Train model
echo ""
echo "Step 2: Training model..."
python main/train_v2.py \
    --dataset synthetic \
    --n_train 1000 \
    --n_val 200 \
    --n_test 200 \
    --epochs 20 \
    --batch_size 16 \
    --num_layers 2 \
    --embedding_dim 64 \
    --num_heads 2 \
    --lr 1e-4 \
    --use_regularization \
    --reg_weight 0.01 \
    --seed ${SEED} \
    --save_dir "${CHECKPOINT_DIR}" \
    --cuda 0 \
    --log_interval 10

echo ""
echo "=========================================="
echo "Quick start experiment completed!"
echo "Checkpoint: ${CHECKPOINT_DIR}"
echo "=========================================="
