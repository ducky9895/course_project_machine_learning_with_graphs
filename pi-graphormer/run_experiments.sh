#!/bin/bash
# Master script to run all experiments for Graphormer-PIGNN v2

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
BASE_DIR=$(pwd)
DATA_DIR="${BASE_DIR}/data"
CHECKPOINT_DIR="${BASE_DIR}/chkpts"
LOG_DIR="${BASE_DIR}/logs"
SEED=42

# Create directories
mkdir -p "${DATA_DIR}" "${CHECKPOINT_DIR}" "${LOG_DIR}"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Graphormer-PIGNN v2 Experiments${NC}"
echo -e "${GREEN}========================================${NC}"

# Function to run experiment
run_experiment() {
    local exp_name=$1
    local script=$2
    local log_file="${LOG_DIR}/${exp_name}.log"
    
    echo -e "${YELLOW}Running: ${exp_name}${NC}"
    echo "Log: ${log_file}"
    
    bash "${script}" > "${log_file}" 2>&1
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ ${exp_name} completed${NC}"
    else
        echo -e "${RED}✗ ${exp_name} failed${NC}"
        echo "Check log: ${log_file}"
    fi
}

# Phase 1: Pre-training on Synthetic Data
echo -e "\n${GREEN}=== Phase 1: Pre-training ===${NC}"

# Generate synthetic dataset if needed
if [ ! -f "${DATA_DIR}/PT-Motifs/raw/train.npy" ]; then
    echo "Generating PT-Motifs dataset..."
    python generate_ptmotifs.py \
        --output_dir "${DATA_DIR}/PT-Motifs/raw" \
        --n_train 5000 \
        --n_val 1000 \
        --n_test 2000 \
        --seed ${SEED}
fi

# Experiment 1.1: Baseline (no regularization)
echo -e "\n${YELLOW}Experiment 1.1: Baseline${NC}"
python main/train_v2.py \
    --dataset synthetic \
    --n_train 5000 --n_val 1000 --n_test 2000 \
    --epochs 50 \
    --batch_size 32 \
    --num_layers 4 \
    --embedding_dim 128 \
    --num_heads 4 \
    --lr 1e-4 \
    --seed ${SEED} \
    --save_dir "${CHECKPOINT_DIR}/phase1_baseline" \
    --cuda 0 \
    2>&1 | tee "${LOG_DIR}/phase1_baseline.log"

# Experiment 1.2: L1 Sparsity
echo -e "\n${YELLOW}Experiment 1.2: L1 Sparsity (λ=0.01)${NC}"
python main/train_v2.py \
    --dataset synthetic \
    --n_train 5000 --n_val 1000 --n_test 2000 \
    --epochs 50 \
    --batch_size 32 \
    --num_layers 4 \
    --embedding_dim 128 \
    --num_heads 4 \
    --lr 1e-4 \
    --use_regularization \
    --reg_weight 0.01 \
    --seed ${SEED} \
    --save_dir "${CHECKPOINT_DIR}/phase1_l1_sparsity" \
    --cuda 0 \
    2>&1 | tee "${LOG_DIR}/phase1_l1_sparsity.log"

# Experiment 1.3: Entropy Sparsity
echo -e "\n${YELLOW}Experiment 1.3: Entropy Sparsity (λ=0.01)${NC}"
# Note: Need to modify train_v2.py to support entropy method
python main/train_v2.py \
    --dataset synthetic \
    --n_train 5000 --n_val 1000 --n_test 2000 \
    --epochs 50 \
    --batch_size 32 \
    --num_layers 4 \
    --embedding_dim 128 \
    --num_heads 4 \
    --lr 1e-4 \
    --use_regularization \
    --reg_weight 0.01 \
    --seed ${SEED} \
    --save_dir "${CHECKPOINT_DIR}/phase1_entropy_sparsity" \
    --cuda 0 \
    2>&1 | tee "${LOG_DIR}/phase1_entropy_sparsity.log"

# Phase 2: Downstream Tasks
echo -e "\n${GREEN}=== Phase 2: Downstream Tasks ===${NC}"

# Find best pre-trained model
PRETRAINED_MODEL=$(find "${CHECKPOINT_DIR}/phase1_l1_sparsity" -name "best_model.pt" | head -1)

if [ -z "${PRETRAINED_MODEL}" ]; then
    echo -e "${RED}Warning: No pre-trained model found. Skipping Phase 2.${NC}"
else
    echo "Using pre-trained model: ${PRETRAINED_MODEL}"
    
    # Experiment 2.1: Fine-tuning on BA-2Motif
    echo -e "\n${YELLOW}Experiment 2.1: Fine-tuning on BA-2Motif${NC}"
    python main/train_v2.py \
        --dataset ba2motif \
        --data_dir "${DATA_DIR}/BA2Motif" \
        --pretrained_path "${PRETRAINED_MODEL}" \
        --epochs 30 \
        --batch_size 32 \
        --num_layers 4 \
        --embedding_dim 128 \
        --num_heads 4 \
        --lr 1e-4 \
        --use_regularization \
        --reg_weight 0.01 \
        --seed ${SEED} \
        --save_dir "${CHECKPOINT_DIR}/phase2_ba2motif_finetune" \
        --cuda 0 \
        2>&1 | tee "${LOG_DIR}/phase2_ba2motif_finetune.log"
fi

# Summary
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Experiments Summary${NC}"
echo -e "${GREEN}========================================${NC}"
echo "Checkpoints: ${CHECKPOINT_DIR}"
echo "Logs: ${LOG_DIR}"
echo ""
echo "To view results:"
echo "  python scripts/evaluate_results.py --checkpoint_dir ${CHECKPOINT_DIR}"
