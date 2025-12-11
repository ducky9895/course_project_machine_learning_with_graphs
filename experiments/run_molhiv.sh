#!/usr/bin/env bash
# Graphormer Paper Replication: OGBG-MolHIV Experiment
# Expected result: Test AUC ~80.56% (Graphormer-v2 with pre-training)

set -e

# Configuration
EXPERIMENT_NAME="molhiv_graphormer_base"
SAVE_DIR="./results/${EXPERIMENT_NAME}"
LOG_DIR="./logs/${EXPERIMENT_NAME}"
CUDA_DEVICE=${CUDA_DEVICE:-0}
PRETRAINED_MODEL=${PRETRAINED_MODEL:-"pcqm4mv1_graphormer_base_for_molhiv"}

# Training hyperparameters
N_GPU=1
EPOCH=4
MAX_EPOCH=$((EPOCH + 1))
BATCH_SIZE=128
TOT_UPDATES=$((33000*EPOCH/BATCH_SIZE/N_GPU))
WARMUP_UPDATES=$((TOT_UPDATES*16/100))

# Get absolute path to Graphormer
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRAPHORMER_DIR="${SCRIPT_DIR}/../Graphormer"
EXAMPLES_DIR="${GRAPHORMER_DIR}/examples/property_prediction"

# Create directories (using absolute paths)
mkdir -p ${SAVE_DIR}
mkdir -p ${LOG_DIR}

# Check if fairseq is installed
if ! command -v fairseq-train &> /dev/null; then
    echo "Error: fairseq-train not found in PATH"
    echo "Please install Graphormer first:"
    echo "  cd ${GRAPHORMER_DIR} && bash install.sh"
    echo ""
    echo "Or use: python -m fairseq_cli.train"
    echo "Attempting to use python -m fairseq_cli.train instead..."
    FAIRSEQ_CMD="python -m fairseq_cli.train"
else
    FAIRSEQ_CMD="fairseq-train"
fi

echo "=========================================="
echo "Running OGBG-MolHIV Experiment"
echo "=========================================="
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Save directory: ${SAVE_DIR}"
echo "Log directory: ${LOG_DIR}"
echo "CUDA device: ${CUDA_DEVICE}"
echo "Pre-trained model: ${PRETRAINED_MODEL}"
echo "Epochs: ${EPOCH}"
echo "Batch size: ${BATCH_SIZE}"
echo "Total updates: ${TOT_UPDATES}"
echo "Warmup updates: ${WARMUP_UPDATES}"
echo "Fairseq command: ${FAIRSEQ_CMD}"
echo "=========================================="

# Change to Graphormer examples directory
cd ${EXAMPLES_DIR}

# Run training
CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} ${FAIRSEQ_CMD} \
--user-dir ../../graphormer \
--num-workers 16 \
--ddp-backend=legacy_ddp \
--dataset-name ogbg-molhiv \
--dataset-source ogb \
--task graph_prediction_with_flag \
--criterion binary_logloss_with_flag \
--arch graphormer_base \
--num-classes 1 \
--attention-dropout 0.1 --act-dropout 0.1 --dropout 0.0 \
--optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 --clip-norm 5.0 --weight-decay 0.0 \
--lr-scheduler polynomial_decay --power 1 --warmup-updates ${WARMUP_UPDATES} --total-num-update ${TOT_UPDATES} \
--lr 2e-4 --end-learning-rate 1e-5 \
--batch-size ${BATCH_SIZE} \
--fp16 \
--data-buffer-size 20 \
--encoder-layers 12 \
--encoder-embed-dim 768 \
--encoder-ffn-embed-dim 768 \
--encoder-attention-heads 32 \
--max-epoch ${MAX_EPOCH} \
--save-dir ${SAVE_DIR} \
--pretrained-model-name ${PRETRAINED_MODEL} \
--seed 1 \
--flag-m 3 \
--flag-step-size 0.01 \
--flag-mag 0 \
--pre-layernorm \
2>&1 | tee ${LOG_DIR}/training.log

echo "=========================================="
echo "OGBG-MolHIV experiment completed!"
echo "Results saved to: ${SAVE_DIR}"
echo "Logs saved to: ${LOG_DIR}/training.log"
echo "=========================================="

