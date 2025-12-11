#!/usr/bin/env bash
# Graphormer Paper Replication: ZINC-500K Experiment
# Expected result: Test MAE ~0.122 (Graphormer-Slim)

set -e

# Configuration
EXPERIMENT_NAME="zinc_graphormer_slim"
SAVE_DIR="./results/${EXPERIMENT_NAME}"
LOG_DIR="./logs/${EXPERIMENT_NAME}"
CUDA_DEVICE=${CUDA_DEVICE:-0}

# Create directories
mkdir -p ${SAVE_DIR}
mkdir -p ${LOG_DIR}

# Get absolute path to Graphormer
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRAPHORMER_DIR="${SCRIPT_DIR}/../Graphormer"
EXAMPLES_DIR="${GRAPHORMER_DIR}/examples/property_prediction"

echo "=========================================="
echo "Running ZINC-500K Experiment"
echo "=========================================="
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Save directory: ${SAVE_DIR}"
echo "Log directory: ${LOG_DIR}"
echo "CUDA device: ${CUDA_DEVICE}"
echo "=========================================="

# Change to Graphormer examples directory
cd ${EXAMPLES_DIR}

# Run training
CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} fairseq-train \
--user-dir ../../graphormer \
--num-workers 16 \
--ddp-backend=legacy_ddp \
--dataset-name zinc \
--dataset-source pyg \
--task graph_prediction \
--criterion l1_loss \
--arch graphormer_slim \
--num-classes 1 \
--attention-dropout 0.1 --act-dropout 0.1 --dropout 0.0 \
--optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 --clip-norm 5.0 --weight-decay 0.01 \
--lr-scheduler polynomial_decay --power 1 --warmup-updates 60000 --total-num-update 400000 \
--lr 2e-4 --end-learning-rate 1e-9 \
--batch-size 64 \
--fp16 \
--data-buffer-size 20 \
--encoder-layers 12 \
--encoder-embed-dim 80 \
--encoder-ffn-embed-dim 80 \
--encoder-attention-heads 8 \
--max-epoch 10000 \
--save-dir ${SAVE_DIR} \
2>&1 | tee ${LOG_DIR}/training.log

echo "=========================================="
echo "ZINC experiment completed!"
echo "Results saved to: ${SAVE_DIR}"
echo "Logs saved to: ${LOG_DIR}/training.log"
echo "=========================================="

