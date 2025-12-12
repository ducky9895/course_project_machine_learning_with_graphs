#!/bin/bash
# Experiment 1: Predictive Accuracy Comparison

set -e

CHECKPOINT_DIR=${1:-chkpts}
N_TEST=${2:-500}
OUTPUT=${3:-results/accuracy_comparison.csv}

echo "=========================================="
echo "Experiment 1: Predictive Accuracy"
echo "=========================================="

mkdir -p $(dirname ${OUTPUT})

python scripts/eval_predictive_accuracy.py \
    --checkpoint_dir ${CHECKPOINT_DIR} \
    --dataset synthetic \
    --n_test ${N_TEST} \
    --output ${OUTPUT}

echo ""
echo "✓ Accuracy evaluation complete!"
echo "Results saved to: ${OUTPUT}"
