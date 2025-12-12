#!/bin/bash
# Experiment 3: Motif Detection

set -e

MODEL_PATH=${1:-$(find chkpts -name "best_model.pt" | head -1)}
N_TEST=${2:-500}
OUTPUT=${3:-results/motif_detection.csv}

if [ -z "${MODEL_PATH}" ] || [ ! -f "${MODEL_PATH}" ]; then
    echo "Error: Model not found. Please provide model path:"
    echo "  bash scripts/eval_motif.sh <path_to_best_model.pt>"
    echo ""
    echo "Or train a model first:"
    echo "  bash scripts/train_explainer.sh"
    exit 1
fi

echo "=========================================="
echo "Experiment 3: Motif Detection"
echo "Model: ${MODEL_PATH}"
echo "=========================================="

mkdir -p $(dirname ${OUTPUT})

python scripts/eval_motif_detection.py \
    --model_path ${MODEL_PATH} \
    --dataset synthetic \
    --n_test ${N_TEST} \
    --output ${OUTPUT}

echo ""
echo "✓ Motif detection complete!"
echo "Results saved to: ${OUTPUT}"
