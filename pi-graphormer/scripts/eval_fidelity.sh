#!/bin/bash
# Experiment 2: Fidelity Curves

set -e

MODEL_PATH=${1:-$(find chkpts -name "best_model.pt" | head -1)}
N_TEST=${2:-200}
OUTPUT_DIR=${3:-results/fidelity_curves}

if [ -z "${MODEL_PATH}" ] || [ ! -f "${MODEL_PATH}" ]; then
    echo "Error: Model not found. Please provide model path:"
    echo "  bash scripts/eval_fidelity.sh <path_to_best_model.pt>"
    echo ""
    echo "Or train a model first:"
    echo "  bash scripts/train_explainer.sh"
    exit 1
fi

echo "=========================================="
echo "Experiment 2: Fidelity Curves"
echo "Model: ${MODEL_PATH}"
echo "=========================================="

mkdir -p ${OUTPUT_DIR}

python scripts/eval_fidelity_curves.py \
    --model_path ${MODEL_PATH} \
    --dataset synthetic \
    --n_test ${N_TEST} \
    --output_dir ${OUTPUT_DIR}

echo ""
echo "✓ Fidelity curves complete!"
echo "Results saved to: ${OUTPUT_DIR}/"
