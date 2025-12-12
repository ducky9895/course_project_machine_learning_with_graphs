#!/bin/bash
# Experiment 4: Qualitative Visualization

set -e

MODEL_PATH=${1:-$(find chkpts -name "best_model.pt" | head -1)}
NUM_SAMPLES=${2:-5}
N_TEST=${3:-50}
OUTPUT_DIR=${4:-results/visualizations}

if [ -z "${MODEL_PATH}" ] || [ ! -f "${MODEL_PATH}" ]; then
    echo "Error: Model not found. Please provide model path:"
    echo "  bash scripts/eval_visualize.sh <path_to_best_model.pt>"
    echo ""
    echo "Or train a model first:"
    echo "  bash scripts/train_explainer.sh"
    exit 1
fi

echo "=========================================="
echo "Experiment 4: Qualitative Visualization"
echo "Model: ${MODEL_PATH}"
echo "=========================================="

mkdir -p ${OUTPUT_DIR}

python scripts/visualize_explanations.py \
    --model_path ${MODEL_PATH} \
    --num_samples ${NUM_SAMPLES} \
    --n_test ${N_TEST} \
    --output_dir ${OUTPUT_DIR}

echo ""
echo "✓ Visualization complete!"
echo "Images saved to: ${OUTPUT_DIR}/"
