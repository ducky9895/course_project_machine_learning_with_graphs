#!/bin/bash
# Experiment 2: Fidelity Curves

set -e

MODEL_PATH_ARG=${1:-""}
N_TEST=${2:-200}
OUTPUT_DIR=${3:-results/fidelity_curves}

# Auto-detect model if not provided or if path contains "..."
if [ -z "${MODEL_PATH_ARG}" ] || [[ "${MODEL_PATH_ARG}" == *"..."* ]]; then
    # Try to find graphormer_explainer models first
    MODEL_PATH=$(find chkpts -path "*graphormer_explainer*" -name "best_model.pt" -type f 2>/dev/null | sort -r | head -1)
    
    if [ -z "${MODEL_PATH}" ]; then
        # Fallback to any best_model.pt
        MODEL_PATH=$(find chkpts -name "best_model.pt" -type f 2>/dev/null | sort -r | head -1)
    fi
    
    if [ -n "${MODEL_PATH_ARG}" ] && [[ "${MODEL_PATH_ARG}" == *"..."* ]]; then
        # User provided a pattern, try to match it
        PATTERN=$(echo "${MODEL_PATH_ARG}" | sed 's/\.\.\./.*/g')
        MATCHED=$(find chkpts -path "${PATTERN}" -name "best_model.pt" -type f 2>/dev/null | sort -r | head -1)
        if [ -n "${MATCHED}" ]; then
            MODEL_PATH="${MATCHED}"
        fi
    fi
else
    MODEL_PATH="${MODEL_PATH_ARG}"
fi

if [ -z "${MODEL_PATH}" ] || [ ! -f "${MODEL_PATH}" ]; then
    echo "Error: Model not found."
    echo ""
    echo "Available models:"
    find chkpts -name "best_model.pt" -type f 2>/dev/null | head -5
    echo ""
    echo "Usage:"
    echo "  bash scripts/eval_fidelity.sh <path_to_best_model.pt>"
    echo "  bash scripts/eval_fidelity.sh chkpts/graphormer_explainer_reg_0.01/.../best_model.pt  (auto-detect)"
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
