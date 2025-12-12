#!/bin/bash
# Run all evaluation experiments

set -e

echo "=========================================="
echo "Running All Evaluation Experiments"
echo "=========================================="

CHECKPOINT_DIR="chkpts"
OUTPUT_DIR="results"
MODEL_PATH="${CHECKPOINT_DIR}/graphormer_explainer/best_model.pt"

# Try to find model in alternative locations
if [ ! -f "${MODEL_PATH}" ]; then
    # Look for any best_model.pt in checkpoint directory
    ALT_MODEL=$(find ${CHECKPOINT_DIR} -name "best_model.pt" | head -1)
    if [ -n "${ALT_MODEL}" ]; then
        MODEL_PATH="${ALT_MODEL}"
        echo "Found model at: ${MODEL_PATH}"
    fi
fi

# Create output directories
mkdir -p ${OUTPUT_DIR}/{accuracy,fidelity_curves,motif_detection,visualizations,ablations}

# Experiment 1: Predictive Accuracy
echo ""
echo "=========================================="
echo "Experiment 1: Predictive Accuracy"
echo "=========================================="
python scripts/eval_predictive_accuracy.py \
    --checkpoint_dir ${CHECKPOINT_DIR} \
    --dataset synthetic \
    --n_test 500 \
    --output ${OUTPUT_DIR}/accuracy/comparison.csv \
    2>&1 | tee ${OUTPUT_DIR}/accuracy/log.txt

# Experiment 2: Fidelity Curves
echo ""
echo "=========================================="
echo "Experiment 2: Fidelity Curves"
echo "=========================================="
if [ -f "${MODEL_PATH}" ]; then
    python scripts/eval_fidelity_curves.py \
        --model_path ${MODEL_PATH} \
        --dataset synthetic \
        --n_test 200 \
        --output_dir ${OUTPUT_DIR}/fidelity_curves \
        2>&1 | tee ${OUTPUT_DIR}/fidelity_curves/log.txt
else
    echo "Warning: Model not found at ${MODEL_PATH}. Skipping fidelity curves."
    echo "Train a model first with: python main/train_v2.py --dataset synthetic"
fi

# Experiment 3: Motif Detection
echo ""
echo "=========================================="
echo "Experiment 3: Motif Detection"
echo "=========================================="
if [ -f "${MODEL_PATH}" ]; then
    python scripts/eval_motif_detection.py \
        --model_path ${MODEL_PATH} \
        --dataset synthetic \
        --n_test 500 \
        --output ${OUTPUT_DIR}/motif_detection/results.csv \
        2>&1 | tee ${OUTPUT_DIR}/motif_detection/log.txt
else
    echo "Warning: Model not found. Skipping motif detection."
fi

# Experiment 4: Visualization
echo ""
echo "=========================================="
echo "Experiment 4: Qualitative Visualization"
echo "=========================================="
if [ -f "${MODEL_PATH}" ]; then
    python scripts/visualize_explanations.py \
        --model_path ${MODEL_PATH} \
        --num_samples 5 \
        --n_test 50 \
        --output_dir ${OUTPUT_DIR}/visualizations \
        2>&1 | tee ${OUTPUT_DIR}/visualizations/log.txt
else
    echo "Warning: Model not found. Skipping visualization."
fi

# Experiment 5: Ablations
echo ""
echo "=========================================="
echo "Experiment 5: Ablation Studies"
echo "=========================================="
python scripts/run_ablations.py \
    --experiment all \
    --output_dir ${OUTPUT_DIR}/ablations \
    --n_test 200 \
    --epochs 10 \
    2>&1 | tee ${OUTPUT_DIR}/ablations/log.txt

echo ""
echo "=========================================="
echo "All Experiments Completed!"
echo "=========================================="
echo "Results saved to: ${OUTPUT_DIR}/"
echo ""
echo "Summary:"
echo "  - Accuracy comparison: ${OUTPUT_DIR}/accuracy/comparison.csv"
echo "  - Fidelity curves: ${OUTPUT_DIR}/fidelity_curves/"
echo "  - Motif detection: ${OUTPUT_DIR}/motif_detection/results.csv"
echo "  - Visualizations: ${OUTPUT_DIR}/visualizations/"
echo "  - Ablations: ${OUTPUT_DIR}/ablations/"
echo "=========================================="
