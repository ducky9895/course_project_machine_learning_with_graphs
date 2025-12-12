#!/bin/bash
# Quick Pipeline: Fast test run (small dataset, fewer epochs)
# Use this for quick testing before running the full pipeline

set -e  # Exit on error

echo "=========================================="
echo "Quick Pipeline (Fast Test)"
echo "=========================================="
echo ""

# Configuration for quick test
DATASET=${1:-synthetic}
N_TRAIN=${2:-500}
N_VAL=${3:-100}
N_TEST=${4:-200}
EPOCHS=${5:-5}
BATCH_SIZE=${6:-16}
CUDA_DEVICE=${7:-0}

echo "Quick Test Configuration:"
echo "  Dataset: ${DATASET}"
echo "  Train/Val/Test: ${N_TRAIN}/${N_VAL}/${N_TEST}"
echo "  Epochs: ${EPOCHS}"
echo "  Batch Size: ${BATCH_SIZE}"
echo ""

# Create output directories
mkdir -p chkpts/quick_test
mkdir -p results/quick_test

# Step 1: Generate Dataset
if [ "${DATASET}" == "synthetic" ]; then
    echo "Generating synthetic dataset..."
    bash scripts/generate_dataset.sh ${N_TRAIN} ${N_VAL} ${N_TEST}
    echo ""
fi

# Step 2: Train Quick Test Model
echo "Training quick test model..."
python main/train_v2.py \
    --dataset ${DATASET} \
    --n_train ${N_TRAIN} \
    --n_val ${N_VAL} \
    --n_test ${N_TEST} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --num_layers 2 \
    --embedding_dim 64 \
    --num_heads 2 \
    --lr 1e-4 \
    --cuda ${CUDA_DEVICE} \
    --save_dir chkpts/quick_test

# Step 3: Quick Evaluation
echo ""
echo "Running quick evaluation..."
MODEL_PATH=$(find chkpts/quick_test -name "best_model.pt" | head -1)

if [ -n "${MODEL_PATH}" ] && [ -f "${MODEL_PATH}" ]; then
    echo "Found model: ${MODEL_PATH}"
    
    # Quick accuracy check
    python scripts/eval_predictive_accuracy.py \
        --checkpoint_dir chkpts/quick_test \
        --dataset ${DATASET} \
        --n_test ${N_TEST} \
        --cuda ${CUDA_DEVICE} \
        --output results/quick_test/accuracy.csv
    
    # Quick visualization
    python scripts/visualize_explanations.py \
        --model_path ${MODEL_PATH} \
        --num_samples 3 \
        --n_test 20 \
        --cuda ${CUDA_DEVICE} \
        --output_dir results/quick_test/visualizations
else
    echo "Model not found, skipping evaluation"
fi

echo ""
echo "=========================================="
echo "Quick Pipeline Complete!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - Accuracy: results/quick_test/accuracy.csv"
echo "  - Visualizations: results/quick_test/visualizations/"
echo ""
