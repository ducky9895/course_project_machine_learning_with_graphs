#!/bin/bash
# Full Pipeline: Train all models and run all evaluations
# This script runs the complete experimental pipeline from start to finish

set -e  # Exit on error

echo "=========================================="
echo "Full Experimental Pipeline"
echo "=========================================="
echo ""

# Configuration
DATASET=${1:-synthetic}  # synthetic or ba2motif
N_TRAIN=${2:-5000}
N_VAL=${3:-1000}
N_TEST=${4:-2000}
EPOCHS=${5:-50}
BATCH_SIZE=${6:-32}
CUDA_DEVICE=${7:-0}

echo "Configuration:"
echo "  Dataset: ${DATASET}"
echo "  Train/Val/Test: ${N_TRAIN}/${N_VAL}/${N_TEST}"
echo "  Epochs: ${EPOCHS}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  CUDA Device: ${CUDA_DEVICE}"
echo ""

# Create output directories
mkdir -p chkpts
mkdir -p results/{accuracy,fidelity_curves,motif_detection,visualizations,ablations}

# Step 1: Generate Dataset (if synthetic)
if [ "${DATASET}" == "synthetic" ]; then
    echo "=========================================="
    echo "Step 1: Generate Synthetic Dataset"
    echo "=========================================="
    
    # Check if dataset already exists
    DATASET_DIR="data/PT-Motifs/raw"
    if [ -d "${DATASET_DIR}" ] && [ "$(ls -A ${DATASET_DIR} 2>/dev/null)" ]; then
        echo "Dataset already exists at ${DATASET_DIR}"
        echo "Skipping dataset generation. Delete ${DATASET_DIR} to regenerate."
    else
        echo "Generating new dataset..."
        bash scripts/generate_dataset.sh ${N_TRAIN} ${N_VAL} ${N_TEST}
    fi
    echo ""
fi

# Step 2: Train Models
echo "=========================================="
echo "Step 2: Train Models"
echo "=========================================="

# 2.1: Train Pure Graphormer Baseline
echo ""
echo "2.1: Training Pure Graphormer Baseline..."
python main/train_v2.py \
    --dataset ${DATASET} \
    --n_train ${N_TRAIN} \
    --n_val ${N_VAL} \
    --n_test ${N_TEST} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --num_layers 4 \
    --embedding_dim 128 \
    --num_heads 4 \
    --lr 1e-4 \
    --baseline \
    --cuda ${CUDA_DEVICE} \
    --save_dir chkpts/pure_graphormer || echo "Warning: Pure Graphormer training failed or already exists"

# 2.2: Train Graphormer with Explainer
echo ""
echo "2.2: Training Graphormer with Explainer..."
python main/train_v2.py \
    --dataset ${DATASET} \
    --n_train ${N_TRAIN} \
    --n_val ${N_VAL} \
    --n_test ${N_TEST} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --num_layers 4 \
    --embedding_dim 128 \
    --num_heads 4 \
    --lr 1e-4 \
    --cuda ${CUDA_DEVICE} \
    --save_dir chkpts/graphormer_explainer || echo "Warning: Graphormer Explainer training failed or already exists"

# 2.3: Train Graphormer with Explainer + Regularization
echo ""
echo "2.3: Training Graphormer with Explainer + Regularization..."
python main/train_v2.py \
    --dataset ${DATASET} \
    --n_train ${N_TRAIN} \
    --n_val ${N_VAL} \
    --n_test ${N_TEST} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --num_layers 4 \
    --embedding_dim 128 \
    --num_heads 4 \
    --lr 1e-4 \
    --use_regularization \
    --reg_weight 0.01 \
    --cuda ${CUDA_DEVICE} \
    --save_dir chkpts/graphormer_explainer_reg || echo "Warning: Graphormer Explainer + Reg training failed or already exists"

# 2.4: Train Baselines (GCN, GIN) - optional
echo ""
echo "2.4: Training Baseline Models (GCN, GIN)..."
if [ -f "baselines/train_baselines.py" ]; then
    python baselines/train_baselines.py \
        --model gcn \
        --dataset ${DATASET} \
        --epochs ${EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --cuda ${CUDA_DEVICE} || echo "Warning: GCN training failed"
    
    python baselines/train_baselines.py \
        --model gin \
        --dataset ${DATASET} \
        --epochs ${EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --cuda ${CUDA_DEVICE} || echo "Warning: GIN training failed"
else
    echo "  Skipping baseline training (train_baselines.py not found)"
fi

echo ""
echo "=========================================="
echo "Step 3: Run All Evaluations"
echo "=========================================="

# 3.1: Predictive Accuracy Comparison
echo ""
echo "3.1: Experiment 1 - Predictive Accuracy..."
# Evaluate on all available datasets
if [ "${DATASET}" == "synthetic" ]; then
    # For synthetic, evaluate on synthetic dataset
    python scripts/eval_predictive_accuracy.py \
        --checkpoint_dir chkpts/ \
        --dataset synthetic \
        --n_test ${N_TEST} \
        --cuda ${CUDA_DEVICE} \
        --output results/accuracy/synthetic_comparison.csv || echo "Warning: Synthetic accuracy evaluation failed"
    
    # Also try to evaluate on BA2Motif if available
    python scripts/eval_predictive_accuracy.py \
        --checkpoint_dir chkpts/ \
        --dataset ba2motif \
        --cuda ${CUDA_DEVICE} \
        --output results/accuracy/ba2motif_comparison.csv || echo "BA2Motif dataset not available, skipping"
else
    # For other datasets, evaluate on that dataset
    python scripts/eval_predictive_accuracy.py \
        --checkpoint_dir chkpts/ \
        --dataset ${DATASET} \
        --n_test ${N_TEST} \
        --cuda ${CUDA_DEVICE} \
        --output results/accuracy/${DATASET}_comparison.csv || echo "Warning: Accuracy evaluation failed"
fi

# Combine all available results
python scripts/eval_all_datasets.py \
    --checkpoint_dir chkpts/ \
    --output_dir results/accuracy \
    --skip_eval || echo "Could not combine results (some datasets may be missing)"

# 3.2: Find best model for explainer-based evaluations
EXPLAINER_MODEL=$(find chkpts -name "best_model.pt" -path "*/graphormer_explainer*" | head -1)
if [ -z "${EXPLAINER_MODEL}" ]; then
    EXPLAINER_MODEL=$(find chkpts -name "best_model.pt" | grep -v "pure_graphormer" | head -1)
fi

if [ -n "${EXPLAINER_MODEL}" ] && [ -f "${EXPLAINER_MODEL}" ]; then
    echo ""
    echo "Using model: ${EXPLAINER_MODEL}"
    
    # 3.2: Fidelity Curves
    echo ""
    echo "3.2: Experiment 2 - Fidelity Curves..."
    python scripts/eval_fidelity_curves.py \
        --model_path ${EXPLAINER_MODEL} \
        --dataset ${DATASET} \
        --n_test $((N_TEST / 10)) \
        --cuda ${CUDA_DEVICE} \
        --output_dir results/fidelity_curves || echo "Warning: Fidelity curves failed"
    
    # 3.3: Motif Detection
    echo ""
    echo "3.3: Experiment 3 - Motif Detection..."
    python scripts/eval_motif_detection.py \
        --model_path ${EXPLAINER_MODEL} \
        --dataset ${DATASET} \
        --n_test ${N_TEST} \
        --cuda ${CUDA_DEVICE} \
        --output results/motif_detection/results.csv || echo "Warning: Motif detection failed"
    
    # 3.4: Qualitative Visualization
    echo ""
    echo "3.4: Experiment 4 - Qualitative Visualization..."
    python scripts/visualize_explanations.py \
        --model_path ${EXPLAINER_MODEL} \
        --num_samples 5 \
        --n_test $((N_TEST / 20)) \
        --cuda ${CUDA_DEVICE} \
        --output_dir results/visualizations || echo "Warning: Visualization failed"
else
    echo ""
    echo "Warning: No explainer model found, skipping explainer-based evaluations"
    echo "  Train a model with explainer first: bash scripts/train_explainer.sh"
fi

# 3.5: Ablation Studies (optional, can be slow)
echo ""
echo "3.5: Experiment 5 - Ablation Studies..."
echo "  (Skipping - run separately with: python scripts/run_ablations.py --experiment all)"
# python scripts/run_ablations.py \
#     --experiment all \
#     --output_dir results/ablations \
#     --n_test $((N_TEST / 10)) \
#     --epochs 10 || echo "Warning: Ablations failed"

# Step 4: View Results Summary
echo ""
echo "=========================================="
echo "Step 4: Results Summary"
echo "=========================================="
echo ""

if [ -f "results/accuracy/comparison.csv" ]; then
    echo "Predictive Accuracy Results:"
    cat results/accuracy/comparison.csv
    echo ""
fi

echo "All results saved to:"
echo "  - Accuracy: results/accuracy/comparison.csv"
echo "  - Fidelity curves: results/fidelity_curves/"
echo "  - Motif detection: results/motif_detection/results.csv"
echo "  - Visualizations: results/visualizations/"
echo ""

# Step 5: View Detailed Results
echo "=========================================="
echo "Step 5: View Detailed Results"
echo "=========================================="
echo ""
echo "To view detailed results, run:"
echo "  bash scripts/view_results.sh"
echo ""

echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. View results: bash scripts/view_results.sh"
echo "  2. Check visualizations: ls results/visualizations/"
echo "  3. Run ablations: python scripts/run_ablations.py --experiment all"
echo ""
