#!/bin/bash
# Evaluate all models on all available datasets

set -e

CHECKPOINT_DIR=${1:-chkpts}
OUTPUT_DIR=${2:-results/accuracy}

echo "=========================================="
echo "Evaluating All Models on All Datasets"
echo "=========================================="
echo ""

# Create output directory
mkdir -p ${OUTPUT_DIR}

# List of datasets to evaluate
DATASETS=("synthetic" "ba2motif" "mutag")

# Evaluate each dataset
for dataset in "${DATASETS[@]}"; do
    echo "=========================================="
    echo "Evaluating on: ${dataset}"
    echo "=========================================="
    
    OUTPUT_FILE="${OUTPUT_DIR}/${dataset}_comparison.csv"
    
    python scripts/eval_predictive_accuracy.py \
        --dataset ${dataset} \
        --checkpoint_dir ${CHECKPOINT_DIR} \
        --output ${OUTPUT_FILE} \
        --n_test 2000 || echo "Warning: Failed to evaluate ${dataset}"
    
    echo ""
done

# Combine results
echo "=========================================="
echo "Combining Results"
echo "=========================================="

python scripts/eval_all_datasets.py \
    --checkpoint_dir ${CHECKPOINT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --skip_eval

echo ""
echo "✓ All evaluations complete!"
echo "Results saved to: ${OUTPUT_DIR}/"
