#!/bin/bash
# Train Graphormer with Explainer + Regularization

set -e

REG_WEIGHT=${1:-0.01}  # Default to 0.01 if not provided

echo "=========================================="
echo "Training Graphormer with Explainer + Regularization"
echo "Regularization weight: ${REG_WEIGHT}"
echo "=========================================="

python main/train_v2.py \
    --dataset synthetic \
    --n_train 5000 \
    --n_val 1000 \
    --n_test 2000 \
    --epochs 50 \
    --batch_size 32 \
    --num_layers 4 \
    --embedding_dim 128 \
    --num_heads 4 \
    --lr 1e-4 \
    --use_regularization \
    --reg_weight ${REG_WEIGHT} \
    --save_dir chkpts/graphormer_explainer_reg_${REG_WEIGHT}

echo ""
echo "✓ Training with regularization complete!"
echo "Model saved to: chkpts/graphormer_explainer_reg_${REG_WEIGHT}/"
