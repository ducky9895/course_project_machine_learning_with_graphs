#!/bin/bash
# Train Pure Graphormer (Baseline - No Explainer)

set -e

echo "=========================================="
echo "Training Pure Graphormer Baseline"
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
    --baseline \
    --save_dir chkpts/pure_graphormer

echo ""
echo "✓ Baseline training complete!"
echo "Model saved to: chkpts/pure_graphormer/"
