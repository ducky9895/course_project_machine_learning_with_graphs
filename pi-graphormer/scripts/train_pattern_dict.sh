#!/bin/bash
# Train Graphormer with Pattern Dictionary (PI-GNN Style)

set -e

echo "=========================================="
echo "Training Graphormer with Pattern Dictionary"
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
    --use_pattern_dict \
    --num_pattern_atoms 32 \
    --pattern_sparsity_weight 0.01 \
    --save_dir chkpts/graphormer_pattern_dict

echo ""
echo "✓ Pattern dictionary training complete!"
echo "Model saved to: chkpts/graphormer_pattern_dict/"
