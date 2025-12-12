#!/bin/bash
# Quick test training (5 minutes) - Minimal model to verify everything works

set -e

echo "=========================================="
echo "Quick Test Training"
echo "=========================================="

python main/train_v2.py \
    --dataset synthetic \
    --n_train 500 \
    --n_val 100 \
    --n_test 200 \
    --epochs 5 \
    --batch_size 16 \
    --num_layers 2 \
    --embedding_dim 64 \
    --save_dir chkpts/quick_test

echo ""
echo "✓ Quick test training complete!"
echo "Check results in: chkpts/quick_test/"
