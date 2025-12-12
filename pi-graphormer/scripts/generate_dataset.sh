#!/bin/bash
# Generate synthetic dataset manually (optional - auto-generated if missing)

set -e

N_TRAIN=${1:-5000}
N_VAL=${2:-1000}
N_TEST=${3:-2000}

echo "=========================================="
echo "Generating Synthetic Dataset"
echo "Train: ${N_TRAIN}, Val: ${N_VAL}, Test: ${N_TEST}"
echo "=========================================="

python generate_ptmotifs.py \
    --output_dir data/PT-Motifs/raw \
    --n_train ${N_TRAIN} \
    --n_val ${N_VAL} \
    --n_test ${N_TEST}

echo ""
echo "✓ Dataset generation complete!"
echo "Data saved to: data/PT-Motifs/raw/"
