#!/bin/bash
# View training results from checkpoints

set -e

CHECKPOINT_DIR=${1:-chkpts}
EXPERIMENT=${2:-}

echo "=========================================="
echo "Viewing Results"
echo "=========================================="

if [ -z "${EXPERIMENT}" ]; then
    python scripts/evaluate_results.py --checkpoint_dir ${CHECKPOINT_DIR}
else
    python scripts/evaluate_results.py \
        --checkpoint_dir ${CHECKPOINT_DIR} \
        --experiment ${EXPERIMENT}
fi
