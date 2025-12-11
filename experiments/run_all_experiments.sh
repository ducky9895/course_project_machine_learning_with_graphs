#!/usr/bin/env bash
# Master script to run all Graphormer paper replication experiments

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd ${SCRIPT_DIR}

# Configuration
CUDA_DEVICE=${CUDA_DEVICE:-0}
RUN_ZINC=${RUN_ZINC:-true}
RUN_PCQM4M=${RUN_PCQM4M:-true}
RUN_MOLHIV=${RUN_MOLHIV:-true}

echo "=========================================="
echo "Graphormer Paper Replication Experiments"
echo "=========================================="
echo "CUDA Device: ${CUDA_DEVICE}"
echo ""
echo "Experiments to run:"
echo "  - ZINC-500K: ${RUN_ZINC}"
echo "  - PCQM4M: ${RUN_PCQM4M}"
echo "  - OGBG-MolHIV: ${RUN_MOLHIV}"
echo "=========================================="
echo ""

# Create results and logs directories
mkdir -p results
mkdir -p logs

# Run ZINC experiment
if [ "${RUN_ZINC}" = "true" ]; then
    echo ""
    echo ">>> Starting ZINC-500K experiment..."
    echo ""
    export CUDA_DEVICE=${CUDA_DEVICE}
    bash run_zinc.sh
    echo ""
    echo ">>> ZINC-500K experiment completed!"
    echo ""
fi

# Run PCQM4M experiment
if [ "${RUN_PCQM4M}" = "true" ]; then
    echo ""
    echo ">>> Starting PCQM4M experiment..."
    echo ""
    export CUDA_DEVICE=${CUDA_DEVICE}
    bash run_pcqm4m.sh
    echo ""
    echo ">>> PCQM4M experiment completed!"
    echo ""
fi

# Run MolHIV experiment
if [ "${RUN_MOLHIV}" = "true" ]; then
    echo ""
    echo ">>> Starting OGBG-MolHIV experiment..."
    echo ""
    export CUDA_DEVICE=${CUDA_DEVICE}
    bash run_molhiv.sh
    echo ""
    echo ">>> OGBG-MolHIV experiment completed!"
    echo ""
fi

echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo "Results are saved in: ./results/"
echo "Logs are saved in: ./logs/"
echo "=========================================="

