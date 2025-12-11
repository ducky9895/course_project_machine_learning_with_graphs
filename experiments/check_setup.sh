#!/usr/bin/env bash
# Script to check if Graphormer is properly set up

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRAPHORMER_DIR="${SCRIPT_DIR}/../Graphormer"

echo "=========================================="
echo "Checking Graphormer Setup"
echo "=========================================="

# Check if Graphormer directory exists
if [ ! -d "${GRAPHORMER_DIR}" ]; then
    echo "❌ Graphormer directory not found at: ${GRAPHORMER_DIR}"
    exit 1
else
    echo "✅ Graphormer directory found"
fi

# Check if fairseq directory exists
if [ ! -d "${GRAPHORMER_DIR}/fairseq" ]; then
    echo "❌ Fairseq directory not found in Graphormer"
    echo "   You may need to: cd ${GRAPHORMER_DIR} && git submodule update --init --recursive"
    exit 1
else
    echo "✅ Fairseq directory found"
fi

# Check if fairseq-train command exists
if command -v fairseq-train &> /dev/null; then
    echo "✅ fairseq-train command found in PATH"
    FAIRSEQ_CMD="fairseq-train"
elif python -c "import fairseq_cli.train" 2>/dev/null; then
    echo "✅ fairseq_cli.train module found (can use: python -m fairseq_cli.train)"
    FAIRSEQ_CMD="python -m fairseq_cli.train"
else
    echo "❌ fairseq-train not found"
    echo "   Please install Graphormer:"
    echo "   cd ${GRAPHORMER_DIR} && bash install.sh"
    exit 1
fi

# Check Python packages
echo ""
echo "Checking Python packages..."
python -c "import torch; print(f'✅ PyTorch {torch.__version__}')" || echo "❌ PyTorch not found"
python -c "import torch_geometric; print(f'✅ PyTorch Geometric {torch_geometric.__version__}')" || echo "❌ PyTorch Geometric not found"
python -c "import ogb; print(f'✅ OGB {ogb.__version__}')" || echo "❌ OGB not found"

# Check CUDA availability
echo ""
echo "Checking CUDA..."
python -c "import torch; print(f'✅ CUDA available: {torch.cuda.is_available()}')" || echo "❌ Could not check CUDA"

# Check if graphormer module can be imported
echo ""
echo "Checking Graphormer module..."
if python -c "import sys; sys.path.insert(0, '${GRAPHORMER_DIR}'); from graphormer import *" 2>/dev/null; then
    echo "✅ Graphormer module can be imported"
else
    echo "⚠️  Graphormer module import check failed (may still work with --user-dir)"
fi

echo ""
echo "=========================================="
echo "Setup check complete!"
echo "=========================================="
echo "To run experiments, use:"
echo "  bash experiments/run_zinc.sh"
echo "  bash experiments/run_pcqm4m.sh"
echo "  bash experiments/run_molhiv.sh"
echo "=========================================="

