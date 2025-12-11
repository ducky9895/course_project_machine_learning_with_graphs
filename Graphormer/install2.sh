#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# Usage:
#   1. Create conda environment: conda env create -f environment.yml
#   2. Activate: conda activate graphormer
#   3. Run this script: bash install2.sh

# Install PyTorch Geometric and DGL (these need special handling)
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
pip install torch-geometric
pip install dgl -f https://data.dgl.ai/wheels/cu121/repo.html

# Install fairseq dependencies manually to avoid conflicts
pip install "omegaconf==2.0.0" --no-deps || pip install "omegaconf<2.1"
pip install "hydra-core>=1.0.7,<1.1" "regex" "sacrebleu>=1.4.12" "bitarray" "cffi" "tqdm"

# Then continue with fairseq installation
cd fairseq
pip uninstall -y fairseq fairseq-nat 2>/dev/null || true
pip install --no-deps -e .
python setup.py build_ext --inplace
