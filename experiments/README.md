# Graphormer Paper Replication Experiments

This directory contains scripts to replicate the experiments from the Graphormer paper:

**"Do Transformers Really Perform Badly for Graph Representation?"** (NeurIPS 2021)
- Ying et al., Microsoft Research

## Experiments

### 1. ZINC-500K (Molecular Property Prediction - Regression)
- **Dataset**: ZINC-500K from PyTorch Geometric
- **Model**: Graphormer-Slim
- **Expected Result**: Test MAE ~0.122
- **Script**: `run_zinc.sh`
- **Training Time**: ~400K updates (several hours on GPU)

### 2. PCQM4M (Large-Scale Molecular Property Prediction)
- **Dataset**: PCQM4M from Open Graph Benchmark
- **Model**: Graphormer-Base
- **Expected Result**: Valid MAE ~0.1201
- **Script**: `run_pcqm4m.sh`
- **Training Time**: ~1M updates (requires significant GPU resources)

### 3. OGBG-MolHIV (Molecular Property Prediction - Classification)
- **Dataset**: OGBG-MolHIV from Open Graph Benchmark
- **Model**: Graphormer-Base (pre-trained on PCQM4M)
- **Expected Result**: Test AUC ~80.56%
- **Script**: `run_molhiv.sh`
- **Training Time**: ~4 epochs (fine-tuning)

## Prerequisites

1. **Install Graphormer**:
   ```bash
   cd Graphormer
   bash install.sh
   ```

2. **Install Dependencies**:
   - PyTorch (with CUDA support)
   - PyTorch Geometric
   - Open Graph Benchmark (OGB)
   - Fairseq (included in Graphormer)

3. **GPU Requirements**:
   - ZINC: 1 GPU recommended
   - PCQM4M: Multiple GPUs recommended (can use 1 GPU but will be slow)
   - MolHIV: 1 GPU sufficient (fine-tuning)

## Usage

### Run Individual Experiments

```bash
# ZINC-500K
bash experiments/run_zinc.sh

# PCQM4M
bash experiments/run_pcqm4m.sh

# OGBG-MolHIV (requires pre-trained PCQM4M model)
bash experiments/run_molhiv.sh
```

### Run All Experiments

```bash
# Run all experiments sequentially
bash experiments/run_all_experiments.sh

# Run specific experiments only
RUN_ZINC=true RUN_PCQM4M=false RUN_MOLHIV=false bash experiments/run_all_experiments.sh
```

### Customize GPU Usage

```bash
# Use specific GPU
CUDA_DEVICE=0 bash experiments/run_zinc.sh

# Use multiple GPUs (for PCQM4M)
CUDA_DEVICE=0,1,2,3 bash experiments/run_pcqm4m.sh
```

## Output Structure

```
experiments/
├── results/
│   ├── zinc_graphormer_slim/
│   │   └── checkpoints and best model
│   ├── pcqm4m_graphormer_base/
│   │   └── checkpoints and best model
│   └── molhiv_graphormer_base/
│       └── checkpoints and best model
└── logs/
    ├── zinc_graphormer_slim/
    │   └── training.log
    ├── pcqm4m_graphormer_base/
    │   └── training.log
    └── molhiv_graphormer_base/
        └── training.log
```

## Evaluation

After training, you can evaluate the models using Graphormer's evaluation scripts:

```bash
cd Graphormer/examples/property_prediction
python check_results.py --checkpoint-dir ../../experiments/results/zinc_graphormer_slim
```

## Expected Results (from Paper)

| Dataset | Model | Metric | Expected Value |
|---------|-------|--------|----------------|
| ZINC-500K | Graphormer-Slim | Test MAE | 0.122 |
| PCQM4M | Graphormer-Base | Valid MAE | 0.1201 |
| OGBG-MolHIV | Graphormer-Base (pre-trained) | Test AUC | 80.56% |

## Notes

1. **PCQM4M Pre-training**: The MolHIV experiment requires a pre-trained model on PCQM4M. You can either:
   - Train PCQM4M first, then use that checkpoint
   - Download a pre-trained model from Graphormer's model zoo

2. **Memory Requirements**: 
   - ZINC: ~8GB GPU memory
   - PCQM4M: ~24GB GPU memory (or use gradient accumulation)
   - MolHIV: ~16GB GPU memory

3. **Training Time** (approximate on V100):
   - ZINC: 6-8 hours
   - PCQM4M: 2-3 days
   - MolHIV: 1-2 hours (fine-tuning)

4. **Data Download**: Datasets will be automatically downloaded on first run:
   - ZINC: Downloaded via PyTorch Geometric
   - PCQM4M: Downloaded via OGB
   - MolHIV: Downloaded via OGB

## Troubleshooting

1. **Out of Memory**: Reduce batch size or use gradient accumulation
2. **Fairseq not found**: Make sure Graphormer is installed and in PYTHONPATH
3. **Dataset download issues**: Check internet connection and disk space

## Citation

If you use these scripts, please cite the Graphormer paper:

```bibtex
@inproceedings{ying2021do,
  title={Do Transformers Really Perform Badly for Graph Representation?},
  author={Ying, Chengxuan and Cai, Tianle and Luo, Shengjie and Zheng, Shuxin and Ke, Guolin and He, Di and Shen, Yanming and Liu, Tie-Yan},
  booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
  year={2021}
}
```

