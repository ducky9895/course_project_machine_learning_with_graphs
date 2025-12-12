# Complete Implementation Guide

## Overview

This guide provides step-by-step instructions to implement and run all evaluation experiments.

## Prerequisites

### 1. Install Dependencies

```bash
pip install torch torch-geometric numpy scikit-learn matplotlib pandas networkx tqdm
```

### 2. Prepare Data

```bash
# Generate synthetic dataset
python generate_ptmotifs.py \
    --output_dir data/PT-Motifs/raw \
    --n_train 5000 --n_val 1000 --n_test 2000

# Or use BA-2Motif (if available)
# Download BA-2Motif dataset to data/BA2Motif/
```

## Step-by-Step Implementation

### Step 1: Train Models

#### 1.1 Pure Graphormer (Baseline)
```bash
python main/train_v2.py \
    --dataset synthetic \
    --n_train 5000 --n_val 1000 --n_test 2000 \
    --epochs 50 \
    --batch_size 32 \
    --num_layers 4 \
    --embedding_dim 128 \
    --baseline \
    --save_dir chkpts/pure_graphormer
```

#### 1.2 Graphormer + Explainer
```bash
python main/train_v2.py \
    --dataset synthetic \
    --n_train 5000 --n_val 1000 --n_test 2000 \
    --epochs 50 \
    --batch_size 32 \
    --num_layers 4 \
    --embedding_dim 128 \
    --save_dir chkpts/graphormer_explainer
```

#### 1.3 Graphormer + Explainer + Regularization
```bash
python main/train_v2.py \
    --dataset synthetic \
    --n_train 5000 --n_val 1000 --n_test 2000 \
    --epochs 50 \
    --batch_size 32 \
    --num_layers 4 \
    --embedding_dim 128 \
    --use_regularization \
    --reg_weight 0.01 \
    --save_dir chkpts/graphormer_explainer_reg
```

#### 1.4 Train GCN Baseline
```bash
python baselines/train_baselines.py \
    --model gcn \
    --dataset synthetic \
    --n_train 5000 --n_val 1000 --n_test 2000 \
    --epochs 50 \
    --batch_size 32 \
    --save_dir chkpts/baselines
```

#### 1.5 Train GIN Baseline
```bash
python baselines/train_baselines.py \
    --model gin \
    --dataset synthetic \
    --n_train 5000 --n_val 1000 --n_test 2000 \
    --epochs 50 \
    --batch_size 32 \
    --save_dir chkpts/baselines
```

### Step 2: Run Evaluation Experiments

#### Option A: Run All Experiments
```bash
bash scripts/run_all_evaluations.sh
```

#### Option B: Run Individual Experiments

**Experiment 1: Predictive Accuracy**
```bash
python scripts/eval_predictive_accuracy.py \
    --checkpoint_dir chkpts/ \
    --dataset synthetic \
    --n_test 500 \
    --output results/accuracy/comparison.csv
```

**Experiment 2: Fidelity Curves**
```bash
python scripts/eval_fidelity_curves.py \
    --model_path chkpts/graphormer_explainer/best_model.pt \
    --dataset synthetic \
    --n_test 200 \
    --output_dir results/fidelity_curves
```

**Experiment 3: Motif Detection**
```bash
python scripts/eval_motif_detection.py \
    --model_path chkpts/graphormer_explainer/best_model.pt \
    --dataset synthetic \
    --n_test 500 \
    --output results/motif_detection/results.csv
```

**Experiment 4: Visualization**
```bash
python scripts/visualize_explanations.py \
    --model_path chkpts/graphormer_explainer/best_model.pt \
    --num_samples 5 \
    --output_dir results/visualizations
```

**Experiment 5: Ablations**
```bash
# Sparsity coefficient
python scripts/run_ablations.py \
    --experiment sparsity \
    --output_dir results/ablations

# Edge scoring heads
python scripts/run_ablations.py \
    --experiment edge_heads \
    --output_dir results/ablations
```

## Expected Outputs

### Experiment 1: Predictive Accuracy
- **File**: `results/accuracy/comparison.csv`
- **Content**: Table with accuracy, F1, precision, recall for each model

### Experiment 2: Fidelity Curves
- **Files**: 
  - `results/fidelity_curves/deletion_curve.png`
  - `results/fidelity_curves/insertion_curve.png`
- **Content**: Plots showing accuracy vs % edges removed/added

### Experiment 3: Motif Detection
- **File**: `results/motif_detection/results.csv`
- **Content**: Precision@k, Recall@k, F1@k for each method

### Experiment 4: Visualization
- **Files**: `results/visualizations/sample_*.png`
- **Content**: Side-by-side visualizations of ground truth, Graphormer, Random

### Experiment 5: Ablations
- **Files**:
  - `results/ablations/ablation_sparsity.png`
  - `results/ablations/ablation_edge_heads.png`
- **Content**: Plots showing ablation study results

## Troubleshooting

### Issue: Model Not Found
**Solution**: Check checkpoint directory structure:
```bash
ls -R chkpts/
# Should see: chkpts/pure_graphormer/best_model.pt
#            chkpts/graphormer_explainer/best_model.pt
```

### Issue: CUDA Out of Memory
**Solution**: Reduce batch size:
```bash
--batch_size 16  # instead of 32
```

### Issue: Dataset Not Found
**Solution**: Generate dataset first:
```bash
python generate_ptmotifs.py --output_dir data/PT-Motifs/raw --n_train 1000
```

### Issue: Import Errors
**Solution**: Install missing packages:
```bash
pip install matplotlib networkx scikit-learn pandas
```

## Quick Start (Minimal)

For a quick test with minimal data:

```bash
# 1. Generate small dataset
python generate_ptmotifs.py --output_dir data/PT-Motifs/raw --n_train 500 --n_val 100 --n_test 100

# 2. Train quick model
python main/train_v2.py \
    --dataset synthetic \
    --n_train 500 --n_val 100 --n_test 100 \
    --epochs 10 \
    --batch_size 16 \
    --save_dir chkpts/quick_test

# 3. Run evaluation
python scripts/eval_predictive_accuracy.py \
    --checkpoint_dir chkpts/ \
    --dataset synthetic \
    --n_test 100
```

## Complete Workflow

```bash
# 1. Setup
mkdir -p chkpts results data

# 2. Generate data
python generate_ptmotifs.py --output_dir data/PT-Motifs/raw --n_train 5000 --n_val 1000 --n_test 2000

# 3. Train all models
python main/train_v2.py --dataset synthetic --n_train 5000 --epochs 50 --baseline --save_dir chkpts/pure_graphormer
python main/train_v2.py --dataset synthetic --n_train 5000 --epochs 50 --save_dir chkpts/graphormer_explainer
python baselines/train_baselines.py --model gcn --dataset synthetic --n_train 5000 --epochs 50
python baselines/train_baselines.py --model gin --dataset synthetic --n_train 5000 --epochs 50

# 4. Run all evaluations
bash scripts/run_all_evaluations.sh

# 5. View results
ls results/
```

## Next Steps

1. **Review Results**: Check all output files in `results/`
2. **Analyze**: Compare metrics across experiments
3. **Visualize**: Review plots and visualizations
4. **Report**: Document findings in paper/report

Good luck! 🎉
