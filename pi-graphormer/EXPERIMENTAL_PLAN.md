# Experimental Plan: Graphormer-PIGNN v2

## Overview

This document outlines the experimental design for evaluating Graphormer-PIGNN v2 (simplified architecture).

## Experimental Phases

### Phase 1: Pre-training on Synthetic Data
**Goal**: Learn universal structural patterns for explanation

**Datasets**:
- **PT-Motifs**: Large-scale synthetic dataset (20K train, 5K val, 5K test)
- **Simple Synthetic**: Quick sanity check (5K train, 1K val, 2K test)

**Experiments**:
1. **Baseline**: No regularization
2. **L1 Sparsity**: `λ = 0.01, 0.05, 0.1`
3. **Entropy Sparsity**: `λ = 0.01, 0.05, 0.1`
4. **Combined**: L1 + Connectivity + Size regularizers

**Metrics**:
- Prediction accuracy
- Explanation ROC-AUC (vs ground truth)
- Explanation sparsity (fraction of edges with score > 0.5)
- Average edge score magnitude

### Phase 2: Downstream Task Evaluation
**Goal**: Evaluate transfer learning and fine-tuning

**Datasets**:
- **BA-2Motif**: Binary classification (house vs cycle motif)
- **Mutag**: Mutagenicity prediction (if available)

**Experiments**:
1. **From Scratch**: Train without pre-training
2. **Pre-trained**: Load pre-trained weights, fine-tune
3. **Frozen Encoder**: Freeze Graphormer, only train classifier + edge scorer

**Metrics**:
- Prediction accuracy
- Explanation quality (if ground truth available)
- Training time/convergence speed

### Phase 3: Ablation Studies
**Goal**: Understand contribution of each component

**Experiments**:
1. **No Regularization**: Baseline
2. **L1 Only**: `λ = 0.01`
3. **Entropy Only**: `λ = 0.01`
4. **L1 + Connectivity**: `λ_l1 = 0.01, λ_conn = 0.01`
5. **All Regularizers**: `λ_l1 = 0.01, λ_conn = 0.01, λ_size = 0.01`

**Metrics**:
- Prediction accuracy
- Explanation sparsity
- Explanation connectivity (fraction of nodes in largest component)
- Number of important edges

### Phase 4: Hyperparameter Sensitivity
**Goal**: Find optimal hyperparameters

**Hyperparameters to Sweep**:
- `embedding_dim`: [64, 128, 256]
- `num_layers`: [2, 4, 6]
- `num_heads`: [2, 4, 8]
- `sparsity_weight`: [0.001, 0.01, 0.1]
- `learning_rate`: [1e-5, 1e-4, 1e-3]

**Method**: Grid search or random search (limited budget)

## Experiment Structure

```
experiments/
├── phase1_pretraining/
│   ├── exp1_baseline.sh
│   ├── exp2_l1_sparsity.sh
│   ├── exp3_entropy_sparsity.sh
│   └── exp4_combined.sh
├── phase2_downstream/
│   ├── exp1_from_scratch.sh
│   ├── exp2_pretrained.sh
│   └── exp3_frozen.sh
├── phase3_ablation/
│   ├── exp1_no_reg.sh
│   ├── exp2_l1_only.sh
│   └── ...
└── phase4_hyperparams/
    └── sweep_config.yaml
```

## Quick Start Experiments

### Experiment 1: Quick Sanity Check
```bash
# Generate small synthetic dataset
python generate_ptmotifs.py --n_train 1000 --n_val 200 --n_test 200

# Train baseline model
python main/train_v2.py \
    --dataset synthetic \
    --n_train 1000 --n_val 200 --n_test 200 \
    --epochs 10 \
    --batch_size 16 \
    --num_layers 2 \
    --embedding_dim 64
```

### Experiment 2: Pre-training with Regularization
```bash
python main/train_v2.py \
    --dataset synthetic \
    --n_train 5000 --n_val 1000 --n_test 2000 \
    --epochs 50 \
    --use_regularization \
    --reg_weight 0.01
```

### Experiment 3: Fine-tuning on BA-2Motif
```bash
python main/train_v2.py \
    --dataset ba2motif \
    --pretrained_path chkpts/synthetic_*/best_model.pt \
    --epochs 30 \
    --use_regularization \
    --reg_weight 0.01
```

## Evaluation Metrics

### Prediction Metrics
- **Accuracy**: Classification accuracy
- **F1-Score**: Per-class F1 scores
- **Confusion Matrix**: For multi-class tasks

### Explanation Metrics (when ground truth available)
- **ROC-AUC**: Area under ROC curve
- **Precision**: Precision at threshold 0.5
- **Recall**: Recall at threshold 0.5
- **F1-Score**: F1 score for explanation

### Explanation Properties
- **Sparsity**: Fraction of edges with score > threshold
- **Connectivity**: Fraction of nodes in largest connected component
- **Size**: Average number of important edges per graph
- **Score Distribution**: Histogram of edge scores

## Expected Results

### Pre-training Phase
- **Baseline**: High accuracy, low explanation sparsity
- **With Regularization**: Slightly lower accuracy, higher explanation sparsity
- **Best**: Balance between accuracy and sparsity

### Downstream Tasks
- **From Scratch**: Lower accuracy, slower convergence
- **Pre-trained**: Higher accuracy, faster convergence
- **Frozen Encoder**: Fastest training, may have lower accuracy

### Ablation Studies
- **No Regularization**: Dense explanations (many edges important)
- **L1**: Sparse explanations (few edges important)
- **Entropy**: Sparse + confident explanations (scores near 0 or 1)
- **Combined**: Best balance of sparsity and connectivity

## Timeline

1. **Week 1**: Phase 1 (Pre-training)
2. **Week 2**: Phase 2 (Downstream tasks)
3. **Week 3**: Phase 3 (Ablation studies)
4. **Week 4**: Phase 4 (Hyperparameter tuning) + Analysis

## Success Criteria

- ✅ Pre-training achieves >90% accuracy on synthetic data
- ✅ Explanation ROC-AUC >0.8 on synthetic data
- ✅ Fine-tuning improves downstream task performance
- ✅ Regularization produces sparse, interpretable explanations
- ✅ Model converges within reasonable time (<2 hours per experiment)
