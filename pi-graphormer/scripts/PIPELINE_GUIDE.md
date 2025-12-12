# Pipeline Guide

## Quick Start

### Option 1: Quick Test Pipeline (5-10 minutes)
Fast test to verify everything works:

```bash
bash scripts/run_quick_pipeline.sh
```

This will:
- Generate small dataset (500/100/200)
- Train a small model (5 epochs, 2 layers, 64 dim)
- Run quick evaluations
- Save results to `results/quick_test/`

### Option 2: Full Pipeline (Several hours)
Complete experimental pipeline:

```bash
bash scripts/run_full_pipeline.sh
```

This will:
- Generate full dataset (5000/1000/2000)
- Train all models (Pure Graphormer, Explainer, Explainer+Reg, GCN, GIN)
- Run all 5 experiments
- Save all results to `results/`

## Custom Configuration

### Quick Pipeline
```bash
bash scripts/run_quick_pipeline.sh [dataset] [n_train] [n_val] [n_test] [epochs] [batch_size] [cuda]

# Example:
bash scripts/run_quick_pipeline.sh synthetic 1000 200 400 10 32 0
```

### Full Pipeline
```bash
bash scripts/run_full_pipeline.sh [dataset] [n_train] [n_val] [n_test] [epochs] [batch_size] [cuda]

# Example:
bash scripts/run_full_pipeline.sh synthetic 10000 2000 4000 100 64 0
```

## What Gets Trained

### Full Pipeline Trains:
1. **Pure Graphormer** - Baseline (no explainer)
2. **Graphormer + Explainer** - With edge scoring
3. **Graphormer + Explainer + Regularization** - With sparsity regularization
4. **GCN** - Baseline GNN (if train_baselines.py exists)
5. **GIN** - Baseline GNN (if train_baselines.py exists)

### Quick Pipeline Trains:
1. **Quick Test Model** - Small model for verification

## What Gets Evaluated

### Full Pipeline Runs:
1. **Predictive Accuracy** - Compare all models
2. **Fidelity Curves** - Deletion and insertion curves
3. **Motif Detection** - Precision@k, Recall@k, F1@k
4. **Visualization** - Sample graph explanations
5. **Ablations** - Skipped by default (run separately)

### Quick Pipeline Runs:
1. **Predictive Accuracy** - Quick check
2. **Visualization** - 3 sample graphs

## Output Structure

```
results/
├── accuracy/
│   └── comparison.csv          # Model comparison table
├── fidelity_curves/
│   ├── deletion_curve.png      # Deletion fidelity plot
│   ├── insertion_curve.png     # Insertion fidelity plot
│   └── results.csv             # Raw data
├── motif_detection/
│   └── results.csv             # Precision/Recall/F1 metrics
└── visualizations/
    ├── graph_0.png             # Explanation visualizations
    ├── graph_1.png
    └── ...
```

## Running Individual Steps

If you want to run steps individually:

```bash
# Step 1: Generate dataset
bash scripts/generate_dataset.sh 5000 1000 2000

# Step 2: Train models
bash scripts/train_baseline.sh
bash scripts/train_explainer.sh
bash scripts/train_with_reg.sh

# Step 3: Run evaluations
bash scripts/eval_accuracy.sh
bash scripts/eval_fidelity.sh chkpts/graphormer_explainer/.../best_model.pt
bash scripts/eval_motif.sh chkpts/graphormer_explainer/.../best_model.pt
bash scripts/eval_visualize.sh chkpts/graphormer_explainer/.../best_model.pt

# Step 4: View results
bash scripts/view_results.sh
```

## Time Estimates

### Quick Pipeline:
- Dataset generation: ~1 minute
- Training: ~5 minutes
- Evaluation: ~2 minutes
- **Total: ~10 minutes**

### Full Pipeline:
- Dataset generation: ~5 minutes
- Training (all models): ~2-4 hours (depending on GPU)
- Evaluation: ~30 minutes
- **Total: ~3-5 hours**

## Troubleshooting

### Out of Memory
Reduce batch size or dataset size:
```bash
bash scripts/run_full_pipeline.sh synthetic 5000 1000 2000 50 16 0
```

### Training Takes Too Long
Reduce epochs or dataset size:
```bash
bash scripts/run_full_pipeline.sh synthetic 2000 500 1000 20 32 0
```

### Skip Baseline Training
Comment out the baseline training section in `run_full_pipeline.sh` if you don't need GCN/GIN.

## Next Steps After Pipeline

1. **View Results:**
   ```bash
   bash scripts/view_results.sh
   ```

2. **Run Ablations:**
   ```bash
   python scripts/run_ablations.py --experiment all
   ```

3. **Compare Experiments:**
   ```bash
   python scripts/evaluate_results.py --checkpoint_dir chkpts/ --compare exp1 exp2
   ```

4. **Generate Report:**
   - Check `results/accuracy/comparison.csv` for accuracy comparison
   - View `results/fidelity_curves/` plots
   - Review `results/motif_detection/results.csv` for explanation quality

## Tips

- **Start with quick pipeline** to verify everything works
- **Run full pipeline overnight** or when you have time
- **Check GPU usage** with `nvidia-smi` during training
- **Monitor progress** by checking `chkpts/` directory
- **Save intermediate results** - pipeline continues even if one step fails
