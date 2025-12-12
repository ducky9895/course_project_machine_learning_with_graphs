# Scripts Reference Guide

All scripts are in the `scripts/` directory. Make them executable with `chmod +x scripts/*.sh` (already done).

## Training Scripts

### `train_quick_test.sh`
Quick test training (5 minutes) - Minimal model to verify everything works.

```bash
bash scripts/train_quick_test.sh
```

**Parameters:** Fixed (500/100/200 train/val/test, 5 epochs, small model)

---

### `train_baseline.sh`
Train Pure Graphormer (Baseline - No Explainer).

```bash
bash scripts/train_baseline.sh
```

**Parameters:** Fixed (5000/1000/2000 train/val/test, 50 epochs, standard model)

**Output:** `chkpts/pure_graphormer/`

---

### `train_explainer.sh`
Train Graphormer with Explainer (No Regularization).

```bash
bash scripts/train_explainer.sh
```

**Parameters:** Fixed (5000/1000/2000 train/val/test, 50 epochs, standard model)

**Output:** `chkpts/graphormer_explainer/`

---

### `train_with_reg.sh [reg_weight]`
Train Graphormer with Explainer + Regularization.

```bash
bash scripts/train_with_reg.sh          # Default: reg_weight=0.01
bash scripts/train_with_reg.sh 0.05    # Custom reg_weight
```

**Parameters:**
- `reg_weight` (optional): Regularization weight (default: 0.01)

**Output:** `chkpts/graphormer_explainer_reg_<reg_weight>/`

---

### `train_pattern_dict.sh`
Train Graphormer with Pattern Dictionary (PI-GNN Style).

```bash
bash scripts/train_pattern_dict.sh
```

**Parameters:** Fixed (5000/1000/2000 train/val/test, 50 epochs, pattern dictionary enabled)

**Output:** `chkpts/graphormer_pattern_dict/`

---

### `generate_dataset.sh [n_train] [n_val] [n_test]`
Generate synthetic dataset manually (optional - auto-generated if missing).

```bash
bash scripts/generate_dataset.sh                    # Default: 5000/1000/2000
bash scripts/generate_dataset.sh 1000 200 400       # Custom sizes
```

**Parameters:**
- `n_train` (optional): Training samples (default: 5000)
- `n_val` (optional): Validation samples (default: 1000)
- `n_test` (optional): Test samples (default: 2000)

**Output:** `data/PT-Motifs/raw/`

---

## Evaluation Scripts

### `eval_accuracy.sh [checkpoint_dir] [n_test] [output]`
Experiment 1: Predictive Accuracy Comparison.

```bash
bash scripts/eval_accuracy.sh                                    # Defaults
bash scripts/eval_accuracy.sh chkpts/ 1000 results/my_acc.csv  # Custom
```

**Parameters:**
- `checkpoint_dir` (optional): Directory with checkpoints (default: `chkpts/`)
- `n_test` (optional): Number of test samples (default: 500)
- `output` (optional): Output CSV file (default: `results/accuracy_comparison.csv`)

**Output:** CSV file with accuracy, F1-score, precision, recall for each model

---

### `eval_fidelity.sh [model_path] [n_test] [output_dir]`
Experiment 2: Fidelity Curves.

```bash
bash scripts/eval_fidelity.sh                                    # Auto-find model
bash scripts/eval_fidelity.sh chkpts/my_exp/best_model.pt       # Specify model
bash scripts/eval_fidelity.sh chkpts/my_exp/best_model.pt 500 results/my_fidelity  # Full custom
```

**Parameters:**
- `model_path` (optional): Path to `best_model.pt` (auto-finds if not provided)
- `n_test` (optional): Number of test samples (default: 200)
- `output_dir` (optional): Output directory (default: `results/fidelity_curves`)

**Output:** 
- `deletion_curve.png`: Accuracy vs % edges removed
- `insertion_curve.png`: Accuracy vs % edges added
- `results.csv`: Raw data

---

### `eval_motif.sh [model_path] [n_test] [output]`
Experiment 3: Motif Detection.

```bash
bash scripts/eval_motif.sh                                      # Auto-find model
bash scripts/eval_motif.sh chkpts/my_exp/best_model.pt         # Specify model
bash scripts/eval_motif.sh chkpts/my_exp/best_model.pt 1000 results/my_motif.csv  # Full custom
```

**Parameters:**
- `model_path` (optional): Path to `best_model.pt` (auto-finds if not provided)
- `n_test` (optional): Number of test samples (default: 500)
- `output` (optional): Output CSV file (default: `results/motif_detection.csv`)

**Output:** CSV with Precision@k, Recall@k, F1@k for different k values

---

### `eval_visualize.sh [model_path] [num_samples] [n_test] [output_dir]`
Experiment 4: Qualitative Visualization.

```bash
bash scripts/eval_visualize.sh                                  # Auto-find model, 5 samples
bash scripts/eval_visualize.sh chkpts/my_exp/best_model.pt     # Specify model
bash scripts/eval_visualize.sh chkpts/my_exp/best_model.pt 10  # 10 samples
```

**Parameters:**
- `model_path` (optional): Path to `best_model.pt` (auto-finds if not provided)
- `num_samples` (optional): Number of graphs to visualize (default: 5)
- `n_test` (optional): Number of test samples to use (default: 50)
- `output_dir` (optional): Output directory (default: `results/visualizations`)

**Output:** PNG images showing ground truth vs predicted important edges

---

### `run_ablations.py`
Experiment 5: Ablation Studies (Python script, not shell).

```bash
python scripts/run_ablations.py --experiment all --output_dir results/ablations
python scripts/run_ablations.py --experiment sparsity --output_dir results/ablations --n_test 200 --epochs 10
```

**Parameters:** See `python scripts/run_ablations.py --help`

---

### `run_all_evaluations.sh`
Run all 5 evaluation experiments sequentially.

```bash
bash scripts/run_all_evaluations.sh
```

**Output:** All results saved to `results/` directory

---

## Utility Scripts

### `view_results.sh [checkpoint_dir] [experiment]`
View training results from checkpoints.

```bash
bash scripts/view_results.sh                    # View all experiments
bash scripts/view_results.sh chkpts/ my_exp   # View specific experiment
```

**Parameters:**
- `checkpoint_dir` (optional): Directory with checkpoints (default: `chkpts/`)
- `experiment` (optional): Specific experiment name to view

---

## Common Workflows

### Workflow 1: Quick Test
```bash
bash scripts/train_quick_test.sh
bash scripts/view_results.sh chkpts/quick_test
```

### Workflow 2: Compare Baseline vs Explainer
```bash
bash scripts/train_baseline.sh
bash scripts/train_explainer.sh
bash scripts/eval_accuracy.sh
```

### Workflow 3: Full Evaluation Pipeline
```bash
bash scripts/train_explainer.sh
bash scripts/run_all_evaluations.sh
```

### Workflow 4: Test Different Regularization Weights
```bash
bash scripts/train_with_reg.sh 0.01
bash scripts/train_with_reg.sh 0.05
bash scripts/train_with_reg.sh 0.1
bash scripts/eval_accuracy.sh
```

---

## Notes

- All scripts use `set -e` to exit on error
- Scripts auto-create output directories if needed
- Model paths can be auto-detected (finds first `best_model.pt`)
- Check script help with `bash scripts/<script_name>.sh` (some show usage on error)
