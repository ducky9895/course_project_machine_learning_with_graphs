# Multi-Dataset Evaluation

## Problem
Previously, we only evaluated models on the `synthetic` dataset, even though we have multiple datasets available (BA2Motif, Mutag, etc.).

## Solution
Created a comprehensive multi-dataset evaluation system:

### 1. Updated `eval_predictive_accuracy.py`
- Now supports: `synthetic`, `ba2motif`, `mutag`
- Adds `Dataset` column to CSV output
- Handles dataset-specific loading and errors gracefully

### 2. Created `eval_all_datasets.py`
- Evaluates all models on all available datasets
- Combines results into a single comparison table
- Generates pivot table showing accuracy by dataset and model

### 3. Created `eval_all_datasets.sh`
- Bash wrapper for easy execution
- Runs evaluation on all datasets sequentially
- Automatically combines results

## Usage

### Evaluate all datasets:
```bash
bash scripts/eval_all_datasets.sh
```

### Evaluate specific datasets:
```bash
python scripts/eval_all_datasets.py --datasets synthetic ba2motif
```

### Evaluate single dataset (as before):
```bash
python scripts/eval_predictive_accuracy.py --dataset ba2motif
```

## Output

Results are saved to:
- `results/accuracy/synthetic_comparison.csv`
- `results/accuracy/ba2motif_comparison.csv`
- `results/accuracy/mutag_comparison.csv`
- `results/accuracy/all_datasets_comparison.csv` (combined)

## Note on 100% Accuracy

The synthetic dataset shows 100% accuracy because:
- **The dataset is deterministic**: Each motif type has a fixed edge count
  - House: 52 edges
  - Cycle: 50 edges  
  - Star: 48 edges
- Models can achieve perfect accuracy by simply counting edges
- This is a dataset issue, not a model issue

**Recommendation**: Focus on BA2Motif and Mutag results for realistic evaluation.
