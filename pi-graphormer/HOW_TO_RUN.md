
## Step-by-Step Training

### Step 1: Prepare Dataset

The synthetic dataset will be generated automatically, but you can also generate it manually:

```bash
# Generate synthetic dataset (optional - auto-generated if missing)
bash scripts/generate_dataset.sh 5000 1000 2000

# Or use defaults (5000/1000/2000)
bash scripts/generate_dataset.sh
```

### Step 2: Train Your First Model

#### A. Pure Graphormer (Baseline - No Explainer)

```bash
bash scripts/train_baseline.sh
```

**What this does:**
- Trains Graphormer without explainer
- Saves checkpoints every 10 epochs
- Saves best model to `chkpts/pure_graphormer/best_model.pt`

#### B. Graphormer with Explainer (No Regularization)

```bash
bash scripts/train_explainer.sh
```

**What this does:**
- Trains Graphormer with edge-scoring explainer
- No regularization (just prediction + explanation losses)
- Saves to `chkpts/graphormer_explainer/`

#### C. Graphormer with Explainer + Regularization

```bash
# Default regularization weight (0.01)
bash scripts/train_with_reg.sh

# Custom regularization weight
bash scripts/train_with_reg.sh 0.05
```

**What this does:**
- Adds sparsity regularization (encourages fewer important edges)
- First argument controls regularization strength (default: 0.01)

#### D. Graphormer with Pattern Dictionary (PI-GNN Style)

```bash
bash scripts/train_pattern_dict.sh
```

**What this does:**
- Uses sparse dictionary/autoencoder for explanations
- Learns structural patterns (atoms)
- More interpretable than simple MLP edge scorer

#### E. Quick Test (5 minutes)

```bash
bash scripts/train_quick_test.sh
```

**What this does:**
- Minimal training to verify everything works
- Small dataset (500/100/200), 5 epochs, small model

### Step 3: Monitor Training

During training, you'll see output like:

```
Epoch 1/50
Train Loss: 0.693 | Pred Loss: 0.693 | Exp Loss: 0.500 | Reg Loss: 0.001
Train Acc: 0.520 | Train Exp ROC-AUC: 0.550
Val Loss: 0.691 | Val Acc: 0.530 | Val Exp ROC-AUC: 0.560
Best Val Acc: 0.530 (improved)
Saving checkpoint...
```

**Key metrics:**
- **Pred Loss**: Prediction loss (lower is better)
- **Exp Loss**: Explanation loss (lower is better)
- **Reg Loss**: Regularization loss (controls sparsity)
- **Acc**: Accuracy (higher is better)
- **ROC-AUC**: Explanation quality (higher is better, max 1.0)

### Step 4: Check Results

After training completes:

```bash
# View saved results
bash scripts/view_results.sh chkpts/graphormer_explainer

# Or view all experiments
bash scripts/view_results.sh
```

---

## Running Evaluations

### Experiment 1: Predictive Accuracy Comparison

Compare all models (Pure Graphormer, Graphormer+Explainer, GCN, GIN):

```bash
# Use defaults (chkpts/, 500 test samples)
bash scripts/eval_accuracy.sh

# Custom checkpoint directory and test size
bash scripts/eval_accuracy.sh chkpts/ 1000 results/my_accuracy.csv
```

**Output:** CSV file with accuracy, F1-score, precision, recall for each model

### Experiment 2: Fidelity Curves

Measure how robust predictions are to edge removal:

```bash
# Auto-find model (uses first best_model.pt found)
bash scripts/eval_fidelity.sh

# Specify model path
bash scripts/eval_fidelity.sh chkpts/my_experiment/best_model.pt

# Custom test size and output directory
bash scripts/eval_fidelity.sh chkpts/my_experiment/best_model.pt 500 results/my_fidelity
```

**Output:** 
- `deletion_curve.png`: Accuracy vs % edges removed
- `insertion_curve.png`: Accuracy vs % edges added
- `results.csv`: Raw data

### Experiment 3: Motif Detection

Evaluate how well explanations identify ground-truth motifs:

```bash
# Auto-find model
bash scripts/eval_motif.sh

# Specify model path
bash scripts/eval_motif.sh chkpts/my_experiment/best_model.pt

# Custom test size and output
bash scripts/eval_motif.sh chkpts/my_experiment/best_model.pt 1000 results/my_motif.csv
```

**Output:** CSV with Precision@k, Recall@k, F1@k for different k values

### Experiment 4: Qualitative Visualization

Visualize explanations on sample graphs:

```bash
# Auto-find model, 5 samples
bash scripts/eval_visualize.sh

# Specify model path
bash scripts/eval_visualize.sh chkpts/my_experiment/best_model.pt

# Custom number of samples
bash scripts/eval_visualize.sh chkpts/my_experiment/best_model.pt 10
```

**Output:** PNG images showing ground truth vs predicted important edges

### Experiment 5: Ablation Studies

Test different hyperparameters:

```bash
# Test sparsity coefficient
python scripts/run_ablations.py \
    --experiment sparsity \
    --output_dir results/ablations \
    --n_test 200 \
    --epochs 10

# Test edge scoring heads
python scripts/run_ablations.py \
    --experiment edge_scoring \
    --output_dir results/ablations \
    --n_test 200 \
    --epochs 10

# Run all ablations
python scripts/run_ablations.py \
    --experiment all \
    --output_dir results/ablations
```

### Run All Evaluations at Once

```bash
bash scripts/run_all_evaluations.sh
```

This runs all 5 experiments sequentially and saves results to `results/`.

---

## Understanding Outputs

### Training Outputs

After training, checkpoints are saved in `chkpts/experiment_name/`:

```
chkpts/experiment_name/
├── best_model.pt              # Best model (lowest validation loss)
├── checkpoint_epoch10.pt      # Checkpoint at epoch 10
├── checkpoint_epoch20.pt      # Checkpoint at epoch 20
├── ...
└── results.pt                  # Test metrics and config
```

**Load a trained model:**

```python
import torch
from model_v2 import GraphormerExplainer

# Load checkpoint
checkpoint = torch.load('chkpts/graphormer_explainer/best_model.pt')
model = GraphormerExplainer(...)  # Initialize with same args
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

### Evaluation Outputs

Results are saved in `results/`:

```
results/
├── accuracy/
│   ├── comparison.csv         # Accuracy comparison table
│   └── log.txt                # Log file
├── fidelity_curves/
│   ├── deletion_curve.png     # Deletion fidelity plot
│   ├── insertion_curve.png    # Insertion fidelity plot
│   └── results.csv            # Raw data
├── motif_detection/
│   ├── results.csv            # Precision/Recall/F1 metrics
│   └── log.txt
├── visualizations/
│   ├── graph_0.png            # Visualization for graph 0
│   ├── graph_1.png
│   └── ...
└── ablations/
    └── ablation_results.csv   # Ablation study results
```

---

## Common Commands Reference

### Training Commands

```bash
# Quick test (5 minutes)
bash scripts/train_quick_test.sh

# Baseline (no explainer)
bash scripts/train_baseline.sh

# Standard training (with explainer)
bash scripts/train_explainer.sh

# With regularization
bash scripts/train_with_reg.sh          # Default: reg_weight=0.01
bash scripts/train_with_reg.sh 0.05     # Custom reg_weight

# With pattern dictionary
bash scripts/train_pattern_dict.sh

# Generate dataset manually
bash scripts/generate_dataset.sh
```

### Evaluation Commands

```bash
# Find trained model
find chkpts -name "best_model.pt"

# Predictive accuracy
bash scripts/eval_accuracy.sh

# Fidelity curves
bash scripts/eval_fidelity.sh                                    # Auto-find model
bash scripts/eval_fidelity.sh chkpts/my_experiment/best_model.pt  # Specify model

# Motif detection
bash scripts/eval_motif.sh                                      # Auto-find model
bash scripts/eval_motif.sh chkpts/my_experiment/best_model.pt    # Specify model

# Visualization
bash scripts/eval_visualize.sh                                  # Auto-find model, 5 samples
bash scripts/eval_visualize.sh chkpts/my_experiment/best_model.pt 10  # Specify model, 10 samples

# All evaluations
bash scripts/run_all_evaluations.sh
```

### Useful Utility Commands

```bash
# View results
bash scripts/view_results.sh                    # View all experiments
bash scripts/view_results.sh chkpts/ my_exp     # View specific experiment

# Check GPU usage
nvidia-smi

# Monitor training (in another terminal)
watch -n 1 nvidia-smi
```

---

## Troubleshooting

### Issue 1: "CUDA out of memory"

**Solution:** Reduce batch size or model size

```bash
# Reduce batch size
--batch_size 16  # instead of 32

# Reduce embedding dimension
--embedding_dim 64  # instead of 128

# Reduce number of layers
--num_layers 2  # instead of 4
```

### Issue 2: "Dataset not found"

**Solution:** Generate dataset first

```bash
python generate_ptmotifs.py \
    --output_dir data/PT-Motifs/raw \
    --n_train 1000
```

Or the dataset will be auto-generated when you run training.

### Issue 3: "Module not found" or ImportError

**Solution:** Make sure you're in the right directory

```bash
cd /teamspace/studios/this_studio/course_project_machine_learning_with_graphs/pi-graphormer
python main/train_v2.py ...
```

### Issue 4: "Model path not found" in evaluation

**Solution:** Find the correct model path

```bash
# List all checkpoints
find chkpts -name "best_model.pt"

# Use the full path
python scripts/eval_fidelity_curves.py \
    --model_path chkpts/graphormer_explainer/synthetic_20251212_044249/best_model.pt
```

### Issue 5: Training is too slow

**Solutions:**

1. **Use GPU** (if available):
   ```bash
   # Check GPU
   python -c "import torch; print(torch.cuda.is_available())"
   
   # Training will use GPU automatically if available
   ```

2. **Reduce dataset size**:
   ```bash
   --n_train 1000  # instead of 5000
   ```

3. **Reduce epochs**:
   ```bash
   --epochs 10  # for quick testing
   ```

### Issue 6: Results don't make sense

**Check:**
- Are you using the right dataset? (`--dataset synthetic`)
- Did training complete? Check `chkpts/experiment_name/results.pt`
- Are you loading the right checkpoint?

```bash
# Check training completed successfully
python scripts/evaluate_results.py --checkpoint_dir chkpts/your_experiment
```

---

## Next Steps

1. **Start Small**: Run a quick 5-epoch test to verify everything works
2. **Train Baseline**: Train Pure Graphormer to establish baseline
3. **Train Explainer**: Train Graphormer with explainer
4. **Compare**: Run evaluation scripts to compare models
5. **Experiment**: Try different hyperparameters and regularization weights

---

## Quick Reference Card

```bash
# QUICK TEST
bash scripts/train_quick_test.sh

# TRAINING
bash scripts/train_baseline.sh          # Pure Graphormer
bash scripts/train_explainer.sh         # With explainer
bash scripts/train_with_reg.sh          # With regularization

# EVALUATION
bash scripts/eval_accuracy.sh           # Accuracy comparison
bash scripts/eval_fidelity.sh           # Fidelity curves
bash scripts/eval_motif.sh              # Motif detection
bash scripts/eval_visualize.sh          # Visualizations
bash scripts/run_all_evaluations.sh     # All experiments

# VIEW RESULTS
bash scripts/view_results.sh

# FIND MODEL
find chkpts -name "best_model.pt"
```

---

## Need Help?

- Check `QUICK_START_EXPERIMENTS.md` for minimal examples
- Check `EXPERIMENTAL_PLAN.md` for detailed experimental design
- Check `ARCHITECTURE_V2.md` for model architecture details
- Check `IMPLEMENTATION_GUIDE.md` for implementation details

Happy experimenting! 🎉
