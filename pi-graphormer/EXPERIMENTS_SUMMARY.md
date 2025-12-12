# Experiments Summary: What to Run

## 🎯 Goal

Evaluate Graphormer-PIGNN v2 with the simplified architecture:
- **Graphormer as predictor** (using graph representation)
- **Edge scorer as explainer** (using node representations)
- **Sparsity regularization** (L1 or entropy)

## 📋 Recommended Experiment Sequence

### Phase 1: Quick Validation (Start Here!) ⭐

**Purpose**: Verify everything works

```bash
bash scripts/quick_start.sh
```

**Expected Time**: ~5-10 minutes  
**What it does**:
- Generates small dataset (1000/200/200)
- Trains small model (2 layers, 64 dim)
- Tests regularization

---

### Phase 2: Main Experiments

#### Experiment A: Baseline (No Regularization)

**Purpose**: Establish baseline performance

```bash
python main/train_v2.py \
    --dataset synthetic \
    --n_train 5000 --n_val 1000 --n_test 2000 \
    --epochs 50 \
    --batch_size 32 \
    --num_layers 4 \
    --embedding_dim 128 \
    --num_heads 4 \
    --lr 1e-4 \
    --save_dir chkpts/exp_baseline
```

**Expected Results**:
- Prediction accuracy: ~90-95%
- Explanation sparsity: Low (many edges important)
- Explanation ROC-AUC: ~0.7-0.8

---

#### Experiment B: L1 Sparsity Regularization

**Purpose**: Test sparsity regularization

```bash
python main/train_v2.py \
    --dataset synthetic \
    --n_train 5000 --n_val 1000 --n_test 2000 \
    --epochs 50 \
    --batch_size 32 \
    --num_layers 4 \
    --embedding_dim 128 \
    --num_heads 4 \
    --lr 1e-4 \
    --use_regularization \
    --reg_weight 0.01 \
    --save_dir chkpts/exp_l1_0.01
```

**Expected Results**:
- Prediction accuracy: ~88-93% (slightly lower)
- Explanation sparsity: High (few edges important)
- Explanation ROC-AUC: ~0.75-0.85

---

#### Experiment C: Different Regularization Weights

**Purpose**: Find optimal regularization strength

```bash
# λ = 0.001 (weak)
python main/train_v2.py ... --reg_weight 0.001 --save_dir chkpts/exp_l1_0.001

# λ = 0.01 (medium) - already done in Experiment B
# λ = 0.05 (strong)
python main/train_v2.py ... --reg_weight 0.05 --save_dir chkpts/exp_l1_0.05

# λ = 0.1 (very strong)
python main/train_v2.py ... --reg_weight 0.1 --save_dir chkpts/exp_l1_0.1
```

**Compare**: Which λ gives best balance of accuracy and sparsity?

---

### Phase 3: Downstream Tasks (Optional)

#### Experiment D: Fine-tuning on BA-2Motif

**Purpose**: Test transfer learning

```bash
# First, get best pre-trained model from Phase 2
PRETRAINED="chkpts/exp_l1_0.01/best_model.pt"

# Fine-tune
python main/train_v2.py \
    --dataset ba2motif \
    --data_dir data/BA2Motif \
    --pretrained_path ${PRETRAINED} \
    --epochs 30 \
    --use_regularization \
    --reg_weight 0.01 \
    --save_dir chkpts/exp_ba2motif_finetune
```

**Compare**: Fine-tuning vs training from scratch

---

## 📊 Evaluation

After running experiments, compare results:

```bash
python scripts/evaluate_results.py --checkpoint_dir chkpts/
```

This will show:
- Prediction accuracy for each experiment
- Explanation quality metrics
- Model configurations
- Best epochs

---

## 🎓 For Course Project

### Minimum Required Experiments:

1. ✅ **Baseline** (no regularization)
2. ✅ **With Regularization** (L1, λ=0.01)
3. ✅ **Comparison** (show regularization improves sparsity)

### Nice to Have:

4. **Hyperparameter Sweep** (different λ values)
5. **Downstream Task** (fine-tuning on BA-2Motif)
6. **Ablation Study** (L1 vs entropy vs combined)

---

## ⏱️ Time Estimates

| Experiment | Time (GPU) | Time (CPU) |
|------------|------------|------------|
| Quick Start | 5-10 min | 30-60 min |
| Baseline | 30-60 min | 2-4 hours |
| With Regularization | 30-60 min | 2-4 hours |
| Hyperparameter Sweep | 2-4 hours | 8-16 hours |
| Fine-tuning | 15-30 min | 1-2 hours |

---

## 🚀 Quick Commands

```bash
# 1. Quick test
bash scripts/quick_start.sh

# 2. Baseline experiment
python main/train_v2.py --dataset synthetic --n_train 5000 --epochs 50 --save_dir chkpts/baseline

# 3. With regularization
python main/train_v2.py --dataset synthetic --n_train 5000 --epochs 50 --use_regularization --reg_weight 0.01 --save_dir chkpts/regularized

# 4. Compare results
python scripts/evaluate_results.py --checkpoint_dir chkpts/
```

---

## 📝 What to Report

For each experiment, report:

1. **Prediction Performance**:
   - Accuracy on test set
   - Confusion matrix (if multi-class)

2. **Explanation Quality**:
   - ROC-AUC (if ground truth available)
   - Sparsity (fraction of edges with score > 0.5)
   - Average edge score magnitude

3. **Comparison**:
   - Baseline vs Regularized
   - Different regularization weights
   - Training curves (loss over epochs)

4. **Visualizations**:
   - Example graphs with highlighted important edges
   - Score distribution histograms
   - Training curves

---

## 🎯 Success Criteria

Your experiments are successful if:

- ✅ Baseline achieves >85% accuracy
- ✅ Regularization produces sparse explanations (<20% edges important)
- ✅ Explanation ROC-AUC >0.7 (if ground truth available)
- ✅ Model converges within reasonable time
- ✅ Results are reproducible (same seed)

---

## 📚 Next Steps

1. **Start**: Run `bash scripts/quick_start.sh`
2. **Baseline**: Run Experiment A
3. **Regularized**: Run Experiment B
4. **Compare**: Use evaluation script
5. **Report**: Document findings

Good luck! 🎉
