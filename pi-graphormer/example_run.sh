#!/bin/bash
# Example script demonstrating the complete workflow
# This is a minimal example - adjust parameters as needed

set -e  # Exit on error

echo "=========================================="
echo "Example: Complete Workflow"
echo "=========================================="

# Step 1: Quick test training (5 epochs, small dataset)
echo ""
echo "Step 1: Quick Test Training"
echo "----------------------------------------"
python main/train_v2.py \
    --dataset synthetic \
    --n_train 500 \
    --n_val 100 \
    --n_test 200 \
    --epochs 5 \
    --batch_size 16 \
    --num_layers 2 \
    --embedding_dim 64 \
    --save_dir chkpts/example_test

echo ""
echo "✓ Training complete!"

# Step 2: Find the trained model
echo ""
echo "Step 2: Finding Trained Model"
echo "----------------------------------------"
MODEL_PATH=$(find chkpts/example_test -name "best_model.pt" | head -1)
if [ -z "$MODEL_PATH" ]; then
    echo "❌ Model not found. Training may have failed."
    exit 1
fi
echo "✓ Found model: $MODEL_PATH"

# Step 3: Evaluate predictive accuracy
echo ""
echo "Step 3: Evaluating Predictive Accuracy"
echo "----------------------------------------"
python scripts/eval_predictive_accuracy.py \
    --checkpoint_dir chkpts/example_test \
    --dataset synthetic \
    --n_test 100 \
    --output results/example_accuracy.csv

echo ""
echo "✓ Accuracy evaluation complete!"

# Step 4: Visualize explanations (if model has explainer)
echo ""
echo "Step 4: Visualizing Explanations"
echo "----------------------------------------"
if [ -f "$MODEL_PATH" ]; then
    python scripts/visualize_explanations.py \
        --model_path "$MODEL_PATH" \
        --num_samples 3 \
        --n_test 20 \
        --output_dir results/example_visualizations
    
    echo ""
    echo "✓ Visualization complete!"
else
    echo "⚠ Skipping visualization (model not found)"
fi

echo ""
echo "=========================================="
echo "Example Workflow Complete!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - Accuracy: results/example_accuracy.csv"
echo "  - Visualizations: results/example_visualizations/"
echo ""
echo "Next steps:"
echo "  1. Check results/example_accuracy.csv"
echo "  2. View visualizations in results/example_visualizations/"
echo "  3. Run full training with more epochs:"
echo "     python main/train_v2.py --dataset synthetic --epochs 50 --save_dir chkpts/full_training"
echo ""
