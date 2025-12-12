#!/usr/bin/env python
"""
Evaluate and compare experiment results.

Usage:
    python scripts/evaluate_results.py --checkpoint_dir chkpts/
    python scripts/evaluate_results.py --checkpoint_dir chkpts/ --experiment phase1_baseline
"""

import os
import sys
import argparse
import torch
import json
from pathlib import Path
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_results(checkpoint_dir, experiment_name=None):
    """Load results from checkpoint directory."""
    checkpoint_path = Path(checkpoint_dir)
    
    results = {}
    
    if experiment_name:
        # Load specific experiment
        exp_dir = checkpoint_path / experiment_name
        results_file = exp_dir / "results.pt"
        
        if results_file.exists():
            data = torch.load(results_file, map_location='cpu')
            results[experiment_name] = {
                'test_metrics': data.get('test_metrics', {}),
                'args': data.get('args', {}),
                'best_epoch': data.get('best_epoch', 0)
            }
    else:
        # Load all experiments
        for exp_dir in checkpoint_path.iterdir():
            if exp_dir.is_dir():
                results_file = exp_dir / "results.pt"
                if results_file.exists():
                    data = torch.load(results_file, map_location='cpu')
                    results[exp_dir.name] = {
                        'test_metrics': data.get('test_metrics', {}),
                        'args': data.get('args', {}),
                        'best_epoch': data.get('best_epoch', 0)
                    }
    
    return results


def print_results_table(results):
    """Print results in a formatted table."""
    print("\n" + "="*80)
    print("Experiment Results Summary")
    print("="*80)
    
    # Header
    print(f"{'Experiment':<30} {'Pred Acc':<12} {'Exp ROC-AUC':<15} {'Exp Acc':<12} {'Best Epoch':<12}")
    print("-"*80)
    
    # Sort by prediction accuracy
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1]['test_metrics'].get('pred_accuracy', 0),
        reverse=True
    )
    
    for exp_name, data in sorted_results:
        metrics = data['test_metrics']
        pred_acc = metrics.get('pred_accuracy', 0)
        exp_roc_auc = metrics.get('exp_roc_auc', 0)
        exp_acc = metrics.get('exp_accuracy', 0)
        best_epoch = data.get('best_epoch', 0)
        
        print(f"{exp_name:<30} {pred_acc:<12.4f} {exp_roc_auc:<15.4f} {exp_acc:<12.4f} {best_epoch:<12}")
    
    print("="*80)


def print_detailed_results(results, experiment_name):
    """Print detailed results for a specific experiment."""
    if experiment_name not in results:
        print(f"Experiment '{experiment_name}' not found.")
        return
    
    data = results[experiment_name]
    metrics = data['test_metrics']
    args = data['args']
    
    print("\n" + "="*80)
    print(f"Detailed Results: {experiment_name}")
    print("="*80)
    
    print("\nTest Metrics:")
    print(f"  Prediction Accuracy: {metrics.get('pred_accuracy', 0):.4f}")
    print(f"  Prediction Loss: {metrics.get('pred_loss', 0):.4f}")
    
    if 'exp_roc_auc' in metrics:
        print(f"\nExplanation Metrics:")
        print(f"  ROC-AUC: {metrics.get('exp_roc_auc', 0):.4f}")
        print(f"  Accuracy: {metrics.get('exp_accuracy', 0):.4f}")
        print(f"  Precision: {metrics.get('exp_precision', 0):.4f}")
        print(f"  Recall: {metrics.get('exp_recall', 0):.4f}")
    
    print(f"\nModel Configuration:")
    print(f"  Layers: {args.get('num_layers', 'N/A')}")
    print(f"  Embedding Dim: {args.get('embedding_dim', 'N/A')}")
    print(f"  Attention Heads: {args.get('num_heads', 'N/A')}")
    print(f"  Regularization: {args.get('use_regularization', False)}")
    if args.get('use_regularization'):
        print(f"  Reg Weight: {args.get('reg_weight', 'N/A')}")
    
    print(f"\nTraining Configuration:")
    print(f"  Best Epoch: {data.get('best_epoch', 0)}")
    print(f"  Learning Rate: {args.get('lr', 'N/A')}")
    print(f"  Batch Size: {args.get('batch_size', 'N/A')}")
    print(f"  Epochs: {args.get('epochs', 'N/A')}")
    
    print("="*80)


def compare_experiments(results, exp_names):
    """Compare multiple experiments."""
    print("\n" + "="*80)
    print("Experiment Comparison")
    print("="*80)
    
    for exp_name in exp_names:
        if exp_name not in results:
            print(f"Warning: '{exp_name}' not found. Skipping.")
            continue
        
        data = results[exp_name]
        metrics = data['test_metrics']
        
        print(f"\n{exp_name}:")
        print(f"  Prediction Accuracy: {metrics.get('pred_accuracy', 0):.4f}")
        print(f"  Explanation ROC-AUC: {metrics.get('exp_roc_auc', 0):.4f}")
        print(f"  Best Epoch: {data.get('best_epoch', 0)}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate experiment results')
    parser.add_argument('--checkpoint_dir', type=str, default='chkpts/',
                       help='Directory containing experiment checkpoints')
    parser.add_argument('--experiment', type=str, default=None,
                       help='Specific experiment to show details for')
    parser.add_argument('--compare', type=str, nargs='+', default=None,
                       help='List of experiments to compare')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for JSON results')
    
    args = parser.parse_args()
    
    # Load results
    results = load_results(args.checkpoint_dir, args.experiment)
    
    if not results:
        print(f"No results found in {args.checkpoint_dir}")
        return
    
    # Print summary table
    print_results_table(results)
    
    # Print detailed results if requested
    if args.experiment:
        print_detailed_results(results, args.experiment)
    
    # Compare experiments if requested
    if args.compare:
        compare_experiments(results, args.compare)
    
    # Save to JSON if requested
    if args.output:
        # Convert to JSON-serializable format
        json_results = {}
        for exp_name, data in results.items():
            json_results[exp_name] = {
                'test_metrics': {k: float(v) if isinstance(v, torch.Tensor) else v 
                               for k, v in data['test_metrics'].items()},
                'best_epoch': data['best_epoch'],
                'args': {k: v for k, v in data['args'].items() 
                        if isinstance(v, (int, float, str, bool, type(None)))}
            }
        
        with open(args.output, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
