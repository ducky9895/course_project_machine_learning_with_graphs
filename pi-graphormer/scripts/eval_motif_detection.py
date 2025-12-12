#!/usr/bin/env python
"""
Experiment 3: Ground Truth Motif Detection

Evaluate explanation quality on datasets with ground truth motifs.
Computes Precision@k, Recall@k, F1@k metrics.
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_v2 import GraphormerExplainer
from data_utils import GraphormerCollator
from dataset.datasets import BA2Motif, generate_synthetic_dataset
from torch.utils.data import DataLoader


def precision_at_k(y_true, y_scores, k):
    """Compute Precision@k."""
    if len(y_true) == 0:
        return 0.0
    
    top_k_indices = np.argsort(y_scores)[-k:]
    y_pred = np.zeros_like(y_true)
    y_pred[top_k_indices] = 1
    
    if y_true.sum() == 0:
        return 0.0
    
    return precision_score(y_true, y_pred, zero_division=0)


def recall_at_k(y_true, y_scores, k):
    """Compute Recall@k."""
    if len(y_true) == 0:
        return 0.0
    
    top_k_indices = np.argsort(y_scores)[-k:]
    y_pred = np.zeros_like(y_true)
    y_pred[top_k_indices] = 1
    
    if y_true.sum() == 0:
        return 0.0
    
    return recall_score(y_true, y_pred, zero_division=0)


def f1_at_k(y_true, y_scores, k):
    """Compute F1@k."""
    if len(y_true) == 0:
        return 0.0
    
    top_k_indices = np.argsort(y_scores)[-k:]
    y_pred = np.zeros_like(y_true)
    y_pred[top_k_indices] = 1
    
    if y_true.sum() == 0:
        return 0.0
    
    return f1_score(y_true, y_pred, zero_division=0)


def evaluate_motif_detection(model, loader, device, k_values=[5, 10, 20]):
    """Evaluate motif detection performance."""
    model.eval()
    
    all_scores = []
    all_labels = []
    
    print("Collecting edge scores and ground truth...")
    with torch.no_grad():
        for batch in loader:
            graphormer_data = {k: v.to(device) for k, v in batch['graphormer_data'].items()}
            pyg_batch = batch['pyg_batch'].to(device)
            edge_gt = batch['edge_gt_att']
            
            if edge_gt is not None:
                _, edge_scores = model(graphormer_data, pyg_batch, return_explanation=True)
                
                all_scores.append(edge_scores.cpu().numpy())
                all_labels.append(edge_gt.numpy())
            else:
                print("Warning: No ground truth labels found in batch")
    
    if len(all_scores) == 0:
        print("Error: No data collected")
        return None
    
    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)
    
    print(f"Total edges: {len(all_scores)}")
    print(f"Positive edges (ground truth): {all_labels.sum()}")
    
    results = {}
    for k in k_values:
        k = min(k, len(all_scores))  # Don't exceed number of edges
        results[f'precision@{k}'] = precision_at_k(all_labels, all_scores, k)
        results[f'recall@{k}'] = recall_at_k(all_labels, all_scores, k)
        results[f'f1@{k}'] = f1_at_k(all_labels, all_scores, k)
    
    return results


def evaluate_random_baseline(loader, k_values=[5, 10, 20]):
    """Evaluate random baseline."""
    all_labels = []
    
    for batch in loader:
        edge_gt = batch['edge_gt_att']
        if edge_gt is not None:
            all_labels.append(edge_gt.numpy())
    
    if len(all_labels) == 0:
        return None
    
    all_labels = np.concatenate(all_labels)
    
    # Random scores
    random_scores = np.random.rand(len(all_labels))
    
    results = {}
    for k in k_values:
        k = min(k, len(all_labels))
        results[f'precision@{k}'] = precision_at_k(all_labels, random_scores, k)
        results[f'recall@{k}'] = recall_at_k(all_labels, random_scores, k)
        results[f'f1@{k}'] = f1_at_k(all_labels, random_scores, k)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate Motif Detection')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--dataset', type=str, default='synthetic',
                       choices=['synthetic', 'ba2motif', 'ba_shapes'])
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--k_values', type=int, nargs='+', default=[5, 10, 20],
                       help='k values for Precision@k, Recall@k, F1@k')
    parser.add_argument('--output', type=str, default='results/motif_detection/results.csv')
    parser.add_argument('--n_test', type=int, default=500,
                       help='Number of test graphs (for synthetic)')
    
    args = parser.parse_args()
    
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    num_classes = 3 if args.dataset == 'synthetic' else 2
    model = GraphormerExplainer(
        num_encoder_layers=4,
        embedding_dim=128,
        ffn_embedding_dim=128,
        num_attention_heads=4,
        num_in_degree=64,
        num_out_degree=64,
        num_spatial=64,
        num_edges=512,
        num_classes=num_classes,
        use_pattern_dict=False,
        edge_hidden_dim=64,
        classifier_hidden_dim=64
    ).to(device)
    
    checkpoint = torch.load(args.model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    try:
        model.load_state_dict(state_dict, strict=False)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Attempting to load compatible parameters only...")
        # Try loading only matching parameters
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        loaded_ratio = len(pretrained_dict) / len(state_dict) if len(state_dict) > 0 else 0
        print(f"Loaded {len(pretrained_dict)}/{len(state_dict)} parameters ({loaded_ratio:.1%})")
        if loaded_ratio < 0.5:
            print("ERROR: Less than 50% of parameters loaded. Model will not work correctly!")
            raise RuntimeError(f"Failed to load model: only {loaded_ratio:.1%} parameters matched")
    
    # Create dataset
    collator = GraphormerCollator()
    
    if args.dataset == 'synthetic':
        print(f"Generating {args.n_test} synthetic test graphs...")
        test_graphs = generate_synthetic_dataset(
            n_graphs=args.n_test,
            motif_types=['house', 'cycle', 'star'],
            base_nodes=20
        )
        test_loader = DataLoader(test_graphs, batch_size=args.batch_size,
                                shuffle=False, collate_fn=collator)
    elif args.dataset in ['ba2motif', 'ba_shapes']:
        print(f"Loading {args.dataset} test dataset...")
        try:
            test_dataset = BA2Motif(args.data_dir, mode='test')
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                                    shuffle=False, collate_fn=collator)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Note: BA-Shapes dataset may need to be downloaded separately")
            return
    
    # Evaluate methods
    all_results = {}
    
    # Graphormer Explainer
    print("\n" + "="*60)
    print("Evaluating Graphormer Explainer...")
    print("="*60)
    results = evaluate_motif_detection(model, test_loader, device, args.k_values)
    if results:
        all_results['Graphormer Explainer'] = results
    
    # Random baseline
    print("\n" + "="*60)
    print("Evaluating Random baseline...")
    print("="*60)
    random_results = evaluate_random_baseline(test_loader, args.k_values)
    if random_results:
        all_results['Random'] = random_results
    
    # Print results
    if all_results:
        print("\n" + "="*80)
        print("Motif Detection Results")
        print("="*80)
        
        # Create table
        methods = list(all_results.keys())
        metrics = [f'{m}@{k}' for m in ['precision', 'recall', 'f1'] for k in args.k_values]
        
        print(f"{'Method':<25}", end='')
        for metric in metrics:
            print(f"{metric:<12}", end='')
        print()
        print("-"*80)
        
        for method in methods:
            print(f"{method:<25}", end='')
            for metric in metrics:
                value = all_results[method].get(metric, 0.0)
                print(f"{value:<12.4f}", end='')
            print()
        
        print("="*80)
        
        # Save to CSV
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        df = pd.DataFrame(all_results).T
        df.to_csv(args.output)
        print(f"\nResults saved to {args.output}")
    else:
        print("No results to display")


if __name__ == '__main__':
    main()
