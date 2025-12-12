#!/usr/bin/env python
"""
Experiment 1: Predictive Accuracy Comparison

Compare prediction accuracy across:
- Pure Graphormer
- Graphormer + Explainer
- Graphormer + Explainer + Regularization
- GCN
- GIN
"""

import os
import sys
import argparse
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_v2 import PureGraphormer, GraphormerExplainer
from baselines.gcn import create_gcn_model
from baselines.gin import create_gin_model
from data_utils import GraphormerCollator
from dataset.datasets import BA2Motif, MutagDataset, generate_synthetic_dataset
from torch.utils.data import DataLoader
from torch_geometric.data import DataLoader as PyGDataLoader


def evaluate_graphormer_model(model, loader, device):
    """Evaluate Graphormer-based model."""
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            graphormer_data = {k: v.to(device) for k, v in batch['graphormer_data'].items()}
            pyg_batch = batch['pyg_batch'].to(device)
            
            if isinstance(model, PureGraphormer):
                logits = model(graphormer_data, pyg_batch)
            else:
                # GraphormerExplainer returns only logits when return_explanation=False
                logits = model(graphormer_data, pyg_batch, return_explanation=False)
            
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(pyg_batch.y.cpu())
    
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1_macro': f1_score(all_labels, all_preds, average='macro'),
        'f1_micro': f1_score(all_labels, all_preds, average='micro'),
        'precision': precision_score(all_labels, all_preds, average='macro', zero_division=0),
        'recall': recall_score(all_labels, all_preds, average='macro', zero_division=0),
    }
    
    return metrics


def evaluate_gnn_model(model, loader, device):
    """Evaluate GCN/GIN model."""
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            try:
                batch = batch.to(device)
                
                # Ensure batch attribute exists (PyG DataLoader should create this, but check)
                if not hasattr(batch, 'batch') or batch.batch is None:
                    # If batch_size=1, create batch index manually
                    if batch.x.size(0) > 0:
                        batch.batch = torch.zeros(batch.x.size(0), dtype=torch.long, device=device)
                    else:
                        print(f"  Warning: Empty graph, skipping")
                        continue
                
                # Verify node features have consistent dimension
                if batch.x.dim() != 2:
                    print(f"  Warning: Unexpected node feature shape {batch.x.shape}, skipping")
                    continue
                
                # Forward pass
                logits = model(batch.x, batch.edge_index, batch.batch)
                
                # Handle single graph vs batched graphs
                if logits.dim() == 1:
                    logits = logits.unsqueeze(0)
                
                preds = logits.argmax(dim=1)
                
                all_preds.append(preds.cpu())
                all_labels.append(batch.y.cpu())
            except Exception as e:
                print(f"  Warning: Error processing batch: {e}")
                import traceback
                traceback.print_exc()
                # Skip this batch
                continue
    
    if len(all_preds) == 0:
        return {
            'accuracy': 0.0,
            'f1_macro': 0.0,
            'f1_micro': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'error': 'No batches processed successfully'
        }
    
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1_macro': f1_score(all_labels, all_preds, average='macro'),
        'f1_micro': f1_score(all_labels, all_preds, average='micro'),
        'precision': precision_score(all_labels, all_preds, average='macro', zero_division=0),
        'recall': recall_score(all_labels, all_preds, average='macro', zero_division=0),
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate Predictive Accuracy')
    parser.add_argument('--dataset', type=str, default='synthetic',
                       choices=['synthetic', 'ba2motif', 'mutag', 'all'])
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--checkpoint_dir', type=str, default='chkpts/',
                       help='Directory with trained models')
    parser.add_argument('--n_test', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--output', type=str, default='results/accuracy_comparison.csv')
    
    args = parser.parse_args()
    
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create test dataset
    collator = GraphormerCollator()
    
    if args.dataset == 'synthetic':
        print("Generating synthetic test dataset...")
        test_graphs = generate_synthetic_dataset(
            n_graphs=args.n_test,
            motif_types=['house', 'cycle', 'star'],
            base_nodes=20
        )
        test_loader = DataLoader(test_graphs, batch_size=args.batch_size,
                                shuffle=False, collate_fn=collator)
        # Use batch_size=1 for PyG to avoid batching issues with variable-sized graphs
        # PyG can batch variable-sized graphs, but batch_size=1 is safest
        pyg_test_loader = PyGDataLoader(test_graphs, batch_size=1, shuffle=False)
        num_classes = 3
        input_dim = 4  # Synthetic node features
    elif args.dataset == 'ba2motif':
        print("Loading BA-2Motif test dataset...")
        try:
            test_dataset = BA2Motif(args.data_dir, mode='test')
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                                    shuffle=False, collate_fn=collator)
            pyg_test_loader = PyGDataLoader(test_dataset, batch_size=1, shuffle=False)
            num_classes = 2
            input_dim = test_dataset[0].x.size(1) if len(test_dataset) > 0 else 10
        except Exception as e:
            print(f"  Error loading BA2Motif: {e}")
            print("  Skipping BA2Motif evaluation")
            return {}
    elif args.dataset == 'mutag':
        print("Loading Mutag test dataset...")
        try:
            test_dataset = MutagDataset(args.data_dir, mode='test')
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                                    shuffle=False, collate_fn=collator)
            pyg_test_loader = PyGDataLoader(test_dataset, batch_size=1, shuffle=False)
            num_classes = 2
            input_dim = test_dataset[0].x.size(1) if len(test_dataset) > 0 else 10
        except Exception as e:
            print(f"  Error loading Mutag: {e}")
            print("  Skipping Mutag evaluation")
            return {}
    else:
        print(f"Unknown dataset: {args.dataset}")
        return {}
    
    results = {}
    
    # 1. Pure Graphormer
    print("\n" + "="*60)
    print("1. Evaluating Pure Graphormer...")
    print("="*60)
    # Try direct path first
    pure_model_path = os.path.join(args.checkpoint_dir, 'pure_graphormer', 'best_model.pt')
    if not os.path.exists(pure_model_path):
        # Try to find in subdirectories
        import glob
        matches = glob.glob(os.path.join(args.checkpoint_dir, 'pure_graphormer', '*', 'best_model.pt'))
        if matches:
            pure_model_path = matches[0]  # Use most recent (last in sorted list)
            print(f"  Found model in subdirectory: {pure_model_path}")
    
    if os.path.exists(pure_model_path):
        try:
            model = PureGraphormer(
                num_encoder_layers=4,
                embedding_dim=128,
                ffn_embedding_dim=128,
                num_attention_heads=4,
                num_in_degree=64,
                num_out_degree=64,
                num_spatial=64,
                num_edges=512,
                num_classes=num_classes,
                classifier_hidden_dim=64
            ).to(device)
            checkpoint = torch.load(pure_model_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
            results['Pure Graphormer'] = evaluate_graphormer_model(model, test_loader, device)
            print(f"  Accuracy: {results['Pure Graphormer']['accuracy']:.4f}")
        except Exception as e:
            print(f"  Error loading model: {e}")
            results['Pure Graphormer'] = {'accuracy': 0.0, 'error': str(e)}
    else:
        print(f"  Model not found at {pure_model_path}")
        results['Pure Graphormer'] = {'accuracy': 0.0, 'note': 'Model not found'}
    
    # 2. Graphormer + Explainer
    print("\n" + "="*60)
    print("2. Evaluating Graphormer + Explainer...")
    print("="*60)
    explainer_model_path = os.path.join(args.checkpoint_dir, 'graphormer_explainer', 'best_model.pt')
    if not os.path.exists(explainer_model_path):
        # Try to find in subdirectories
        import glob
        matches = glob.glob(os.path.join(args.checkpoint_dir, 'graphormer_explainer', '*', 'best_model.pt'))
        if matches:
            explainer_model_path = matches[-1]  # Use most recent (last in sorted list)
            print(f"  Found model in subdirectory: {explainer_model_path}")
    
    if os.path.exists(explainer_model_path):
        try:
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
                classifier_hidden_dim=64,
                use_regularization=False  # No regularization for basic explainer
            ).to(device)
            checkpoint = torch.load(explainer_model_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
            results['Graphormer + Explainer'] = evaluate_graphormer_model(model, test_loader, device)
            print(f"  Accuracy: {results['Graphormer + Explainer']['accuracy']:.4f}")
        except Exception as e:
            print(f"  Error loading model: {e}")
            import traceback
            traceback.print_exc()
            results['Graphormer + Explainer'] = {'accuracy': 0.0, 'error': str(e)}
    else:
        print(f"  Model not found at {explainer_model_path}")
        results['Graphormer + Explainer'] = {'accuracy': 0.0, 'note': 'Model not found'}
    
    # 2.5. Graphormer + Explainer + Regularization
    print("\n" + "="*60)
    print("2.5. Evaluating Graphormer + Explainer + Regularization...")
    print("="*60)
    reg_model_path = os.path.join(args.checkpoint_dir, 'graphormer_explainer_reg', 'best_model.pt')
    if not os.path.exists(reg_model_path):
        # Try to find in subdirectories
        import glob
        matches = glob.glob(os.path.join(args.checkpoint_dir, 'graphormer_explainer_reg', '*', 'best_model.pt'))
        if matches:
            reg_model_path = matches[-1]  # Use most recent
            print(f"  Found model in subdirectory: {reg_model_path}")
    
    if os.path.exists(reg_model_path):
        try:
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
                classifier_hidden_dim=64,
                use_regularization=True  # With regularization
            ).to(device)
            checkpoint = torch.load(reg_model_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
            results['Graphormer + Explainer + Reg'] = evaluate_graphormer_model(model, test_loader, device)
            print(f"  Accuracy: {results['Graphormer + Explainer + Reg']['accuracy']:.4f}")
        except Exception as e:
            print(f"  Error loading model: {e}")
            results['Graphormer + Explainer + Reg'] = {'accuracy': 0.0, 'error': str(e)}
    else:
        print(f"  Model not found at {reg_model_path}")
        results['Graphormer + Explainer + Reg'] = {'accuracy': 0.0, 'note': 'Model not found'}
    
    # 3. GCN
    print("\n" + "="*60)
    print("3. Evaluating GCN...")
    print("="*60)
    gcn_model_path = os.path.join(args.checkpoint_dir, 'baselines', f'gcn_{args.dataset}_best.pt')
    if os.path.exists(gcn_model_path):
        try:
            gcn_model = create_gcn_model(input_dim, hidden_dim=64, num_classes=num_classes, num_layers=2).to(device)
            checkpoint = torch.load(gcn_model_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                gcn_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                gcn_model.load_state_dict(checkpoint)
            results['GCN'] = evaluate_gnn_model(gcn_model, pyg_test_loader, device)
            if 'error' not in results['GCN']:
                print(f"  Accuracy: {results['GCN']['accuracy']:.4f}")
            else:
                print(f"  Error: {results['GCN'].get('error', 'Unknown error')}")
        except Exception as e:
            print(f"  Error loading/evaluating GCN: {e}")
            import traceback
            traceback.print_exc()
            results['GCN'] = {'accuracy': 0.0, 'error': str(e)}
    else:
        print(f"  GCN model not found at {gcn_model_path}")
        print(f"  Train with: python baselines/train_baselines.py --model gcn --dataset {args.dataset}")
        results['GCN'] = {'accuracy': 0.0, 'note': 'Model not found - needs training'}
    
    # 4. GIN
    print("\n" + "="*60)
    print("4. Evaluating GIN...")
    print("="*60)
    gin_model_path = os.path.join(args.checkpoint_dir, 'baselines', f'gin_{args.dataset}_best.pt')
    if os.path.exists(gin_model_path):
        try:
            gin_model = create_gin_model(input_dim, hidden_dim=64, num_classes=num_classes, num_layers=2).to(device)
            checkpoint = torch.load(gin_model_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                gin_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                gin_model.load_state_dict(checkpoint)
            results['GIN'] = evaluate_gnn_model(gin_model, pyg_test_loader, device)
            if 'error' not in results['GIN']:
                print(f"  Accuracy: {results['GIN']['accuracy']:.4f}")
            else:
                print(f"  Error: {results['GIN'].get('error', 'Unknown error')}")
        except Exception as e:
            print(f"  Error loading/evaluating GIN: {e}")
            import traceback
            traceback.print_exc()
            results['GIN'] = {'accuracy': 0.0, 'error': str(e)}
    else:
        print(f"  GIN model not found at {gin_model_path}")
        print(f"  Train with: python baselines/train_baselines.py --model gin --dataset {args.dataset}")
        results['GIN'] = {'accuracy': 0.0, 'note': 'Model not found - needs training'}
    
    # Print results table
    print("\n" + "="*80)
    print("Predictive Accuracy Comparison")
    print("="*80)
    print(f"{'Model':<30} {'Accuracy':<12} {'F1-Macro':<12} {'F1-Micro':<12} {'Precision':<12} {'Recall':<12}")
    print("-"*80)
    
    for model_name, metrics in results.items():
        if 'note' in metrics or 'error' in metrics:
            note = metrics.get('note', metrics.get('error', 'N/A'))
            print(f"{model_name:<30} {note}")
        else:
            print(f"{model_name:<30} {metrics['accuracy']:<12.4f} "
                  f"{metrics['f1_macro']:<12.4f} {metrics['f1_micro']:<12.4f} "
                  f"{metrics['precision']:<12.4f} {metrics['recall']:<12.4f}")
    
    print("="*80)
    
    # Save to CSV
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df = pd.DataFrame(results).T
    df.index.name = 'Model'
    df.reset_index(inplace=True)
    df['Dataset'] = args.dataset  # Add dataset column
    df.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")
    
    return results


if __name__ == '__main__':
    main()
