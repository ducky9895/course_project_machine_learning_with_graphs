#!/usr/bin/env python
"""
Experiment 2: Fidelity Curves

Plot deletion and insertion curves to measure explanation fidelity.
Deletion: Remove top-k important edges, measure accuracy drop
Insertion: Add bottom-k important edges, measure accuracy increase
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_v2 import GraphormerExplainer, PureGraphormer
from data_utils import GraphormerCollator, preprocess_pyg_item
from dataset.datasets import generate_synthetic_dataset, BA2Motif
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch


def get_edge_scores_for_batch(model, batched_data, pyg_batch, device):
    """Get edge importance scores for a batch."""
    model.eval()
    
    # PureGraphormer doesn't have explainer, return None
    if isinstance(model, PureGraphormer):
        return None
    
    with torch.no_grad():
        result = model(batched_data, pyg_batch, return_explanation=True)
        if isinstance(result, tuple) and len(result) >= 2:
            _, edge_scores = result[0], result[1]
        else:
            # If only logits returned, can't get edge scores
            return None
    return edge_scores.cpu().numpy()


def get_random_scores(num_edges):
    """Get random edge scores."""
    return np.random.rand(num_edges)


def evaluate_with_modified_edges(model, original_graphs, device, edge_masks, collator):
    """
    Evaluate model with modified edges.
    
    Args:
        original_graphs: List of original PyG Data objects (before batching)
        edge_masks: List of boolean masks (one per graph) indicating which edges to keep
        collator: GraphormerCollator instance to recompute features
    """
    model.eval()
    
    modified_graphs = []
    for graph, edge_mask in zip(original_graphs, edge_masks):
        # Get kept edges
        kept_edges = edge_mask.nonzero().squeeze(1)
        if len(kept_edges) == 0:
            # No edges - create empty graph
            modified_edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            modified_edge_index = graph.edge_index[:, kept_edges]
        
        # Create new graph with modified edges
        from torch_geometric.data import Data
        modified_graph = Data(
            x=graph.x.clone(),
            edge_index=modified_edge_index,
            y=graph.y
        )
        
        # Recompute Graphormer features for modified graph
        from data_utils import preprocess_pyg_item
        modified_graph = preprocess_pyg_item(modified_graph, max_dist=20)
        modified_graphs.append(modified_graph)
    
    # Collate modified graphs
    modified_batch = collator(modified_graphs)
    modified_batched_data = {k: v.to(device) for k, v in modified_batch['graphormer_data'].items()}
    modified_pyg_batch = modified_batch['pyg_batch'].to(device)
    
    with torch.no_grad():
        if isinstance(model, PureGraphormer):
            logits = model(modified_batched_data, modified_pyg_batch)
        else:
            logits = model(modified_batched_data, modified_pyg_batch, return_explanation=False)
        preds = logits.argmax(dim=1)
    
    return preds.cpu()


def compute_fidelity_curve(model, loader, device, method='graphormer', ratios=np.linspace(0, 1, 21), collator=None):
    """
    Compute fidelity curve by removing/adding edges.
    
    Args:
        method: 'graphormer', 'random', 'pgexplainer', 'gnnexplainer'
        collator: GraphormerCollator instance for recomputing features
    """
    # First pass: collect edge scores, labels, and original graphs
    edge_scores_list = []  # List of edge scores per graph
    labels_list = []  # List of labels per graph
    original_graphs_list = []  # List of original PyG Data objects
    
    print("Collecting edge scores and extracting graphs...")
    for batch in tqdm(loader, desc="Collecting"):
        graphormer_data = {k: v.to(device) for k, v in batch['graphormer_data'].items()}
        pyg_batch = batch['pyg_batch'].to(device)
        
        # Extract individual graphs from batch
        batch_size = pyg_batch.batch.max().item() + 1
        batch_edge_scores = None
        
        if method == 'graphormer':
            batch_edge_scores = get_edge_scores_for_batch(model, graphormer_data, pyg_batch, device)
            if batch_edge_scores is None:
                print(f"  Warning: Model doesn't have explainer, using random scores")
                batch_edge_scores = get_random_scores(pyg_batch.edge_index.size(1))
            else:
                # Debug: check if scores are meaningful
                if len(edge_scores_list) == 0:  # First batch
                    print(f"  Edge scores stats: min={batch_edge_scores.min():.4f}, max={batch_edge_scores.max():.4f}, "
                          f"mean={batch_edge_scores.mean():.4f}, std={batch_edge_scores.std():.4f}")
        elif method == 'random':
            batch_edge_scores = get_random_scores(pyg_batch.edge_index.size(1))
        else:
            batch_edge_scores = get_random_scores(pyg_batch.edge_index.size(1))
        
        # Split batch into individual graphs
        edge_idx = 0
        for graph_idx in range(batch_size):
            graph_mask = (pyg_batch.batch == graph_idx)
            graph_nodes = graph_mask.nonzero().squeeze(1)
            
            # Get edges for this graph
            graph_edge_mask = graph_mask[pyg_batch.edge_index[0]] & graph_mask[pyg_batch.edge_index[1]]
            graph_edges = pyg_batch.edge_index[:, graph_edge_mask]
            
            # Remap to 0-based indices
            node_mapping = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(graph_nodes)}
            if len(graph_edges) > 0:
                remapped_edges = torch.zeros_like(graph_edges)
                for i in range(graph_edges.size(1)):
                    remapped_edges[0, i] = node_mapping[graph_edges[0, i].item()]
                    remapped_edges[1, i] = node_mapping[graph_edges[1, i].item()]
            else:
                remapped_edges = torch.zeros((2, 0), dtype=torch.long)
            
            # Get node features (from pyg_batch.x if available, else create dummy)
            N = graph_nodes.size(0)
            if hasattr(pyg_batch, 'x') and pyg_batch.x is not None:
                x_original = pyg_batch.x[graph_nodes].clone()
            else:
                # Fallback: create from graphormer x (reverse embedding - simplified)
                x_original = torch.zeros(N, 1, dtype=torch.float)
            
            # Get edge scores for this graph
            num_graph_edges = graph_edge_mask.sum().item()
            graph_edge_scores = batch_edge_scores[edge_idx:edge_idx + num_graph_edges]
            
            # Create original graph
            from torch_geometric.data import Data
            original_graph = Data(
                x=x_original,
                edge_index=remapped_edges,
                y=pyg_batch.y[graph_idx] if pyg_batch.y.dim() == 0 else pyg_batch.y[graph_idx]
            )
            
            edge_scores_list.append(graph_edge_scores)
            labels_list.append(original_graph.y)
            original_graphs_list.append(original_graph)
            edge_idx += num_graph_edges
    
    # Compute curve for each ratio
    print(f"\nComputing fidelity curve for {method}...")
    accuracies = []
    
    for ratio in tqdm(ratios, desc="Fidelity curve"):
        batch_accuracies = []
        
        for i, (edge_scores, label, original_graph) in enumerate(
            zip(edge_scores_list, labels_list, original_graphs_list)
        ):
            num_edges = len(edge_scores)
            num_keep = int(num_edges * (1 - ratio))  # Keep bottom (1-ratio) edges
            
            if num_keep == 0:
                # No edges - random prediction
                num_classes = len(torch.unique(torch.stack([l if isinstance(l, torch.Tensor) else torch.tensor(l) for l in labels_list])))
                batch_accuracies.append(1.0 / num_classes)
                continue
            
            # Get indices of edges to keep (least important)
            keep_indices = np.argsort(edge_scores)[:num_keep]
            edge_mask = torch.zeros(num_edges, dtype=torch.bool)
            edge_mask[keep_indices] = True
            
            # Evaluate with modified edges (process single graph)
            preds = evaluate_with_modified_edges(model, [original_graph], device, [edge_mask], collator)
            pred_label = preds[0].item() if len(preds) > 0 else 0
            true_label = label.item() if isinstance(label, torch.Tensor) else label
            batch_accuracies.append(float(pred_label == true_label))
        
        accuracies.append(np.mean(batch_accuracies))
    
    return np.array(ratios), np.array(accuracies)


def compute_insertion_curve(model, loader, device, method='graphormer', ratios=np.linspace(0, 1, 21), collator=None):
    """Compute insertion curve by adding edges."""
    # Use same graph extraction logic as deletion curve
    edge_scores_list = []
    labels_list = []
    original_graphs_list = []
    
    print("Collecting edge scores and extracting graphs...")
    for batch in tqdm(loader, desc="Collecting"):
        graphormer_data = {k: v.to(device) for k, v in batch['graphormer_data'].items()}
        pyg_batch = batch['pyg_batch'].to(device)
        
        batch_size = pyg_batch.batch.max().item() + 1
        batch_edge_scores = None
        
        if method == 'graphormer':
            batch_edge_scores = get_edge_scores_for_batch(model, graphormer_data, pyg_batch, device)
            if batch_edge_scores is None:
                batch_edge_scores = get_random_scores(pyg_batch.edge_index.size(1))
        elif method == 'random':
            batch_edge_scores = get_random_scores(pyg_batch.edge_index.size(1))
        else:
            batch_edge_scores = get_random_scores(pyg_batch.edge_index.size(1))
        
        edge_idx = 0
        for graph_idx in range(batch_size):
            graph_mask = (pyg_batch.batch == graph_idx)
            graph_nodes = graph_mask.nonzero().squeeze(1)
            graph_edge_mask = graph_mask[pyg_batch.edge_index[0]] & graph_mask[pyg_batch.edge_index[1]]
            graph_edges = pyg_batch.edge_index[:, graph_edge_mask]
            
            node_mapping = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(graph_nodes)}
            if len(graph_edges) > 0:
                remapped_edges = torch.zeros_like(graph_edges)
                for i in range(graph_edges.size(1)):
                    remapped_edges[0, i] = node_mapping[graph_edges[0, i].item()]
                    remapped_edges[1, i] = node_mapping[graph_edges[1, i].item()]
            else:
                remapped_edges = torch.zeros((2, 0), dtype=torch.long)
            
            N = graph_nodes.size(0)
            if hasattr(pyg_batch, 'x') and pyg_batch.x is not None:
                x_original = pyg_batch.x[graph_nodes].clone()
            else:
                x_original = torch.zeros(N, 1, dtype=torch.float)
            
            num_graph_edges = graph_edge_mask.sum().item()
            graph_edge_scores = batch_edge_scores[edge_idx:edge_idx + num_graph_edges]
            
            from torch_geometric.data import Data
            original_graph = Data(
                x=x_original,
                edge_index=remapped_edges,
                y=pyg_batch.y[graph_idx] if pyg_batch.y.dim() == 0 else pyg_batch.y[graph_idx]
            )
            
            edge_scores_list.append(graph_edge_scores)
            labels_list.append(original_graph.y)
            original_graphs_list.append(original_graph)
            edge_idx += num_graph_edges
    
    print(f"\nComputing insertion curve for {method}...")
    accuracies = []
    
    for ratio in tqdm(ratios, desc="Insertion curve"):
        batch_accuracies = []
        
        for i, (edge_scores, label, original_graph) in enumerate(
            zip(edge_scores_list, labels_list, original_graphs_list)
        ):
            num_edges = len(edge_scores)
            num_add = int(num_edges * ratio)  # Add top ratio edges
            
            if num_add == 0:
                num_classes = len(torch.unique(torch.stack([l if isinstance(l, torch.Tensor) else torch.tensor(l) for l in labels_list])))
                batch_accuracies.append(1.0 / num_classes)
                continue
            
            # Get indices of edges to add (most important)
            add_indices = np.argsort(edge_scores)[-num_add:]
            edge_mask = torch.zeros(num_edges, dtype=torch.bool)
            edge_mask[add_indices] = True
            
            preds = evaluate_with_modified_edges(model, [original_graph], device, [edge_mask], collator)
            pred_label = preds[0].item() if len(preds) > 0 else 0
            true_label = label.item() if isinstance(label, torch.Tensor) else label
            batch_accuracies.append(float(pred_label == true_label))
        
        accuracies.append(np.mean(batch_accuracies))
    
    return np.array(ratios), np.array(accuracies)


def plot_fidelity_curves(results, output_dir):
    """Plot fidelity curves."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Deletion curve
    plt.figure(figsize=(10, 6))
    for method, (ratios, accuracies) in results['deletion'].items():
        plt.plot(ratios * 100, accuracies, label=method, marker='o', linewidth=2, markersize=6)
    plt.xlabel('% Edges Removed', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Deletion Curve (Fidelity)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 100)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'deletion_curve.png'), dpi=300, bbox_inches='tight')
    print(f"Saved deletion curve to {output_dir}/deletion_curve.png")
    plt.close()
    
    # Insertion curve
    plt.figure(figsize=(10, 6))
    for method, (ratios, accuracies) in results['insertion'].items():
        plt.plot(ratios * 100, accuracies, label=method, marker='s', linewidth=2, markersize=6)
    plt.xlabel('% Edges Added', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Insertion Curve (Fidelity)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 100)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'insertion_curve.png'), dpi=300, bbox_inches='tight')
    print(f"Saved insertion curve to {output_dir}/insertion_curve.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate Fidelity Curves')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--dataset', type=str, default='synthetic',
                       choices=['synthetic', 'ba2motif'])
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--n_test', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='results/fidelity_curves')
    parser.add_argument('--num_points', type=int, default=21,
                       help='Number of points on curve')
    parser.add_argument('--methods', type=str, nargs='+', 
                       default=['graphormer', 'random'],
                       help='Methods to compare')
    
    args = parser.parse_args()
    
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model - detect model type from path or try both
    print(f"Loading model from {args.model_path}...")
    
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Check if checkpoint contains model config
    if isinstance(checkpoint, dict) and 'config' in checkpoint:
        config = checkpoint['config']
        model_type = config.get('model_type', 'explainer')
        num_classes = config.get('num_classes', 3 if args.dataset == 'synthetic' else 2)
        num_encoder_layers = config.get('num_encoder_layers', 4)
        embedding_dim = config.get('embedding_dim', 128)
        num_in_degree = config.get('num_in_degree', 64)
        num_out_degree = config.get('num_out_degree', 64)
        num_spatial = config.get('num_spatial', 64)
        num_edges = config.get('num_edges', 512)
        ffn_embedding_dim = config.get('ffn_embedding_dim', 128)
        edge_hidden_dim = config.get('edge_hidden_dim', 64)
        classifier_hidden_dim = config.get('classifier_hidden_dim', 64)
    else:
        # Try to infer from path
        if 'pure_graphormer' in args.model_path or 'baseline' in args.model_path:
            model_type = 'pure'
        else:
            model_type = 'explainer'
        
        num_classes = 3 if args.dataset == 'synthetic' else 2
        num_encoder_layers = 4
        embedding_dim = 128
        ffn_embedding_dim = 128
        num_attention_heads = 4
        num_in_degree = 64  # Default used in training
        num_out_degree = 64
        num_spatial = 64
        num_edges = 512
        edge_hidden_dim = 64
        classifier_hidden_dim = 64
    
    # Create appropriate model
    if model_type == 'pure':
        print("Detected PureGraphormer model")
        model = PureGraphormer(
            num_encoder_layers=num_encoder_layers,
            embedding_dim=embedding_dim,
            ffn_embedding_dim=ffn_embedding_dim,
            num_attention_heads=num_attention_heads,
            num_in_degree=num_in_degree,
            num_out_degree=num_out_degree,
            num_spatial=num_spatial,
            num_edges=num_edges,
            num_classes=num_classes,
            classifier_hidden_dim=classifier_hidden_dim
        ).to(device)
    else:
        print("Detected GraphormerExplainer model")
        model = GraphormerExplainer(
            num_encoder_layers=num_encoder_layers,
            embedding_dim=embedding_dim,
            ffn_embedding_dim=ffn_embedding_dim,
            num_attention_heads=num_attention_heads,
            num_in_degree=num_in_degree,
            num_out_degree=num_out_degree,
            num_spatial=num_spatial,
            num_edges=num_edges,
            num_classes=num_classes,
            use_pattern_dict=False,
            edge_hidden_dim=edge_hidden_dim,
            classifier_hidden_dim=classifier_hidden_dim
        ).to(device)
    
    # Load state dict
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
            print("WARNING: Less than 50% of parameters loaded. Model may not work correctly!")
            raise RuntimeError(f"Failed to load model: only {loaded_ratio:.1%} parameters matched")
    
    # Create test dataset
    collator = GraphormerCollator()
    
    if args.dataset == 'synthetic':
        # Check for cached dataset
        import pickle
        import hashlib
        cache_key = f"synthetic_test_{args.n_test}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
        cache_file = f"data/.cache_{cache_hash}.pkl"
        
        if os.path.exists(cache_file):
            print(f"Loading cached dataset from {cache_file}...")
            with open(cache_file, 'rb') as f:
                test_graphs = pickle.load(f)
        else:
            print(f"Generating {args.n_test} synthetic test graphs...")
            test_graphs = generate_synthetic_dataset(
                n_graphs=args.n_test, 
                motif_types=['house', 'cycle', 'star'],
                base_nodes=20
            )
            # Cache the dataset
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(test_graphs, f)
            print(f"Cached dataset to {cache_file}")
        
        test_loader = DataLoader(test_graphs, batch_size=args.batch_size, 
                                shuffle=False, collate_fn=collator)
    elif args.dataset == 'ba2motif':
        print("Loading BA-2Motif test dataset...")
        test_dataset = BA2Motif(args.data_dir, mode='test')
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                                shuffle=False, collate_fn=collator)
    
    # Compute curves
    ratios = np.linspace(0, 1, args.num_points)
    results = {
        'deletion': {},
        'insertion': {}
    }
    
    # Graphormer Explainer
    if 'graphormer' in args.methods:
        print("\n" + "="*60)
        print("Computing Graphormer Explainer curves...")
        print("="*60)
        del_ratios, del_accs = compute_fidelity_curve(model, test_loader, device, 
                                                      method='graphormer', ratios=ratios, collator=collator)
        results['deletion']['Graphormer Explainer'] = (del_ratios, del_accs)
        
        ins_ratios, ins_accs = compute_insertion_curve(model, test_loader, device,
                                                       method='graphormer', ratios=ratios, collator=collator)
        results['insertion']['Graphormer Explainer'] = (ins_ratios, ins_accs)
    
    # Random baseline
    if 'random' in args.methods:
        print("\n" + "="*60)
        print("Computing Random baseline curves...")
        print("="*60)
        del_ratios, del_accs = compute_fidelity_curve(model, test_loader, device,
                                                      method='random', ratios=ratios, collator=collator)
        results['deletion']['Random'] = (del_ratios, del_accs)
        
        ins_ratios, ins_accs = compute_insertion_curve(model, test_loader, device,
                                                       method='random', ratios=ratios, collator=collator)
        results['insertion']['Random'] = (ins_ratios, ins_accs)
    
    # Plot results
    print("\n" + "="*60)
    print("Plotting fidelity curves...")
    print("="*60)
    plot_fidelity_curves(results, args.output_dir)
    
    # Compute AUC
    print("\nFidelity AUC (Deletion):")
    for method, (ratios, accs) in results['deletion'].items():
        auc = np.trapz(accs, ratios) / ratios[-1]  # Normalized AUC
        print(f"  {method}: {auc:.4f}")
    
    print("\nFidelity AUC (Insertion):")
    for method, (ratios, accs) in results['insertion'].items():
        auc = np.trapz(accs, ratios) / ratios[-1]  # Normalized AUC
        print(f"  {method}: {auc:.4f}")
    
    print(f"\nAll results saved to {args.output_dir}")


if __name__ == '__main__':
    main()
