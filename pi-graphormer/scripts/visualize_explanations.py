#!/usr/bin/env python
"""
Experiment 4: Qualitative Visualization

Visualize explanations for sample graphs.
Shows ground truth, Graphormer explanation, and random baseline side-by-side.
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_v2 import GraphormerExplainer
from data_utils import GraphormerCollator
from dataset.datasets import generate_synthetic_dataset, BA2Motif
from torch.utils.data import DataLoader
from torch_geometric.utils import to_networkx


def visualize_graph(edge_index, edge_scores, edge_gt=None, title="Graph", ax=None, node_labels=None):
    """
    Visualize graph with edge importance scores.
    
    Args:
        edge_index: [2, num_edges] tensor
        edge_scores: [num_edges] array of importance scores
        edge_gt: [num_edges] array of ground truth (optional)
        title: Plot title
        ax: Matplotlib axis (optional)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    
    # Convert to numpy
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.cpu().numpy()
    if isinstance(edge_scores, torch.Tensor):
        edge_scores = edge_scores.cpu().numpy()
    if edge_gt is not None and isinstance(edge_gt, torch.Tensor):
        edge_gt = edge_gt.cpu().numpy()
    
    # Create NetworkX graph
    G = nx.Graph()
    edge_list = edge_index.T
    
    # Add nodes
    num_nodes = edge_index.max() + 1
    G.add_nodes_from(range(num_nodes))
    
    # Add edges with scores
    for i, (src, dst) in enumerate(edge_list):
        if i < len(edge_scores):
            G.add_edge(int(src), int(dst), weight=edge_scores[i])
        else:
            G.add_edge(int(src), int(dst), weight=0.5)
    
    # Layout
    try:
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    except:
        pos = nx.circular_layout(G)
    
    # Normalize edge scores for visualization
    if len(edge_scores) > 0:
        edge_scores_norm = (edge_scores - edge_scores.min()) / (edge_scores.max() - edge_scores.min() + 1e-8)
    else:
        edge_scores_norm = np.ones(len(edge_list))
    
    # Draw edges with color based on importance
    edges = list(G.edges())
    edge_colors = []
    edge_widths = []
    
    for u, v in edges:
        # Find edge index
        edge_idx = None
        for i, (src, dst) in enumerate(edge_list):
            if (src == u and dst == v) or (src == v and dst == u):
                edge_idx = i
                break
        
        if edge_idx is not None and edge_idx < len(edge_scores_norm):
            score = edge_scores_norm[edge_idx]
            edge_colors.append(score)
            edge_widths.append(1 + score * 2)  # Width 1-3 based on importance
        else:
            edge_colors.append(0.5)
            edge_widths.append(1)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.6, width=edge_widths,
                          edge_color=edge_colors, edge_cmap=plt.cm.Reds,
                          ax=ax)
    
    # Highlight ground truth edges if available
    if edge_gt is not None:
        gt_edges = []
        for i, (src, dst) in enumerate(edge_list):
            if i < len(edge_gt) and edge_gt[i] > 0.5:
                gt_edges.append((int(src), int(dst)))
        
        if len(gt_edges) > 0:
            nx.draw_networkx_edges(G, pos, edgelist=gt_edges, alpha=0.9,
                                  width=3, edge_color='green', style='dashed',
                                  label='Ground Truth', ax=ax)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500,
                          alpha=0.9, ax=ax)
    
    # Draw labels
    if node_labels is None:
        node_labels = {i: str(i) for i in range(num_nodes)}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, ax=ax)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    
    return ax


def main():
    parser = argparse.ArgumentParser(description='Visualize Explanations')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of sample graphs to visualize')
    parser.add_argument('--dataset', type=str, default='synthetic',
                       choices=['synthetic', 'ba2motif'])
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='results/visualizations')
    parser.add_argument('--n_test', type=int, default=100,
                       help='Number of test graphs to sample from')
    
    args = parser.parse_args()
    
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    num_classes = 3 if args.dataset == 'synthetic' else 2
    model = GraphormerExplainer(
        num_encoder_layers=4,
        embedding_dim=128,
        num_classes=num_classes,
        use_pattern_dict=False,
        classifier_hidden_dim=64
    ).to(device)
    
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device), strict=False)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
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
                                shuffle=True, collate_fn=collator)
    elif args.dataset == 'ba2motif':
        print("Loading BA-2Motif test dataset...")
        test_dataset = BA2Motif(args.data_dir, mode='test')
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                                shuffle=True, collate_fn=collator)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Visualize samples
    model.eval()
    sample_count = 0
    
    print(f"\nVisualizing {args.num_samples} sample graphs...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if sample_count >= args.num_samples:
                break
            
            graphormer_data = {k: v.to(device) for k, v in batch['graphormer_data'].items()}
            pyg_batch = batch['pyg_batch'].to(device)
            edge_gt = batch['edge_gt_att']
            
            # Get explanations
            _, edge_scores = model(graphormer_data, pyg_batch, return_explanation=True)
            
            # Process each graph in batch
            batch_size = pyg_batch.batch.max().item() + 1
            
            for graph_idx in range(batch_size):
                if sample_count >= args.num_samples:
                    break
                
                # Get edges and scores for this graph
                graph_mask = (pyg_batch.batch == graph_idx)
                graph_edge_mask = graph_mask[pyg_batch.edge_index[0]] & graph_mask[pyg_batch.edge_index[1]]
                
                graph_edges = pyg_batch.edge_index[:, graph_edge_mask]
                graph_scores = edge_scores[graph_edge_mask]
                graph_gt = edge_gt[graph_edge_mask] if edge_gt is not None else None
                
                # Remap node indices to 0-based for this graph
                unique_nodes = torch.unique(graph_edges)
                node_mapping = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(unique_nodes)}
                remapped_edges = torch.zeros_like(graph_edges)
                for i in range(graph_edges.size(1)):
                    remapped_edges[0, i] = node_mapping[graph_edges[0, i].item()]
                    remapped_edges[1, i] = node_mapping[graph_edges[1, i].item()]
                
                # Create visualization
                fig, axes = plt.subplots(1, 3, figsize=(24, 8))
                
                # Ground truth
                visualize_graph(remapped_edges, np.ones(len(graph_scores)), graph_gt,
                              title="Ground Truth", ax=axes[0])
                
                # Graphormer explanation
                visualize_graph(remapped_edges, graph_scores.cpu().numpy(), None,
                              title="Graphormer Explainer", ax=axes[1])
                
                # Random baseline
                random_scores = np.random.rand(len(graph_scores))
                visualize_graph(remapped_edges, random_scores, None,
                              title="Random Baseline", ax=axes[2])
                
                plt.tight_layout()
                output_path = os.path.join(args.output_dir, f'sample_{sample_count}.png')
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"  Saved sample {sample_count} to {output_path}")
                sample_count += 1
    
    print(f"\nVisualizations saved to {args.output_dir}")


if __name__ == '__main__':
    main()
