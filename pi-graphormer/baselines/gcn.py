"""
GCN baseline for comparison.

Simple Graph Convolutional Network implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class GCN(nn.Module):
    """Graph Convolutional Network baseline."""
    
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=2, dropout=0.1):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GCN layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Classification head
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x, edge_index, batch):
        """
        Args:
            x: [total_nodes, input_dim] - node features
            edge_index: [2, num_edges] - edge indices
            batch: [total_nodes] - batch assignment
        Returns:
            logits: [batch_size, num_classes]
        """
        # GCN layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling
        graph_rep = global_mean_pool(x, batch)
        
        # Classification
        logits = self.classifier(graph_rep)
        
        return logits


def create_gcn_model(input_dim, hidden_dim, num_classes, num_layers=2, dropout=0.1):
    """Create GCN model."""
    return GCN(input_dim, hidden_dim, num_classes, num_layers, dropout)
