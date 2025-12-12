"""
GIN baseline for comparison.

Graph Isomorphism Network implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool


class GIN(nn.Module):
    """Graph Isomorphism Network baseline."""
    
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=2, dropout=0.1):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GIN layers
        self.convs = nn.ModuleList()
        
        # First layer
        nn1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.convs.append(GINConv(nn1, train_eps=True))
        
        # Remaining layers
        for _ in range(num_layers - 1):
            nn_layer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(nn_layer, train_eps=True))
        
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
        # GIN layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling
        graph_rep = global_mean_pool(x, batch)
        
        # Classification
        logits = self.classifier(graph_rep)
        
        return logits


def create_gin_model(input_dim, hidden_dim, num_classes, num_layers=2, dropout=0.1):
    """Create GIN model."""
    return GIN(input_dim, hidden_dim, num_classes, num_layers, dropout)
