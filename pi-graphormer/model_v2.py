# Graphormer-PIGNN v2: Simplified Architecture
# Graphormer as the main predictor with integrated edge-scoring explainer
# No separate GNN predictor - Graphormer does both prediction and explanation

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from torch_geometric.utils import to_dense_adj, dense_to_sparse


# ============================================================================
# Graphormer Components (Reused from original)
# ============================================================================

def init_params(module, n_layers):
    """Initialize parameters for Graphormer modules."""
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


class MultiheadAttention(nn.Module):
    """Multi-headed attention with support for attention bias (spatial/edge encoding)."""
    
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.scaling = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1/math.sqrt(2))
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1/math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1/math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
    
    def forward(self, x, attn_bias=None, key_padding_mask=None):
        """
        Args:
            x: [batch, seq_len, embed_dim]
            attn_bias: [batch, num_heads, seq_len, seq_len]
            key_padding_mask: [batch, seq_len] - True for positions to mask
        Returns:
            output: [batch, seq_len, embed_dim]
            attn_weights: [batch, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.size()
        
        # Project Q, K, V
        q = self.q_proj(x) * self.scaling
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1))  # [B, H, L, L]
        
        # Add attention bias (spatial encoding + edge encoding)
        if attn_bias is not None:
            attn_weights = attn_weights + attn_bias
        
        # Apply key padding mask
        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_probs, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(output)
        
        return output, attn_weights


class GraphormerEncoderLayer(nn.Module):
    """Single Graphormer encoder layer."""
    
    def __init__(self, embed_dim, num_heads, ffn_dim, dropout=0.1, attention_dropout=0.1):
        super().__init__()
        
        self.self_attn = MultiheadAttention(embed_dim, num_heads, attention_dropout)
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, attn_bias=None, key_padding_mask=None):
        """
        Args:
            x: [batch, seq_len, embed_dim]
            attn_bias: [batch, num_heads, seq_len, seq_len]
            key_padding_mask: [batch, seq_len]
        Returns:
            x: [batch, seq_len, embed_dim]
            attn_weights: [batch, num_heads, seq_len, seq_len]
        """
        # Pre-norm architecture
        residual = x
        x = self.self_attn_layer_norm(x)
        x, attn_weights = self.self_attn(x, attn_bias, key_padding_mask)
        x = self.dropout(x)
        x = residual + x
        
        residual = x
        x = self.final_layer_norm(x)
        x = self.ffn(x)
        x = residual + x
        
        return x, attn_weights


class GraphNodeFeature(nn.Module):
    """Compute node features including centrality encoding."""
    
    def __init__(self, num_atoms, num_in_degree, num_out_degree, hidden_dim, n_layers):
        super().__init__()
        
        self.atom_encoder = nn.Embedding(num_atoms + 1, hidden_dim, padding_idx=0)
        self.in_degree_encoder = nn.Embedding(num_in_degree, hidden_dim, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(num_out_degree, hidden_dim, padding_idx=0)
        
        # [VNode] token for graph-level representation
        self.graph_token = nn.Embedding(1, hidden_dim)
        
        self.apply(lambda m: init_params(m, n_layers))
    
    def forward(self, x, in_degree, out_degree):
        """
        Args:
            x: [batch, n_node, n_features] - node features (as indices)
            in_degree: [batch, n_node]
            out_degree: [batch, n_node]
        Returns:
            node_features: [batch, n_node+1, hidden_dim] (includes graph token)
        """
        batch_size, n_node = x.size()[:2]
        
        # Clamp indices to valid range for embeddings
        x = x.clamp(0, self.atom_encoder.num_embeddings - 1)
        in_degree = in_degree.clamp(0, self.in_degree_encoder.num_embeddings - 1)
        out_degree = out_degree.clamp(0, self.out_degree_encoder.num_embeddings - 1)
        
        # Encode node features
        node_feature = self.atom_encoder(x).sum(dim=-2)  # [B, N, D]
        
        # Add centrality encoding
        node_feature = (
            node_feature 
            + self.in_degree_encoder(in_degree) 
            + self.out_degree_encoder(out_degree)
        )
        
        # Add graph token
        graph_token = self.graph_token.weight.unsqueeze(0).expand(batch_size, -1, -1)
        node_feature = torch.cat([graph_token, node_feature], dim=1)
        
        return node_feature


class GraphAttnBias(nn.Module):
    """Compute attention bias from spatial and edge encodings."""
    
    def __init__(self, num_heads, num_edges, num_spatial, hidden_dim, n_layers):
        super().__init__()
        
        self.num_heads = num_heads
        
        # Spatial encoding (based on shortest path distance)
        self.spatial_pos_encoder = nn.Embedding(num_spatial, num_heads, padding_idx=0)
        
        # Edge encoding
        self.edge_encoder = nn.Embedding(num_edges + 1, num_heads, padding_idx=0)
        
        # Virtual node distance
        self.graph_token_virtual_distance = nn.Embedding(1, num_heads)
        
        self.apply(lambda m: init_params(m, n_layers))
    
    def forward(self, attn_bias, spatial_pos, attn_edge_type):
        """
        Args:
            attn_bias: [batch, n_node+1, n_node+1] - base attention bias
            spatial_pos: [batch, n_node, n_node] - shortest path distances
            attn_edge_type: [batch, n_node, n_node, n_edge_features]
        Returns:
            graph_attn_bias: [batch, num_heads, n_node+1, n_node+1]
        """
        batch_size, n_node_plus_1, _ = attn_bias.size()
        n_node = n_node_plus_1 - 1
        
        # Clamp indices to valid range
        spatial_pos = spatial_pos.clamp(0, self.spatial_pos_encoder.num_embeddings - 1)
        attn_edge_type = attn_edge_type.clamp(0, self.edge_encoder.num_embeddings - 1)
        
        # Expand attention bias for all heads
        graph_attn_bias = attn_bias.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        
        # Add spatial encoding
        spatial_pos_bias = self.spatial_pos_encoder(spatial_pos).permute(0, 3, 1, 2)
        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + spatial_pos_bias
        
        # Add virtual node distance
        t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
        graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
        graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t
        
        # Add edge encoding
        edge_input = self.edge_encoder(attn_edge_type).mean(-2).permute(0, 3, 1, 2)
        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + edge_input
        
        return graph_attn_bias


class GraphormerEncoder(nn.Module):
    """Full Graphormer encoder."""
    
    def __init__(
        self,
        num_atoms=512,
        num_in_degree=512,
        num_out_degree=512,
        num_edges=512,
        num_spatial=512,
        num_encoder_layers=6,
        embedding_dim=256,
        ffn_embedding_dim=256,
        num_attention_heads=8,
        dropout=0.1,
        attention_dropout=0.1,
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_attention_heads
        
        # Node feature and attention bias modules
        self.graph_node_feature = GraphNodeFeature(
            num_atoms, num_in_degree, num_out_degree, 
            embedding_dim, num_encoder_layers
        )
        
        self.graph_attn_bias = GraphAttnBias(
            num_attention_heads, num_edges, num_spatial,
            embedding_dim, num_encoder_layers
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            GraphormerEncoderLayer(
                embedding_dim, num_attention_heads, ffn_embedding_dim,
                dropout, attention_dropout
            )
            for _ in range(num_encoder_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, batched_data, return_attention=False):
        """
        Args:
            batched_data: dict with keys:
                - x: [batch, n_node, n_features]
                - attn_bias: [batch, n_node+1, n_node+1]
                - spatial_pos: [batch, n_node, n_node]
                - attn_edge_type: [batch, n_node, n_node, n_edge_features]
                - in_degree: [batch, n_node]
                - out_degree: [batch, n_node]
        Returns:
            node_rep: [batch, n_node, embedding_dim] (excluding graph token)
            graph_rep: [batch, embedding_dim]
            all_attn_weights: list of [batch, num_heads, n_node+1, n_node+1] if return_attention
        """
        x = batched_data['x']
        attn_bias = batched_data['attn_bias']
        spatial_pos = batched_data['spatial_pos']
        attn_edge_type = batched_data['attn_edge_type']
        in_degree = batched_data['in_degree']
        out_degree = batched_data['out_degree']
        
        # Compute node features (including graph token)
        x = self.graph_node_feature(x, in_degree, out_degree)  # [B, N+1, D]
        
        # Compute attention bias
        attn_bias = self.graph_attn_bias(attn_bias, spatial_pos, attn_edge_type)
        
        # Compute padding mask
        n_graph, n_node_plus_1 = x.size()[:2]
        padding_mask = (batched_data['x'][:, :, 0] == 0)  # [B, N]
        padding_mask_with_token = torch.cat([
            torch.zeros(n_graph, 1, device=padding_mask.device, dtype=torch.bool),
            padding_mask
        ], dim=1)  # [B, N+1]
        
        # Apply dropout
        x = self.dropout(x)
        
        # Pass through transformer layers
        all_attn_weights = []
        for layer in self.layers:
            x, attn_weights = layer(x, attn_bias, padding_mask_with_token)
            if return_attention:
                all_attn_weights.append(attn_weights)
        
        x = self.layer_norm(x)
        
        # Separate graph token and node representations
        graph_rep = x[:, 0, :]  # [B, D]
        node_rep = x[:, 1:, :]  # [B, N, D]
        
        if return_attention:
            return node_rep, graph_rep, all_attn_weights
        return node_rep, graph_rep


# ============================================================================
# Classification Head (Uses Graphormer's Graph Representation)
# ============================================================================

class GraphormerClassifier(nn.Module):
    """Classification head that uses Graphormer's graph representation."""
    
    def __init__(self, embedding_dim, hidden_dim, num_classes, dropout=0.1):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, graph_rep):
        """
        Args:
            graph_rep: [batch_size, embedding_dim] - graph representation from Graphormer
        Returns:
            logits: [batch_size, num_classes]
        """
        return self.classifier(graph_rep)


# ============================================================================
# Sparse Dictionary / Autoencoder Module (PI-GNN style)
# ============================================================================

class SparseDictionaryEncoder(nn.Module):
    """
    Sparse dictionary encoder that learns structural patterns (atoms).
    
    Inspired by PI-GNN: learns a dictionary of structural patterns and
    produces sparse codes for each node. Patterns are reusable across graphs.
    """
    
    def __init__(self, input_dim, num_atoms, atom_dim, sparsity_weight=0.01):
        """
        Args:
            input_dim: Dimension of input node embeddings
            num_atoms: Number of dictionary atoms (structural patterns)
            atom_dim: Dimension of each atom
            sparsity_weight: Weight for sparsity regularization
        """
        super().__init__()
        
        self.num_atoms = num_atoms
        self.atom_dim = atom_dim
        self.sparsity_weight = sparsity_weight
        
        # Dictionary: learnable atoms (structural patterns)
        # Each atom represents a structural pattern
        self.dictionary = nn.Parameter(
            torch.randn(num_atoms, atom_dim) * 0.1
        )
        
        # Encoder: maps node embeddings to sparse codes
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, atom_dim * 2),
            nn.ReLU(),
            nn.Linear(atom_dim * 2, num_atoms)
        )
        
        # Decoder: reconstructs node embeddings from sparse codes
        self.decoder = nn.Sequential(
            nn.Linear(num_atoms, atom_dim * 2),
            nn.ReLU(),
            nn.Linear(atom_dim * 2, input_dim)
        )
    
    def forward(self, node_rep):
        """
        Args:
            node_rep: [total_nodes, input_dim] - node embeddings from Graphormer
        Returns:
            sparse_codes: [total_nodes, num_atoms] - sparse activation codes
            reconstructed: [total_nodes, input_dim] - reconstructed embeddings
            sparsity_loss: scalar - sparsity regularization loss
        """
        # Encode to sparse codes
        codes = self.encoder(node_rep)  # [N, num_atoms]
        
        # Apply sparsity: soft thresholding or ReLU
        sparse_codes = F.relu(codes)  # [N, num_atoms]
        
        # Normalize codes (L1 normalization for sparsity)
        sparse_codes = sparse_codes / (sparse_codes.sum(dim=1, keepdim=True) + 1e-8)
        
        # Decode: reconstruct embeddings
        reconstructed = self.decoder(sparse_codes)  # [N, input_dim]
        
        # Sparsity loss: encourage sparse codes (L1 penalty)
        sparsity_loss = self.sparsity_weight * sparse_codes.sum()
        
        return sparse_codes, reconstructed, sparsity_loss
    
    def get_pattern_activations(self, sparse_codes, edge_index):
        """
        Get pattern activations for edges.
        
        Edges connecting nodes with high activation for the same atom
        are more important (they share the same structural pattern).
        
        Args:
            sparse_codes: [total_nodes, num_atoms] - sparse codes
            edge_index: [2, num_edges] - edge indices
        Returns:
            edge_pattern_scores: [num_edges, num_atoms] - pattern scores per edge
        """
        src_codes = sparse_codes[edge_index[0]]  # [E, num_atoms]
        dst_codes = sparse_codes[edge_index[1]]  # [E, num_atoms]
        
        # Edge score = product of source and target activations for each atom
        # High score means both nodes activate the same pattern
        edge_pattern_scores = src_codes * dst_codes  # [E, num_atoms]
        
        return edge_pattern_scores


class PatternBasedEdgeScorer(nn.Module):
    """
    Edge scorer based on sparse dictionary patterns.
    
    Derives edge scores from sparse codes: edges connecting nodes with
    high activation for the same pattern (atom) are more important.
    """
    
    def __init__(self, num_atoms, aggregation='max'):
        """
        Args:
            num_atoms: Number of dictionary atoms
            aggregation: How to aggregate pattern scores ('max', 'sum', 'mean')
        """
        super().__init__()
        
        self.num_atoms = num_atoms
        self.aggregation = aggregation
        
        # Optional: learnable weights for each pattern
        self.pattern_weights = nn.Parameter(torch.ones(num_atoms) / num_atoms)
    
    def forward(self, sparse_codes, edge_index):
        """
        Args:
            sparse_codes: [total_nodes, num_atoms] - sparse activation codes
            edge_index: [2, num_edges] - edge indices
        Returns:
            edge_scores: [num_edges] - importance score for each edge
        """
        # Get pattern activations for edges
        src_codes = sparse_codes[edge_index[0]]  # [E, num_atoms]
        dst_codes = sparse_codes[edge_index[1]]  # [E, num_atoms]
        
        # Edge score = product of source and target activations
        # High score means both nodes activate the same pattern
        edge_pattern_scores = src_codes * dst_codes  # [E, num_atoms]
        
        # Weight patterns
        weighted_scores = edge_pattern_scores * self.pattern_weights.unsqueeze(0)  # [E, num_atoms]
        
        # Aggregate across patterns
        if self.aggregation == 'max':
            edge_scores = weighted_scores.max(dim=1)[0]  # [E]
        elif self.aggregation == 'sum':
            edge_scores = weighted_scores.sum(dim=1)  # [E]
        elif self.aggregation == 'mean':
            edge_scores = weighted_scores.mean(dim=1)  # [E]
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
        
        # Normalize to [0, 1]
        edge_scores = torch.sigmoid(edge_scores)
        
        return edge_scores


class GlobalPatternRegularizer(nn.Module):
    """
    Global pattern regularizer to encourage atom reuse across graphs.
    
    Encourages the same dictionary atoms to be used across different graphs,
    promoting interpretability and pattern discovery.
    """
    
    def __init__(self, num_atoms, reuse_weight=0.01):
        """
        Args:
            num_atoms: Number of dictionary atoms
            reuse_weight: Weight for reuse regularization
        """
        super().__init__()
        
        self.num_atoms = num_atoms
        self.reuse_weight = reuse_weight
    
    def forward(self, sparse_codes, batch):
        """
        Args:
            sparse_codes: [total_nodes, num_atoms] - sparse codes for all nodes
            batch: [total_nodes] - batch assignment (which graph each node belongs to)
        Returns:
            reuse_loss: scalar - encourages atom reuse across graphs
        """
        if batch is None:
            # Single graph: no cross-graph reuse
            return torch.tensor(0.0, device=sparse_codes.device)
        
        # Group codes by graph
        num_graphs = batch.max().item() + 1
        graph_codes = []
        
        for g in range(num_graphs):
            mask = (batch == g)
            graph_code = sparse_codes[mask].mean(dim=0)  # [num_atoms] - average activation per graph
            graph_codes.append(graph_code)
        
        graph_codes = torch.stack(graph_codes)  # [num_graphs, num_atoms]
        
        # Encourage diversity: each graph should use different atoms
        # But also encourage reuse: same atoms should be used across graphs
        # We use variance across graphs as a proxy for reuse
        
        # Variance across graphs for each atom (high variance = reused)
        atom_variances = graph_codes.var(dim=0)  # [num_atoms]
        
        # Encourage high variance (atoms are reused across graphs)
        reuse_loss = -self.reuse_weight * atom_variances.mean()
        
        return reuse_loss


# ============================================================================
# Edge Scoring Module (Legacy - kept for backward compatibility)
# ============================================================================

class EdgeScorePredictor(nn.Module):
    """Predict edge importance scores from node representations."""
    
    def __init__(self, node_dim, hidden_dim=128):
        super().__init__()
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, node_rep, edge_index, batch=None):
        """
        Args:
            node_rep: [total_nodes, node_dim] - node representations (unbatched/flattened)
            edge_index: [2, num_edges] - edge indices
            batch: [total_nodes] - batch assignment for each node (optional)
        Returns:
            edge_scores: [num_edges] - importance score for each edge
        """
        # Get source and target node representations
        src_rep = node_rep[edge_index[0]]  # [E, D]
        dst_rep = node_rep[edge_index[1]]  # [E, D]
        
        # Concatenate and predict score
        edge_feat = torch.cat([src_rep, dst_rep], dim=-1)  # [E, 2D]
        edge_scores = torch.sigmoid(self.edge_mlp(edge_feat)).squeeze(-1)  # [E]
        
        return edge_scores


# ============================================================================
# Explanation Regularizers
# ============================================================================

class ExplanationRegularizer(nn.Module):
    """Regularizers for edge explanations to encourage desirable properties."""
    
    def __init__(self):
        super().__init__()
    
    def sparsity_loss(self, edge_scores, method='l1', weight=1.0):
        """
        Encourage sparse edge scores.
        
        Args:
            edge_scores: [num_edges] - edge importance scores
            method: 'l1' or 'entropy'
            weight: regularization weight
        Returns:
            loss: scalar
        """
        if method == 'l1':
            # L1 norm: ||ρ̂||₁ = sum(|ρ̂_ij|)
            # Since scores are in [0,1], |ρ̂_ij| = ρ̂_ij
            return weight * edge_scores.sum()
        elif method == 'entropy':
            # Entropy regularization: encourage scores to be near 0 or 1
            entropy = -edge_scores * torch.log(edge_scores + 1e-8) - \
                     (1 - edge_scores) * torch.log(1 - edge_scores + 1e-8)
            return weight * entropy.mean()
        else:
            raise ValueError(f"Unknown sparsity method: {method}")
    
    def connectivity_loss(self, edge_scores, edge_index, num_nodes, weight=1.0, top_k_ratio=0.3):
        """
        Encourage important edges to form connected components.
        
        Args:
            edge_scores: [num_edges] - edge importance scores
            edge_index: [2, num_edges] - edge indices
            num_nodes: number of nodes in the graph
            weight: regularization weight
            top_k_ratio: ratio of top edges to consider
        Returns:
            loss: scalar (0 if connected, >0 if disconnected)
        """
        # Get top-k edges
        k = max(1, int(num_nodes * top_k_ratio))
        top_k_values, top_k_indices = torch.topk(edge_scores, min(k, len(edge_scores)))
        
        if len(top_k_indices) == 0:
            return torch.tensor(0.0, device=edge_scores.device)
        
        # Build adjacency matrix for top-k edges
        top_edge_index = edge_index[:, top_k_indices]
        adj = torch.zeros(num_nodes, num_nodes, device=edge_scores.device)
        adj[top_edge_index[0], top_edge_index[1]] = 1.0
        adj[top_edge_index[1], top_edge_index[0]] = 1.0  # Make symmetric
        
        # Check connectivity using BFS/DFS (simplified: check if all nodes reachable)
        # For simplicity, we use a heuristic: penalize if there are isolated nodes
        node_degrees = adj.sum(dim=1)
        isolated_nodes = (node_degrees == 0).sum().float()
        
        # Normalize by number of nodes
        connectivity_loss = weight * (isolated_nodes / num_nodes)
        
        return connectivity_loss
    
    def size_loss(self, edge_scores, target_size=None, weight=1.0):
        """
        Encourage a specific number of important edges.
        
        Args:
            edge_scores: [num_edges] - edge importance scores
            target_size: target number of important edges (if None, uses mean)
            weight: regularization weight
        Returns:
            loss: scalar
        """
        if target_size is None:
            # Encourage sparsity: target is small fraction of edges
            target_size = len(edge_scores) * 0.1
        
        # Sum of scores should be close to target_size
        actual_size = edge_scores.sum()
        size_loss = weight * (actual_size - target_size) ** 2
        
        return size_loss
    
    def compute_all_losses(self, edge_scores, edge_index, num_nodes, 
                          sparsity_weight=0.01, connectivity_weight=0.01, 
                          size_weight=0.01, sparsity_method='entropy'):
        """
        Compute all regularization losses.
        
        Returns:
            dict with individual losses and total loss
        """
        losses = {}
        
        # Sparsity loss
        losses['sparsity'] = self.sparsity_loss(edge_scores, method=sparsity_method, 
                                                weight=sparsity_weight)
        
        # Connectivity loss (per graph, need to aggregate)
        # For batched data, we'd need to compute per graph
        # Simplified: compute on first graph or average
        if isinstance(num_nodes, torch.Tensor):
            num_nodes = num_nodes[0].item() if num_nodes.numel() > 0 else len(edge_scores)
        
        losses['connectivity'] = self.connectivity_loss(edge_scores, edge_index, 
                                                        num_nodes, weight=connectivity_weight)
        
        # Size loss
        losses['size'] = self.size_loss(edge_scores, weight=size_weight)
        
        # Total regularization loss
        losses['total'] = losses['sparsity'] + losses['connectivity'] + losses['size']
        
        return losses


# ============================================================================
# Main Unified Model: Graphormer as Predictor + Explainer
# ============================================================================

class GraphormerExplainer(nn.Module):
    """
    Unified Graphormer model that serves as both predictor and explainer.
    
    Architecture:
    1. GraphormerEncoder → graph_rep (for prediction) + node_rep (for explanation)
    2. GraphormerClassifier → predictions from graph_rep
    3. SparseDictionaryEncoder → learns structural patterns (atoms)
    4. PatternBasedEdgeScorer → edge scores from pattern activations
    5. ExplanationRegularizer → regularization losses
    """
    
    def __init__(
        self,
        # Graphormer encoder params
        num_atoms=512,
        num_in_degree=512,
        num_out_degree=512,
        num_edges=512,
        num_spatial=512,
        num_encoder_layers=6,
        embedding_dim=256,
        ffn_embedding_dim=256,
        num_attention_heads=8,
        dropout=0.1,
        attention_dropout=0.1,
        # Sparse dictionary params
        use_pattern_dict=True,
        num_pattern_atoms=32,
        pattern_atom_dim=64,
        pattern_sparsity_weight=0.01,
        pattern_reuse_weight=0.01,
        pattern_aggregation='max',
        # Edge scorer params (legacy, used if use_pattern_dict=False)
        edge_hidden_dim=128,
        # Classifier params
        classifier_hidden_dim=128,
        num_classes=2,
        # Regularization params
        use_regularization=True,
    ):
        super().__init__()
        
        # Graphormer encoder (main model)
        self.graphormer = GraphormerEncoder(
            num_atoms=num_atoms,
            num_in_degree=num_in_degree,
            num_out_degree=num_out_degree,
            num_edges=num_edges,
            num_spatial=num_spatial,
            num_encoder_layers=num_encoder_layers,
            embedding_dim=embedding_dim,
            ffn_embedding_dim=ffn_embedding_dim,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
        )
        
        # Classification head (uses graph representation)
        self.classifier = GraphormerClassifier(
            embedding_dim=embedding_dim,
            hidden_dim=classifier_hidden_dim,
            num_classes=num_classes,
            dropout=dropout
        )
        
        # Sparse dictionary encoder (PI-GNN style)
        self.use_pattern_dict = use_pattern_dict
        if use_pattern_dict:
            self.pattern_dict = SparseDictionaryEncoder(
                input_dim=embedding_dim,
                num_atoms=num_pattern_atoms,
                atom_dim=pattern_atom_dim,
                sparsity_weight=pattern_sparsity_weight
            )
            
            self.edge_scorer = PatternBasedEdgeScorer(
                num_atoms=num_pattern_atoms,
                aggregation=pattern_aggregation
            )
            
            self.pattern_regularizer = GlobalPatternRegularizer(
                num_atoms=num_pattern_atoms,
                reuse_weight=pattern_reuse_weight
            )
        else:
            # Legacy: simple MLP-based edge scorer
            self.edge_scorer = EdgeScorePredictor(
                node_dim=embedding_dim,
                hidden_dim=edge_hidden_dim
            )
            self.pattern_dict = None
            self.pattern_regularizer = None
        
        # Explanation regularizer (for edge score sparsity)
        self.regularizer = ExplanationRegularizer() if use_regularization else None
        
        self.embedding_dim = embedding_dim
        self.use_regularization = use_regularization
    
    def forward(self, batched_data, pyg_batch, return_explanation=True, 
                return_regularization=False):
        """
        Args:
            batched_data: dict formatted for Graphormer
            pyg_batch: PyG batch object with edge_index, batch, x, y, etc.
            return_explanation: whether to return edge scores
            return_regularization: whether to return regularization losses
        Returns:
            logits: [batch_size, num_classes] - class predictions
            edge_scores: [total_edges] - edge importance scores (if return_explanation)
            reg_losses: dict of regularization losses (if return_regularization)
        """
        # Get node and graph representations from Graphormer
        node_rep_batched, graph_rep = self.graphormer(batched_data)
        
        # Convert batched node representation to flat representation
        batch_size, max_nodes, dim = node_rep_batched.size()
        node_rep_list = []
        
        x = batched_data['x']  # [B, max_nodes, features]
        for b in range(batch_size):
            # Find non-padded nodes
            mask = (x[b, :, 0] != 0)  # [max_nodes]
            n_nodes = mask.sum().item()
            node_rep_list.append(node_rep_batched[b, :n_nodes, :])
        
        node_rep = torch.cat(node_rep_list, dim=0)  # [total_nodes, D]
        
        # Prediction: use graph representation
        logits = self.classifier(graph_rep)
        
        # Explanation: compute edge scores from node representations
        edge_scores = None
        reg_losses = None
        pattern_losses = {}
        
        if return_explanation:
            if self.use_pattern_dict:
                # PI-GNN style: use sparse dictionary patterns
                sparse_codes, reconstructed, dict_sparsity_loss = self.pattern_dict(node_rep)
                
                # Compute edge scores from pattern activations
                edge_scores = self.edge_scorer(sparse_codes, pyg_batch.edge_index)
                
                # Pattern reuse regularization
                pattern_reuse_loss = self.pattern_regularizer(sparse_codes, pyg_batch.batch)
                
                pattern_losses = {
                    'dict_sparsity': dict_sparsity_loss,
                    'pattern_reuse': pattern_reuse_loss,
                    'reconstruction': F.mse_loss(reconstructed, node_rep) if return_regularization else torch.tensor(0.0, device=node_rep.device)
                }
            else:
                # Legacy: simple MLP-based edge scorer
                edge_scores = self.edge_scorer(node_rep, pyg_batch.edge_index)
            
            # Compute regularization losses if requested
            if return_regularization and self.use_regularization:
                # Get number of nodes per graph
                num_nodes_per_graph = [len(rep) for rep in node_rep_list]
                # For simplicity, compute regularization on first graph
                # In practice, you'd want to compute per graph and aggregate
                if len(num_nodes_per_graph) > 0:
                    first_graph_nodes = num_nodes_per_graph[0]
                    # Get edges for first graph
                    batch_mask = pyg_batch.batch == 0
                    first_graph_edges = batch_mask[pyg_batch.edge_index[0]] & \
                                        batch_mask[pyg_batch.edge_index[1]]
                    
                    if first_graph_edges.sum() > 0:
                        first_edge_scores = edge_scores[first_graph_edges]
                        first_edge_index = pyg_batch.edge_index[:, first_graph_edges]
                        reg_losses = self.regularizer.compute_all_losses(
                            first_edge_scores, first_edge_index, first_graph_nodes
                        )
                
                # Add pattern losses to reg_losses
                if pattern_losses:
                    if reg_losses is None:
                        reg_losses = {}
                    reg_losses.update(pattern_losses)
                    reg_losses['total'] = reg_losses.get('total', torch.tensor(0.0, device=node_rep.device)) + \
                                         sum(pattern_losses.values())
        
        if return_explanation and return_regularization:
            return logits, edge_scores, reg_losses
        elif return_explanation:
            return logits, edge_scores
        else:
            return logits
    
    def get_pattern_activations(self, batched_data, pyg_batch):
        """Get pattern activations for visualization."""
        if not self.use_pattern_dict:
            return None
        
        node_rep_batched, _ = self.graphormer(batched_data)
        
        # Convert to flat representation
        batch_size, max_nodes, dim = node_rep_batched.size()
        node_rep_list = []
        x = batched_data['x']
        for b in range(batch_size):
            mask = (x[b, :, 0] != 0)
            n_nodes = mask.sum().item()
            node_rep_list.append(node_rep_batched[b, :n_nodes, :])
        
        node_rep = torch.cat(node_rep_list, dim=0)
        
        sparse_codes, _, _ = self.pattern_dict(node_rep)
        return sparse_codes
    
    def explain_only(self, batched_data, pyg_batch):
        """Get only the edge scores (for evaluation)."""
        _, edge_scores = self.forward(batched_data, pyg_batch, return_explanation=True, 
                                      return_regularization=False)
        return edge_scores


# ============================================================================
# Pure Graphormer Baseline (for comparison)
# ============================================================================

class PureGraphormer(nn.Module):
    """
    Pure Graphormer model without explainer.
    
    This is the baseline for comparison - just Graphormer + Classifier,
    no edge scoring or explanation components.
    """
    
    def __init__(
        self,
        # Graphormer encoder params
        num_atoms=512,
        num_in_degree=512,
        num_out_degree=512,
        num_edges=512,
        num_spatial=512,
        num_encoder_layers=6,
        embedding_dim=256,
        ffn_embedding_dim=256,
        num_attention_heads=8,
        dropout=0.1,
        attention_dropout=0.1,
        # Classifier params
        classifier_hidden_dim=128,
        num_classes=2,
    ):
        super().__init__()
        
        # Graphormer encoder
        self.graphormer = GraphormerEncoder(
            num_atoms=num_atoms,
            num_in_degree=num_in_degree,
            num_out_degree=num_out_degree,
            num_edges=num_edges,
            num_spatial=num_spatial,
            num_encoder_layers=num_encoder_layers,
            embedding_dim=embedding_dim,
            ffn_embedding_dim=ffn_embedding_dim,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
        )
        
        # Classification head only
        self.classifier = GraphormerClassifier(
            embedding_dim=embedding_dim,
            hidden_dim=classifier_hidden_dim,
            num_classes=num_classes,
            dropout=dropout
        )
        
        self.embedding_dim = embedding_dim
    
    def forward(self, batched_data, pyg_batch=None):
        """
        Args:
            batched_data: dict formatted for Graphormer
            pyg_batch: PyG batch object (not used, kept for compatibility)
        Returns:
            logits: [batch_size, num_classes] - class predictions
        """
        # Get graph representation from Graphormer
        _, graph_rep = self.graphormer(batched_data)
        
        # Prediction only
        logits = self.classifier(graph_rep)
        
        return logits


# ============================================================================
# Utility function to count parameters
# ============================================================================

def count_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick test
    print("GraphormerExplainer (Unified Model)")
    model = GraphormerExplainer(
        num_encoder_layers=4,
        embedding_dim=128,
        ffn_embedding_dim=128,
        num_attention_heads=4,
        num_classes=3
    )
    print(f"Total parameters: {count_parameters(model):,}")
    print(f"Graphormer encoder: {count_parameters(model.graphormer):,}")
    print(f"Classifier: {count_parameters(model.classifier):,}")
    print(f"Edge scorer: {count_parameters(model.edge_scorer):,}")
