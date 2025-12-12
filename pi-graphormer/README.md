# Pi-Graphormer

## Overview

**Graphormer as the main predictor** with integrated edge-scoring explainer. No separate GNN predictor needed.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                             Pi-Graphormer                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │              GraphormerEncoder (Core Model)                      │   │
│  │                                                                  │   │
│  │  Input: Graph data                                               │   │
│  │    • Node features (x)                                           │   │
│  │    • Centrality (in_degree, out_degree)                          │   │
│  │    • Spatial positions (shortest paths)                          │   │
│  │    • Edge types (attn_edge_type)                                 │   │
│  │                                                                  │   │
│  │  Process:                                                        │   │
│  │    1. GraphNodeFeature: Encode nodes + centrality                │   │
│  │    2. GraphAttnBias: Compute spatial + edge encodings            │   │
│  │    3. Multi-head self-attention layers (N layers)                │   │
│  │                                                                  │   │
│  │  Output:                                                         │   │
│  │    • graph_rep: [B, D] (from graph token) ←──────┐               │   │
│  │    • node_rep: [B, N, D]                         │               │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                    ↓                        ↓                           │
│  ┌──────────────────────────┐  ┌──────────────────────────────┐         │
│  │  GraphormerClassifier    │  │  EdgeScorePredictor          │         │
│  │                          │  │                              │         │
│  │  graph_rep [B, D]        │  │  node_rep [B, N, D]          │         │
│  │       ↓                  │  │       ↓                      │         │
│  │  MLP:                    │  │  Flatten to [total_N, D]     │         │
│  │    D → hidden → hidden   │  │       ↓                      │         │
│  │       ↓                  │  │  For each edge (i,j):        │         │
│  │  logits [B, C]           │  │    concat(node_i, node_j)    │         │
│  │                          │  │    → MLP → sigmoid           │         │
│  │  ✓ Predictions           │  │       ↓                      │         │
│  └──────────────────────────┘  │  edge_scores [E]             │         │
│                                │  ✓ Explanations              │         │
│                                └──────────────────────────────┘         │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │          Explanation Regularizers                                │   │
│  │                                                                  │   │
│  │  1. Sparsity Loss:                                               │   │
│  │     • L1: ||edge_scores||₁                                       │   │
│  │     • Entropy: -Σ(p log p + (1-p) log(1-p))                      │   │
│  │                                                                  │   │
│  │  2. Connectivity Loss:                                           │   │
│  │     • Ensure top-k edges form connected components               │   │
│  │     • Penalize isolated nodes                                    │   │
│  │                                                                  │   │
│  │  3. Size Loss:                                                   │   │
│  │     • Encourage specific number of important edges               │   │
│  │     • ||sum(edge_scores) - target_size||²                        │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. GraphormerEncoder
- **Purpose**: Learn rich node and graph representations
- **Input**: Graph with structural encodings
- **Output**: 
  - `graph_rep`: Graph-level representation (from graph token)
  - `node_rep`: Node-level representations

### 2. GraphormerClassifier
- **Purpose**: Predict graph labels
- **Input**: `graph_rep` from Graphormer
- **Output**: Class logits `[batch_size, num_classes]`
- **Architecture**: Simple MLP (2-3 layers)

### 3. EdgeScorePredictor
- **Purpose**: Compute edge importance scores
- **Input**: `node_rep` from Graphormer
- **Output**: Edge scores `[num_edges]` in [0, 1]
- **Method**: MLP on concatenated node pairs

### 4. ExplanationRegularizer
- **Purpose**: Encourage desirable explanation properties
- **Components**:
  - Sparsity: Encourage sparse explanations
  - Connectivity: Ensure explanations form connected components
  - Size: Control number of important edges

## Data Flow

```
Input Graph (PyG)
    ↓
Preprocessing (data_utils.py)
    ↓
Graphormer Format:
  • x: [B, N, F]
  • attn_bias: [B, N+1, N+1]
  • spatial_pos: [B, N, N]
  • attn_edge_type: [B, N, N, E]
  • in_degree, out_degree: [B, N]
    ↓
GraphormerEncoder
    ↓
    ├─→ graph_rep [B, D] ──→ GraphormerClassifier ──→ logits [B, C]
    │
    └─→ node_rep [B, N, D] ──→ Flatten ──→ EdgeScorePredictor ──→ edge_scores [E]
                                                                        ↓
                                                           ExplanationRegularizer
```

## Loss Function

```
Total Loss = Prediction Loss + Explanation Loss + Regularization Loss

L_total = λ_pred * L_pred + λ_exp * L_exp + L_reg

Where:
  L_pred = CrossEntropy(logits, labels)
  L_exp  = BinaryCrossEntropy(edge_scores, gt_edge_labels)  # if available
  L_reg  = λ_sparse * L_sparse + λ_conn * L_conn + λ_size * L_size
```

## Comparison: v1 vs v2

| Aspect | v1 (Original) | v2 (Simplified) |
|--------|---------------|------------------|
| **Predictor** | Separate GNNPredictor (GCN) | GraphormerClassifier (MLP on graph_rep) |
| **Architecture** | Graphormer → GNN | Graphormer only |
| **Complexity** | Higher (2 models) | Lower (1 model) |
| **Naturalness** | Less natural | More natural (uses graph token) |
| **Efficiency** | 2 forward passes | 1 forward pass |
| **Regularization** | Basic (entropy) | Advanced (sparsity, connectivity, size) |
| **Use Case** | π-GNN framework | Course project / experiment |

## Advantages of v2

1. **Simpler**: One unified model instead of two
2. **More Natural**: Graphormer's graph token is designed for graph-level tasks
3. **Efficient**: Single forward pass through Graphormer
4. **Direct**: Edge scores explain Graphormer's predictions directly
5. **Flexible**: Easy to add more regularizers
6. **Educational**: Better for understanding Graphormer

## Usage Example

```python
from model_v2 import GraphormerExplainer

# Create model
model = GraphormerExplainer(
    num_encoder_layers=4,
    embedding_dim=128,
    num_attention_heads=4,
    num_classes=3,
    use_regularization=True
)

# Forward pass
logits, edge_scores, reg_losses = model(
    batched_data, 
    pyg_batch, 
    return_explanation=True,
    return_regularization=True
)

# Loss computation
pred_loss = F.cross_entropy(logits, labels)
exp_loss = F.binary_cross_entropy(edge_scores, gt_edge_labels)
total_loss = pred_loss + exp_loss + reg_losses['total']
```


## Citation

If you use this code, please cite:

```bibtex
@article{yin2023pignn,
  title={Train Once and Explain Everywhere: Pre-training Interpretable Graph Neural Networks},
  author={Yin, Jun and Li, Chaozhuo and Yan, Hao and Lian, Jianxun and Wang, Senzhang},
  journal={NeurIPS},
  year={2023}
}

@article{ying2021graphormer,
  title={Do Transformers Really Perform Bad for Graph Representation?},
  author={Ying, Chengxuan and Cai, Tianle and Luo, Shengjie and Zheng, Shuxin and Ke, Guolin and He, Di and Shen, Yanming and Liu, Tie-Yan},
  journal={NeurIPS},
  year={2021}
}
```

## License

MIT License
