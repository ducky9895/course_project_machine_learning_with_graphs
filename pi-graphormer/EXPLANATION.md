# Graphormer-PIGNN: How It Works

## Overview

**Graphormer-PIGNN** integrates Graphormer as the explainer component within the **π-GNN** (Pre-training Interpretable GNN) framework. It's a hybrid architecture that combines:

1. **Graphormer** (Transformer for graphs) - for learning rich node representations
2. **π-GNN framework** - for pre-training interpretable GNNs with ground-truth explanations

## Architecture Comparison

### Original Graphormer (`Graphormer/` folder)
The original Graphormer is a **standalone graph transformer** designed for graph-level tasks:

- **Purpose**: Graph representation learning and prediction
- **Output**: Graph-level embeddings for classification/regression
- **Key Features**:
  - Centrality encoding (in-degree, out-degree)
  - Spatial encoding (shortest path distances)
  - Edge encoding (edge features along paths)
  - Multi-head self-attention over all nodes
  - Graph token for global representation

### Graphormer-PIGNN (`graphormer-pignn/` folder)
Graphormer-PIGNN uses Graphormer **as a component** within an explainable GNN framework:

- **Purpose**: Graph classification with **explainability** (edge importance scores)
- **Output**: 
  1. Edge importance scores (explanation)
  2. Graph-level predictions
- **Key Features**:
  - Uses Graphormer encoder to learn node representations
  - Adds an **EdgeScorePredictor** (MLP) to compute edge importance
  - Uses a **GNNPredictor** (GCN) that respects graph structure
  - Trained with both explanation loss and prediction loss

## Detailed Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  GraphormerPIGNN Model                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │          GraphormerExplainer                        │    │
│  │                                                      │    │
│  │  ┌──────────────────────────────────────────────┐   │    │
│  │  │      GraphormerEncoder (from Graphormer)     │   │    │
│  │  │                                               │   │    │
│  │  │  Input: Graph data with:                     │   │    │
│  │  │    • Node features (x)                       │   │    │
│  │  │    • Centrality (in_degree, out_degree)      │   │    │
│  │  │    • Spatial positions (shortest paths)     │   │    │
│  │  │    • Edge types (attn_edge_type)             │   │    │
│  │  │                                               │   │    │
│  │  │  Process:                                     │   │    │
│  │  │    1. Encode node features + centrality      │   │    │
│  │  │    2. Compute attention bias (spatial+edge)  │   │    │
│  │  │    3. Multi-head self-attention layers       │   │    │
│  │  │                                               │   │    │
│  │  │  Output:                                      │   │    │
│  │  │    • Node representations [N, D]             │   │    │
│  │  │    • Graph representation [D]                 │   │    │
│  │  └──────────────────────────────────────────────┘   │    │
│  │                    ↓                                 │    │
│  │  ┌──────────────────────────────────────────────┐   │    │
│  │  │      EdgeScorePredictor (MLP)                │   │    │
│  │  │                                               │   │    │
│  │  │  For each edge (i,j):                        │   │    │
│  │  │    edge_feat = concat(node_i, node_j)        │   │    │
│  │  │    score = MLP(edge_feat) → sigmoid          │   │    │
│  │  │                                               │   │    │
│  │  │  Output: Edge scores ρ̂ ∈ [0,1]^|E|          │   │    │
│  │  └──────────────────────────────────────────────┘   │    │
│  └────────────────────────────────────────────────────┘    │
│                    ↓                                         │
│  ┌────────────────────────────────────────────────────┐    │
│  │              GNNPredictor (GCN)                     │    │
│  │                                                      │    │
│  │  Input:                                             │    │
│  │    • Node representations from Graphormer           │    │
│  │    • Edge scores as edge weights                   │    │
│  │                                                      │    │
│  │  Process:                                           │    │
│  │    1. GCN layers with weighted edges               │    │
│  │    2. Global mean pooling                          │    │
│  │    3. Classification head                          │    │
│  │                                                      │    │
│  │  Output: Class logits [B, C]                       │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## Key Implementation Details

### 1. Graphormer Encoder Implementation

**Does it really implement Graphormer?**

**YES**, but with some simplifications:

#### ✅ **What's Implemented Correctly:**

1. **Centrality Encoding** (`GraphNodeFeature`):
   - In-degree and out-degree embeddings
   - Node feature embeddings
   - Graph token for global representation
   - Same as original Graphormer

2. **Spatial Encoding** (`GraphAttnBias`):
   - Shortest path distance embeddings
   - Added to attention bias matrix
   - Same concept as original

3. **Edge Encoding**:
   - Edge type embeddings along shortest paths
   - Added to attention bias
   - Simplified version (no multi-hop edge distance encoding)

4. **Multi-head Self-Attention**:
   - Standard transformer attention mechanism
   - Attention bias injection
   - Pre-norm architecture

5. **Transformer Layers**:
   - Self-attention + FFN
   - Layer normalization
   - Residual connections

#### ⚠️ **Simplifications/Differences:**

1. **No Fairseq Dependency**:
   - Original Graphormer uses Fairseq framework
   - Graphormer-PIGNN is standalone PyTorch implementation
   - This is actually a **benefit** for easier integration

2. **Simplified Edge Encoding**:
   - Original supports `multi_hop` edge encoding with distance weighting
   - Graphormer-PIGNN uses simpler edge type encoding
   - Still captures edge information, just less sophisticated

3. **Data Preprocessing**:
   - Original uses optimized Cython code for shortest paths
   - Graphormer-PIGNN uses pure Python Floyd-Warshall
   - Slower but more portable

4. **Architecture Details**:
   - Original has more configuration options (layerdrop, quantization, etc.)
   - Graphormer-PIGNN focuses on core functionality
   - Sufficient for the π-GNN use case

### 2. Data Preprocessing Pipeline

The `data_utils.py` file converts PyTorch Geometric graphs to Graphormer format:

1. **Node Feature Encoding**:
   ```python
   x_emb = convert_to_single_emb(x_discrete, offset=64, max_value=511)
   ```
   - Converts multi-feature nodes to single embedding indices
   - Handles continuous features by discretization

2. **Shortest Path Computation**:
   ```python
   dist, path = floyd_warshall_python(adj_np)
   ```
   - Computes all-pairs shortest paths
   - Used for spatial encoding

3. **Edge Input Generation**:
   - Extracts edge features along shortest paths
   - Creates `edge_input` tensor for multi-hop edge encoding

4. **Batching** (`GraphormerCollator`):
   - Pads graphs to same size
   - Creates batched tensors for Graphormer
   - Preserves PyG batch structure for edge indexing

### 3. Training Pipeline

The training follows the π-GNN framework:

**Phase 1: Pre-training (on synthetic data)**
- Train explainer on synthetic graphs with ground-truth explanations
- Loss: `exp_weight * BCE(edge_scores, gt_labels) + pred_weight * CE(predictions, labels)`
- Learn universal structural patterns

**Phase 2: Fine-tuning (on downstream task)**
- Load pre-trained explainer weights
- Fine-tune both explainer and predictor
- Add entropy regularization for sparser explanations

## Comparison Table

| Aspect | Original Graphormer | Graphormer-PIGNN |
|--------|---------------------|------------------|
| **Purpose** | Graph representation learning | Explainable graph classification |
| **Output** | Graph embeddings | Edge scores + predictions |
| **Architecture** | Encoder only | Encoder + Edge Scorer + Predictor |
| **Training** | Supervised on task labels | Supervised on labels + explanations |
| **Dependencies** | Fairseq | Standalone PyTorch |
| **Edge Encoding** | Full multi-hop support | Simplified version |
| **Use Case** | General graph tasks | Interpretable GNNs |

## Key Design Choices

### 1. **Edge Scoring Strategy**
- **Option B**: Graphormer learns node representations, then MLP scores edges
- Only scores **actual edges** (not all node pairs)
- More efficient and respects graph structure

### 2. **Predictor Architecture**
- Uses **sparse GCN** with edge scores as weights
- Naturally respects graph structure
- No spurious edge weights (only real edges)

### 3. **Integration Approach**
- Graphormer encoder is **reused** as-is
- Adds explainer and predictor components
- Maintains Graphormer's structural encoding benefits

## Advantages of Graphormer-PIGNN

1. **Rich Structural Encodings**: Inherits Graphormer's centrality, spatial, and edge encodings
2. **Global Receptive Field**: Attention over all nodes (unlike local GNNs)
3. **Explainability**: Provides edge importance scores
4. **Pre-training**: Can learn universal patterns on synthetic data
5. **Standalone**: No Fairseq dependency, easier to use

## Potential Challenges

1. **Computational Cost**: Quadratic attention complexity (O(N²))
2. **Memory**: Requires padding to max graph size in batch
3. **Hyperparameters**: More parameters to tune than simple GNNs
4. **Data Requirements**: May need more data for pre-training

## Conclusion

**Yes, Graphormer-PIGNN really implements Graphormer**, with the core components faithfully reproduced:

- ✅ Centrality encoding
- ✅ Spatial encoding (shortest paths)
- ✅ Edge encoding
- ✅ Multi-head self-attention
- ✅ Transformer architecture

The main differences are:
- **Simplified** edge encoding (no multi-hop distance weighting)
- **Standalone** implementation (no Fairseq)
- **Integrated** into π-GNN framework (adds explainer/predictor)

The implementation is **sufficient and correct** for the π-GNN use case, providing the structural encoding benefits of Graphormer while adding explainability capabilities.
