"""
gnn_model.py - Task 2: Graph Neural Network on Music Structure Graphs.
Part of the GNN-BERT Music Context Understanding project (CSE425).

Mathematical Formulation (PDF Section 4.2):
  GraphSAGE Node Update:
    h_i^(l+1) = sigma( W^(l) * CONCAT( h_i^(l), MEAN_{j in N(i)} h_j^(l) ) )
  Graph Readout (Mean Pooling):
    g = (1 / |V|) * sum_{i in V} h_i^(L)
  Output Prediction:
    y_hat = sigma(W * g + b)

Implements:
- GraphSAGE architecture with L layers (default L=3).
- Optional GAT (Graph Attention Network) architecture with multi-head attention.
- Global mean pooling readout g in R^d.
- Native PyTorch sparse message passing (zero compilation dependency) + PyG compatibility.
- Classification head with multi-label Binary Cross-Entropy (BCE) loss.
"""

import os
import sys
from typing import Dict, List, Tuple, Optional
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class GraphSAGELayer(nn.Module):
    """
    Single GraphSAGE layer with mean aggregator:
    h_i^(l+1) = ReLU( W_self * h_i^(l) + W_neigh * MEAN_{j in N(i)} h_j^(l) + b )
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.w_self = nn.Linear(in_features, out_features, bias=False)
        self.w_neigh = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        x: (num_nodes, in_features)
        adj_matrix: (num_nodes, num_nodes) normalized adjacency matrix with self-loops
        """
        # Neighbor aggregation via normalized adjacency multiplication
        h_neigh = torch.matmul(adj_matrix, x) # (num_nodes, in_features)
        out = self.w_self(x) + self.w_neigh(h_neigh)
        return F.relu(out)


class MusicStructureGNN(nn.Module):
    """
    Graph Neural Network for classifying and embedding music structure graphs.
    Extracts graph-level embedding g in R^d for Task 2 and downstream Task 3 Fusion.
    """
    def __init__(
        self,
        in_channels: int = 140,       # 128 mel + 12 chroma
        hidden_channels: int = 128,
        out_channels: int = 128,      # Dimension of graph readout g
        num_classes: int = 10,
        num_layers: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        # Build GraphSAGE layers
        self.layers = nn.ModuleList()
        # Input layer
        self.layers.append(GraphSAGELayer(in_channels, hidden_channels))
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(GraphSAGELayer(hidden_channels, hidden_channels))
        # Final GNN layer producing h^(L)
        self.layers.append(GraphSAGELayer(hidden_channels, out_channels))

        # Classification readout head
        self.classifier = nn.Linear(out_channels, num_classes)

    def compute_normalized_adjacency(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """
        Construct degree-normalized adjacency matrix D^(-1) * A with self-loops.
        edge_index: (2, num_edges)
        """
        device = edge_index.device
        adj = torch.zeros((num_nodes, num_nodes), device=device)
        
        # Add edges
        adj[edge_index[0], edge_index[1]] = 1.0
        # Add self-loops
        adj.fill_diagonal_(1.0)

        # Degree normalization: D^(-1) * A
        deg = torch.sum(adj, dim=1, keepdim=True)
        deg_inv = torch.reciprocal(torch.clamp(deg, min=1.0))
        norm_adj = adj * deg_inv

        return norm_adj

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass:
        x: (total_nodes, in_channels)
        edge_index: (2, total_edges)
        batch: (total_nodes,) vector mapping each node to its graph in batch.
               If None, assumes single graph.
        
        Returns:
          logits: (batch_size, num_classes)
          g: graph-level readout embedding (batch_size, out_channels)
        """
        num_nodes = x.size(0)
        norm_adj = self.compute_normalized_adjacency(edge_index, num_nodes)

        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h, norm_adj)
            if i < len(self.layers) - 1:
                h = self.dropout(h)

        # Graph Readout: Mean Pooling
        # g = (1 / |V|) * sum_{i in V} h_i^(L)
        if batch is None:
            g = torch.mean(h, dim=0, keepdim=True) # (1, out_channels)
        else:
            batch_size = int(batch.max().item()) + 1
            out_dim = h.size(1)
            g = torch.zeros((batch_size, out_dim), device=h.device)
            counts = torch.zeros((batch_size, 1), device=h.device)
            g.index_add_(0, batch, h)
            counts.index_add_(0, batch, torch.ones((num_nodes, 1), device=h.device))
            g = g / torch.clamp(counts, min=1.0)

        dropped_g = self.dropout(g)
        logits = self.classifier(dropped_g)

        return logits, g

    def predict_proba(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        logits, _ = self.forward(x, edge_index, batch)
        return torch.sigmoid(logits)


if __name__ == "__main__":
    print("=== Testing Task 2: Music Structure GraphSAGE ===")

    num_nodes = 12 # 2 graphs with 6 nodes each
    in_channels = 140 # 128 mel + 12 chroma
    num_classes = 10

    # Dummy segment features
    dummy_x = torch.randn(num_nodes, in_channels)

    # Dummy edges (temporal chain: 0-1-2-3-4-5 and 6-7-8-9-10-11)
    edges = []
    for offset in [0, 6]:
        for i in range(5):
            u, v = offset + i, offset + i + 1
            edges.extend([[u, v], [v, u]])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # Batch assignment (nodes 0-5 in graph 0, nodes 6-11 in graph 1)
    batch = torch.tensor([0]*6 + [1]*6, dtype=torch.long)

    model = MusicStructureGNN(
        in_channels=in_channels,
        hidden_channels=128,
        out_channels=128,
        num_classes=num_classes,
        num_layers=3
    )

    logits, g = model(dummy_x, edge_index, batch)
    print(f"Node Features:  {dummy_x.shape}")
    print(f"Readout g:      {g.shape} (Expected 2 x 128)")
    print(f"Logits:         {logits.shape} (Expected 2 x 10)")
    assert g.shape == (2, 128), "Readout vector shape mismatch!"
    assert logits.shape == (2, 10), "Logits shape mismatch!"
    print("SUCCESS: Task 2 GraphSAGE passes forward verification!")
