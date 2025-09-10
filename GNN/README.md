# Graph Learning

This directory contains a Graph Neural Network (GNN) project notebook and supporting files.

## Overview
- **Framework:** scikit-network (graph learning)
- **Model:** `GNNClassifier`
- **Hidden dimension:** 16 (default in notebook)
- **Metrics:** accuracy
- **Epochs:** not explicitly set (training logs printed up to epoch 90)

## Data & Features
- **Inputs expected by the notebook:**
  - `adjacency` — sparse adjacency matrix of the graph.
  - `features` — node feature matrix (here, a **spectral embedding** of the bipartite graph is used).
  - `labels` — ground-truth node labels.
  - `mask_train`, `mask_test` — boolean masks to split nodes for training and testing.
- Optional preprocessing utilities used:
  - `directed2undirected` to convert a directed graph to undirected, when needed.

## Model Variants Tested
- **Baseline (no graph):** `gnn_empty.fit(empty, features, labels_empty)` to assess feature-only performance.
- **With graph structure:** `gnn.fit(adjacency, features, labels)` using the true adjacency.
- **Hidden size sweep:** a variant with `hidden_dim = 32` is compared to the default 16.
- **Deeper model:** a two-layer GNN (`gnn_two_layers`) is evaluated.
- **Non-graph baseline:** a logistic regression classifier trained on `features[mask_train]`.

## Model Definition (from the notebook)
```python
from sknetwork.gnn import GNNClassifier

hidden_dim = 16
gnn = GNNClassifier(dims=[hidden_dim, n_labels], verbose=True)
