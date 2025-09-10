# Multi-Layer Perceptron (PyTorch)

This folder contains a PyTorch implementation of a **Multi-Layer Perceptron (MLP)**.  
It includes two variants of the same idea:
- **Classification** (e.g., MNIST digits)
- **Regression** (toy/tabular-style targets)

> Use the notebook in this folder (`*.ipynb`) and run cells in order.

---

## Overview
- **Framework:** PyTorch (+ optional torchvision utilities)
- **Task:** Classification _or_ Regression (depending on the notebook)
- **Architecture:** Fully-connected layers with non-linear activations (ReLU/Tanh)  
- **Optimizer:** SGD (configurable)
- **Loss:**
  - Classification: NLL (via `F.log_softmax` + `nn.NLLLoss`) or CrossEntropy
  - Regression: `nn.MSELoss`
- **Metrics:**
  - Classification: Accuracy
  - Regression: R² (coefficient of determination)

---

## Data
- **Classification:** typically uses `torchvision.datasets` (e.g., **MNIST**).
- **Regression:** uses a small in-notebook dataset (no external files).
- Dataloaders handle batching and shuffling where relevant.

---

## Model (typical patterns)
```python
# Example block-style MLP
self.fc1 = nn.Linear(n_in, n_h1)
self.fc2 = nn.Linear(n_h1, n_h2)          # omit for 1-hidden-layer model
self.fc3 = nn.Linear(n_h2, n_out)         # or Linear(n_h1, n_out)

# Example Sequential for regression
model = nn.Sequential(
    nn.Linear(n_in, n_h),
    nn.Tanh(),                             # or ReLU
    nn.Linear(n_h, n_out)
)

# Heads / outputs:
# - Classification: use F.log_softmax on output and nn.NLLLoss
# - Regression: raw linear output with nn.MSELoss
