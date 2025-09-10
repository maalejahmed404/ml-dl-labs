# TPs — Machine Learning & Deep Learning Labs

This repository gathers my **Travaux Pratiques (TPs)** across core ML/DL topics:
- **CNN** for image/signal modeling,
- **GNN** for learning on graphs,
- **MLP** for classification & regression,
- **RNN** for sequence classification & text generation,
- **NLP** end-to-end text classification (classic features → Transformers).

Each TP is self-contained with a notebook and a minimal `requirements.txt`. See each folder’s README for details.

---

## Structure

TPs/
├─ CNN/
│  ├─ CNN.ipynb
│  ├─ README.md
│  └─ requirements.txt
├─ GNN/
│  ├─ GNN.ipynb
│  ├─ README.md
│  └─ requirements.txt
├─ MLP-CLASS/
│  ├─ MLP-CLASS.ipynb
│  ├─ README.md
│  └─ requirements.txt
├─ MLP-REGRESSION/
│  ├─ MLP-REGRESSION.ipynb
│  ├─ README.md
│  └─ requirements.txt
├─ NLP/
│  ├─ NLP.ipynb
│  ├─ README.md
│  └─ requirements.txt
└─ RNN/
   ├─ RNN-classification.ipynb
   ├─ RNN-generation.ipynb
   └─ README.md

---

## Contents Overview

| Folder | Topic | Main Framework(s) | Highlights |
|---|---|---|---|
| `CNN/` | Convolutional Neural Networks | TensorFlow/Keras | 1D/2D CNN pipeline, training & evaluation (AUC, Precision, Recall). |
| `GNN/` | Graph Neural Networks | scikit-network | `GNNClassifier` with spectral features; graph vs. feature-only ablations. |
| `MLP-CLASS/` | MLP for Classification | PyTorch | MNIST MLP, SGD training, accuracy reporting. |
| `MLP-REGRESSION/` | MLP for Regression | PyTorch | Sequential & class-based MLPs, `MSELoss`, R² metric. |
| `NLP/` | Text Classification | PyTorch, scikit-learn, HF Transformers | Baselines (TF-IDF + LR), GloVe + LR, and `roberta-base` fine-tune. Approx results: TF-IDF ≈ 0.80 acc, GloVe ≈ 0.84, Transformer ≈ 0.91. |
| `RNN/` | RNNs (LSTM) | PyTorch | Text classification (Embedding + LSTM) and text generation (stacked LSTMs with sampling). |

---
