# Recurrent Neural Networks (PyTorch) — Classification & Generation

This directory contains two notebooks:
- **RNN-classification.ipynb** — supervised text/sequence classification with Embedding + LSTM.
- **RNN-generation.ipynb** — language modeling / text generation with stacked LSTMs.

Some cells may import Keras/TensorFlow utilities, but models and training loops are in **PyTorch**.

---

## Environment

    python -m venv .venv
    # Windows: .venv\Scripts\activate
    source .venv/bin/activate

    pip install torch numpy matplotlib
    # Optional: pip install torchtext  # only if you use it
    # Optional: pip install tensorflow # only if a Keras/TensorFlow cell is required

Open the notebooks with:

    jupyter lab    # or: jupyter notebook

---

## Quick Start

### 1) Classification (RNN-classification.ipynb)

1. **Preprocess**
   - Tokenize text; build vocabulary `n_word`.
   - Integer-encode + pad sequences.
   - Create train/val/test splits; wrap in `DataLoader(batch_size=64)`.

2. **Model**

        embedding = nn.Embedding(n_word, n_embedding)
        encoder   = nn.LSTM(input_size=n_embedding, hidden_size=n_lstm, batch_first=True)
        head      = nn.Linear(n_lstm, n_classes)  # logits

3. **Train**
   - Loss: `nn.CrossEntropyLoss()` (or `F.log_softmax` + `nn.NLLLoss`).
   - Optimizer: SGD/Adam (default LR ≈ **0.001**).

4. **Evaluate**
   - Report **accuracy** (and macro-F1 if classes are imbalanced).

### 2) Generation (RNN-generation.ipynb)

1. **Preprocess**
   - Build character/token vocabulary; integer-encode corpus.
   - Create sliding windows (context → next token).

2. **Model**

        lstm1 = nn.LSTM(n_x,  n_a, batch_first=True)
        lstm2 = nn.LSTM(n_a,  n_a, batch_first=True)
        lstm3 = nn.LSTM(n_a,  n_a, batch_first=True)
        head  = nn.Linear(n_a, vocab_size)        # logits

3. **Train**
   - Objective: next-token cross-entropy.
   - Optimizer: e.g., SGD (default LR ≈ **0.01**), `batch_size=64`.

4. **Generate**
   - Provide a prompt; sample with temperature (and optionally top-k/top-p).

---

## Architecture (detected)

### Classification (Embedding + LSTM → Head)

    self.embedding = nn.Embedding(n_word, n_embedding)
    self.lstm      = nn.LSTM(input_size=n_embedding, hidden_size=n_lstm, batch_first=True)
    # pooled / last hidden state → Linear → logits

### Generation (Stacked LSTMs → Projection)

    self.lstm1 = nn.LSTM(n_x,  n_a, batch_first=True)
    self.lstm2 = nn.LSTM(n_a,  n_a, batch_first=True)
    self.lstm3 = nn.LSTM(n_a,  n_a, batch_first=True)
    self.head  = nn.Linear(n_a, vocab_size)  # logits

---

## Hyperparameters (defaults to tune)

| Notebook            | Batch Size | Learning Rate | Hidden Size | Layers | Notes                 |
|---------------------|-----------:|--------------:|------------:|-------:|-----------------------|
| RNN-classification  | 64         | 0.001         | `n_lstm`    |   1    | Add dropout if needed |
| RNN-generation      | 64         | 0.01          | `n_a`       |   3    | Temperature sampling  |

**Tips**
- Try **bidirectional LSTM** for classification.
- Adjust sequence length, hidden size, and dropout for better generalization.
- For generation, tune temperature (e.g., 0.7–1.2) and optionally top-k/top-p.

---

## Results (fill after running)

**Classification**

    Validation accuracy: ...
    Test accuracy: ...
    Macro-F1 (test): ...

**Generation**

    Training loss / Validation loss: ...
    (Optionally) Perplexity: ...
    Sample output (a few lines):
    <paste a short generated snippet>

---

## Dependencies (minimum)

- torch
- numpy
- matplotlib
- torchtext (optional; only if used)
- tensorflow (optional; only if a Keras/TensorFlow cell is required)
