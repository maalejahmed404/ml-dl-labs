## Process

1. **Data preparation**
   - Split with `train_test_split`.
   - Cleaning: lowercasing, regex cleanup, stopword removal, lemmatization/stemming, tokenization.

2. **Classic ML baseline**
   - Features: `TfidfVectorizer` / `CountVectorizer`.
   - Classifier: `LogisticRegression`.
   - Metrics: Accuracy, F1, plus full `classification_report`.

3. **Embedding baseline**
   - Word embeddings (e.g., **GloVe** via Gensim).
   - Sentence/document vectors formed by aggregating token embeddings (e.g., mean).
   - Classifier: `LogisticRegression` (same evaluation protocol).

4. **Transformer fine-tuning**
   - Model: **`roberta-base`** with `AutoTokenizer` + `AutoModelForSequenceClassification`.
   - Trainer: Hugging Face `Trainer` with:
     - `num_train_epochs=3`, `per_device_train_batch_size=16`, `per_device_eval_batch_size=16`,
     - `learning_rate=2e-5`, `max_length=200`, truncation enabled.
   - Metric callback (`compute_metrics`) to report Accuracy and F1 on the validation/test split.

> Notes: No class-imbalance handling is enabled by default (no weights/SMOTE). Set seeds if you need strict reproducibility.

---

## Results

| Pipeline                         | Accuracy (≈) | Support (test) | Notes                                |
|----------------------------------|--------------|----------------|--------------------------------------|
| TF-IDF / CountVectorizer + LR    | 0.8024–0.8028| 2,201          | Classic linear baseline              |
| GloVe embeddings + LR            | 0.8406       | 7,317          | Embedding aggregation + linear model |
| `roberta-base` (fine-tuned)      | 0.9147       | 8,800          | Transformer fine-tune with Trainer   |

- The notebook prints scikit-learn classification reports (precision/recall/F1 per class and **overall accuracy**).
- These numbers are taken from the notebook outputs; they can vary with data splits, random seeds, and preprocessing choices.
- F1 is computed in the `compute_metrics` function for the Transformer run; summarize your best F1 here after execution if needed.

**Takeaway:** Each step up the modeling ladder improves performance — TF-IDF → GloVe → fine-tuned `roberta-base` — with the Transformer achieving the strongest accuracy on this dataset and setup.
