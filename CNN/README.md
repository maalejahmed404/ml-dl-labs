# CNN Notebook

This directory contains a Convolutional Neural Network (CNN) project notebook and supporting files.

## Overview
- **Framework:** TensorFlow/Keras
- **Epochs:** 100
- **Batch size:** 64
- **Learning rate:** 0.0005
- **Metrics:** AUC, Precision, Recall
- **Random seed:** 42

## Data
The notebook references the following data paths/files:
- `data/`
- `train/`
- `test/`

Update the paths as needed for your environment.

## Model Architecture (detected layer calls)
```python
Conv1D(...)
BatchNormalization(...)
MaxPooling1D(...)
Conv1D(...)
BatchNormalization(...)
MaxPooling1D(...)
Conv1D(...)
BatchNormalization(...)
MaxPooling1D(...)
Conv1D(...)
BatchNormalization(...)
GlobalAveragePooling1D(...)
Dense(256, activation='relu')
Dropout(...)
Dense(128, activation='relu')
Dropout(...)
Dense(num_classes, activation='softmax')
