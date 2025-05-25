# Hub9 Assignment

This repository contains two Jupyter notebooks for the Hub9 assignment:

## Files

- [heartbeat-cnn-inder.ipynb](heartbeat-cnn-inder.ipynb)  
  Implements and analyzes deep learning models (1D and 2D CNNs) for heartbeat classification using ECG datasets. The notebook covers:
  - Data loading and preprocessing
  - Handling class imbalance (SMOTE, class weights)
  - Model building (1D CNN and 2D CNN with spectrograms)
  - Training with early stopping
  - Evaluation (accuracy, confusion matrix, classification report)
  - Visualization of results

- [fine-tuning-llama-3-2-1b-instruct.ipynb](fine-tuning-llama-3-2-1b-instruct.ipynb)  
  Demonstrates fine-tuning of the Llama 3 2.1B Instruct language model on a text summarization dataset. The notebook includes:
  - Data loading and preprocessing (CNN/DailyMail dataset)
  - Model setup and tokenizer configuration
  - Training and evaluation
  - Inference and result analysis

## Instructions

1. Open the notebooks in [Visual Studio Code](https://code.visualstudio.com/) or [Jupyter Notebook](https://jupyter.org/).
2. Follow the code cells and markdown explanations to understand the workflow and results.
3. Ensure you have the required dependencies installed (TensorFlow, scikit-learn, pandas, matplotlib, seaborn, imbalanced-learn, transformers, etc.).

## Datasets

- Heartbeat datasets are loaded from Kaggle input directories (see notebook for details).
- The Llama 3 fine-tuning notebook uses the CNN/DailyMail summarization dataset.

## Requirements

- Python 3.10+
- Jupyter Notebook or VS Code with Jupyter extension
- TensorFlow, scikit-learn, pandas, matplotlib, seaborn, imbalanced-learn, transformers, etc.

---

**Author:** Inder  
**Assignment:** Hub9