# Web Retrival and Mining 2025 PA2: Personalized Item Recommendation System

This repository contains the implementation of **Matrix Factorization** models for personalized item recommendation. We compare two loss functions: **BPR (Bayesian Personalized Ranking)** and **BCE (Binary Cross Entropy)**.

## 📂 Folder Structure

```plaintext
.
├── main.py                     # Main training script
├── model.py               # BPRMF model definition
├── dataset.py                 # Dataset preprocessing and dataloader
├── utils.py                   # MAP@50 metric and helper functions
├── run.sh                     # Single experiment runner
├── output/                    # Output predictions
├── weights/                   # Saved model checkpoints
└── README.md                  # This file
```

## 🚀 Quick Start
**🔧 Environment**

- Python 3.10+
- PyTorch
- NumPy
- tqdm
- matplotlib (for plotting)

**📦 Installation**

```bash
pip install -r requirements.txt
```

## ⚙️ Command Line Options (main.py)
| Argument          | Description                                 | Default      |
|-------------------|---------------------------------------------|--------------|
| `--train_path`    | Path to training data CSV                   | `train.csv`  |
| `--output_path`   | Path to submission output CSV               | `output.csv` |
| `--loss`          | Loss type: `bpr` or `bce`                   | `bpr`        |
| `--embedding_dim` | Latent embedding dimension size             | `64`         |
| `--neg`           | Number of negative samples per positive     | `3`          |
| `--epochs`        | Number of training epochs                   | `20`         |
| `--batch_size`    | Size of each training batch                 | `1024`       |

