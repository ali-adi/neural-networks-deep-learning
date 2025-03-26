# 🎙️ Speech Emotion Recognition (SER) Project

Welcome to the Speech Emotion Recognition (SER) project repository for SC4001 / CZ4042! 🚀

In this project, we leverage deep learning techniques to accurately classify emotional states from speech audio signals. The objective is to extract meaningful audio features, train robust models, and evaluate their performance. 🧠🎶

## 📁 Repository Structure

```
.
├── data                   # Dataset storage
│   ├── raw                # Original unprocessed audio files
│   │   └── EMODB
│   ├── processed          # Processed and labeled datasets
│   │   ├── EMODB
│   │   └── EMODB_MFCC_96
│   └── MFCC               # MFCC feature extraction intermediate files
├── results                # Experiment results (performance metrics, spreadsheets)
├── saved_models           # Trained model weights and checkpoints
├── test_models            # Models specifically for testing/evaluation
├── src                    # Source code
│   ├── data_processing    # Scripts for data preparation and feature extraction
│   ├── models             # Model architecture definitions
│   ├── training           # Training and evaluation scripts
│   └── utils              # Utility scripts (logging, configuration management, etc.)
├── EMODB.npy              # Processed dataset for model training
├── README.md              # Documentation guide (you're here!) 🗺️
└── requirements.txt       # Python dependencies
```

## 🛠️ Environment Setup

Let's set up your working environment (Python 3.11.11):

```bash
# Set up local python environment
pyenv install 3.11.11
pyenv local 3.11.11

# Create and activate a Python virtual environment
python3 -m venv env
source env/bin/activate

# Install project dependencies
pip install -r requirements.txt
```

✅ You're now ready to proceed!

## 📋 Step-by-Step Guide

### 1️⃣ Data Organization 📂

Organize your raw audio dataset into emotion-labeled subdirectories:

```bash
python -m src.data_processing.reorganize_data \
  --source_dir data/raw/EMODB \
  --dest_dir data/processed/EMODB
```

**Available Arguments:**
- `--source_dir` (required): Path containing the original `.wav` audio files.
- `--dest_dir` (required): Destination path for creating emotion-labeled subdirectories.

This prepares your data structure for efficient feature extraction.

### 2️⃣ Feature Extraction (MFCC) 🎶

Extract Mel Frequency Cepstral Coefficients (MFCCs) from audio files:

```bash
python -m src.data_processing.extract_feature \
  --data_name EMODB \
  --mean_signal_length 96000 \
  --embed_len 39 \
  --output_dir data/processed/EMODB_MFCC_96
```

**Available Arguments:**
- `--data_name`: Name of the dataset (default: `EMODB`).
- `--mean_signal_length`: Length (in samples) to pad or crop each audio signal (default: `96000`).
- `--feature_type`: Type of audio feature to extract, currently supports `MFCC` (default: `MFCC`).
- `--embed_len`: Number of MFCC coefficients to extract (default: `39`).
- `--output_dir`: Directory to store CSV files of extracted features (default: `./EMODB_MFCC_96`).

This script generates `EMODB.npy`, containing numerical features for training.

### 3️⃣ Model Training 🚀

Train your speech emotion recognition model using cross-validation:

```bash
python -m src.training.main \
  --mode train \
  --data EMODB \
  --epoch 10 \
  --batch_size 64 \
  --split_fold 5 \
  --model_path ./saved_models/ \
  --result_path ./results/
```

**Available Arguments:**
- `--mode`: Run mode, either `train` or `test` (default: `train`).
- `--model_path`: Path to save model checkpoints (default: `./saved_models/`).
- `--result_path`: Path to save performance metrics and confusion matrices (default: `./results/`).
- `--data`: Dataset to use (default: `EMODB`).
- `--epoch`: Number of epochs for training (default: `500`).
- `--batch_size`: Batch size for training (default: `64`).
- `--split_fold`: Number of folds for cross-validation (default: `10`).
- `--lr`: Learning rate (default: `0.001`).
- `--dropout`: Dropout rate (default: `0.1`).
- `--activation`: Activation function (`relu`, etc.) (default: `relu`).
- `--filter_size`: Number of convolutional filters (default: `39`).
- `--dilation_size`: Maximum dilation rate for convolutions (default: `8`).
- `--kernel_size`: Kernel size for convolutions (default: `2`).
- `--stack_size`: Number of stacked blocks (default: `1`).
- `--random_seed`: Seed for reproducibility (default: `46`).
- `--gpu`: GPU ID to use (default: `0`).

Customize hyperparameters as needed.

### 4️⃣ Model Evaluation 🧪

Evaluate your trained model:

```bash
python -m src.training.main \
  --mode test \
  --data EMODB \
  --split_fold 5 \
  --test_path ./saved_models/EMODB_46_2025-03-26_17-28-31
```

**Available Arguments (additional to training):**
- `--test_path`: Directory containing the trained model weights (default: `./test_models/EMODB_46`).

Evaluate model generalization and effectiveness.

## 📌 Important Notes

- ⚠️ Execute commands from the project root directory.
- 🔧 Adjust paths and parameters to align with your local environment.
- 📚 Use the `--help` flag with scripts for additional details.

Happy exploring! 🎧✨


