# Speech Emotion Recognition (SER) System

A deep learning-based system for recognizing emotions in speech using FUSION features (LogMel + HuBERT) for optimal performance, with support for domain adaptation between datasets.

## 🚀 Features

- FUSION feature extraction combining:
  - LogMel spectrograms (128 mel bands)
  - HuBERT embeddings (768 dimensions)
- Deep learning model with dilated convolutional layers
- K-fold cross-validation
- Cross-corpus evaluation support (EMODB ↔ RAVDESS)
- Comprehensive evaluation metrics
- Support for multiple datasets (EMODB, RAVDESS)
- **NEW**: Domain adaptation using LMMD (Local Maximum Mean Discrepancy) loss
- **NEW**: Robust cross-corpus testing with automatic emotion mapping

## 📁 Project Structure

```
.
├── data/
│   ├── raw/                    # Original audio files
│   └── features/               # Extracted features
│       ├── LOGMEL/            # LogMel spectrograms
│       │   ├── EMODB_LOGMEL_128/
│       │   └── RAVDESS_LOGMEL_128/
│       ├── HUBERT/            # HuBERT embeddings
│       │   ├── EMODB_HUBERT/
│       │   └── RAVDESS_HUBERT/
│       └── FUSION/            # Combined features
│           ├── EMODB_FUSION/
│           └── RAVDESS_FUSION/
├── src/
│   ├── data_processing/       # Data preprocessing scripts
│   ├── feature_extraction/    # Feature extraction modules
│   ├── models/               # Model architecture
│   │   ├── model.py          # Core SER model
│   │   └── lmmd_loss.py      # LMMD loss for domain adaptation
│   └── training/             # Training scripts
├── saved_models/             # Trained model checkpoints
├── results/                  # Training results and metrics
└── test_models/             # Test results and evaluations
```

## 🛠️ Setup

1. Create and activate virtual environment:
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🎯 Usage

### 1. Feature Extraction and Processing

Run the feature extraction script to process audio files and prepare FUSION features:
```bash
python -m src.data_processing.extract_feature
```

This will automatically:
- Extract LogMel spectrograms (128 mel bands)
- Extract HuBERT embeddings (768 dimensions)
- Combine them into FUSION features (896 dimensions)
- Convert features to .npy format for training

### 2. Training

#### Regular Training
Train the model with FUSION features:
```bash
# Train on EMODB
python -m src.training.main --mode train --data EMODB --feature_type FUSION --epoch 50 --batch_size 32

# Train on RAVDESS
python -m src.training.main --mode train --data RAVDESS --feature_type FUSION --epoch 50 --batch_size 32
```

#### Domain Adaptation Training (NEW)
Train with domain adaptation using LMMD loss:
```bash
# EMODB → RAVDESS adaptation
python -m src.training.main --mode train-lmmd --data EMODB --feature_type FUSION --epoch 100 --visualize

# RAVDESS → EMODB adaptation
python -m src.training.main --mode train-lmmd --data RAVDESS --feature_type FUSION --epoch 100 --visualize

# Customize LMMD weight
python -m src.training.main --mode train-lmmd --data EMODB --feature_type FUSION --lmmd_weight 0.3 --visualize
```

### 3. Testing

#### Same-Corpus Testing
Test the trained model on the same dataset:
```bash
# Test on EMODB
python -m src.training.main --mode test --data EMODB --feature_type FUSION --test_path ./test_models/EMODB/FUSION

# Test on RAVDESS
python -m src.training.main --mode test --data RAVDESS --feature_type FUSION --test_path ./test_models/RAVDESS/FUSION
```

#### Cross-Corpus Testing (Improved)
Test a model trained on one dataset against another dataset:
```bash
# Train on EMODB, test on RAVDESS
python -m src.training.main --mode test-cross-corpus --data EMODB --feature_type FUSION --test_path ./test_models/EMODB/FUSION/ --visualize

# Train on RAVDESS, test on EMODB
python -m src.training.main --mode test-cross-corpus --data RAVDESS --feature_type FUSION --test_path ./test_models/RAVDESS/FUSION/ --visualize
```

## 📊 Model Performance

### FUSION Model Results

#### Training Results
- Fold 1 Accuracy: 74.63%
- Fold 2 Accuracy: 79.78%
- Average Training Accuracy: 77.2%

#### Test Results
- Overall Test Accuracy: 85.42%
- Test Loss: 0.5281

#### Per-Class Performance (Test Set)
| Emotion  | Precision | Recall | F1-Score | Support |
|----------|-----------|---------|-----------|----------|
| Angry    | 0.94      | 0.78    | 0.85      | 127      |
| Boredom  | 0.86      | 0.80    | 0.83      | 81       |
| Disgust  | 0.91      | 0.93    | 0.92      | 46       |
| Fear     | 0.92      | 0.86    | 0.89      | 69       |
| Happy    | 0.62      | 0.92    | 0.74      | 71       |
| Neutral  | 0.84      | 0.89    | 0.86      | 79       |
| Sad      | 1.00      | 0.90    | 0.95      | 62       |

## 🔄 Cross-Corpus Evaluation

Cross-corpus evaluation is a crucial aspect of our system that tests the model's ability to generalize across different datasets. This is important because:

1. **Real-world Applicability**: In practice, models will encounter data from various sources with different recording conditions
2. **Robustness Testing**: Helps identify if the model is overfitting to dataset-specific characteristics
3. **Generalization Assessment**: Measures how well the model performs on unseen data from different sources

### Supported Dataset Pairs
- EMODB ↔ RAVDESS

### Emotion Mapping
The system automatically maps emotions between datasets:

**EMODB → RAVDESS mapping:**
- "angry" → ["angry"]
- "boredom" → ["neutral", "calm"]
- "disgust" → ["disgust"]
- "fear" → ["fear"]
- "happy" → ["happy"]
- "neutral" → ["neutral"]
- "sad" → ["sad"]

**RAVDESS → EMODB mapping:**
- "angry" → ["angry"]
- "calm" → ["neutral", "boredom"]
- "disgust" → ["disgust"]
- "fear" → ["fear"]
- "happy" → ["happy"]
- "neutral" → ["neutral"]
- "sad" → ["sad"]
- "surprised" → ["fear"]

### Enhanced Cross-Corpus Features

The system now supports improved cross-corpus capabilities:

1. **Automatic Data Processing**: The system handles different dataset structures automatically
2. **Smart Weight Loading**: Adapts to different model architectures during cross-corpus testing
3. **Visualization**: Generates confusion matrices to analyze performance
4. **Many-to-One/One-to-Many Mappings**: Supports mapping one emotion to multiple target emotions

## 🔬 Domain Adaptation with LMMD (NEW)

This project now includes Local Maximum Mean Discrepancy (LMMD) loss for domain adaptation, which:

- Aligns feature distributions between source and target domains
- Performs class-conditional alignment for better emotion recognition
- Improves cross-corpus performance without requiring target domain labels
- Uses weighted LMMD to balance the contribution of classification and adaptation losses

### How LMMD Works

LMMD measures and minimizes the difference between probability distributions of source and target domains in a reproducing kernel Hilbert space (RKHS), with class-specific alignment.

The loss function:
- Utilizes Gaussian kernels to measure similarity in the feature space
- Computes class-conditional distribution distances
- Updates pseudo-labels during training to improve alignment
- Combines standard classification loss with the LMMD adaptation loss

### Benefits

- Reduces domain shift between different speech emotion datasets
- Preserves emotion-specific characteristics across different languages/datasets
- Improves generalization to new, unseen datasets
- Maintains discriminative power for emotion classification

## 📝 Notes

- The FUSION model combines LogMel spectrograms (128 features) and HuBERT embeddings (768 features) for a total of 896 features per sample
- Model architecture uses dilated convolutional layers for better temporal feature extraction
- K-fold cross-validation (k=2) is used during training
- Cross-corpus evaluation is supported between EMODB and RAVDESS with emotion mapping
- Results are saved in both .h5 (model weights) and .xlsx (evaluation metrics) formats
- LMMD loss enables domain adaptation for better cross-corpus performance

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

[@ali-adi]
[@YvesSamsonLi]
