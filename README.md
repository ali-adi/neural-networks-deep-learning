# Speech Emotion Recognition (SER) System

A deep learning-based system for recognizing emotions in speech using FUSION features (LogMel + HuBERT) for optimal performance.

## 🚀 Features

- FUSION feature extraction combining:
  - LogMel spectrograms (128 mel bands)
  - HuBERT embeddings (768 dimensions)
- Deep learning model with dilated convolutional layers
- K-fold cross-validation
- Cross-corpus evaluation support (EMODB ↔ IEMOCAP)
- Comprehensive evaluation metrics
- Support for multiple datasets (EMODB, IEMOCAP, RAVDE)

## 📁 Project Structure

```
.
├── data/
│   ├── raw/                    # Original audio files
│   └── features/               # Extracted features
│       ├── LOGMEL/            # LogMel spectrograms
│       │   ├── EMODB_LOGMEL_128/
│       │   └── IEMOCAP_LOGMEL_128/
│       ├── HUBERT/            # HuBERT embeddings
│       │   ├── EMODB_HUBERT/
│       │   └── IEMOCAP_HUBERT/
│       └── FUSION/            # Combined features
│           ├── EMODB_FUSION/
│           └── IEMOCAP_FUSION/
├── src/
│   ├── data_processing/       # Data preprocessing scripts
│   ├── feature_extraction/    # Feature extraction modules
│   ├── models/               # Model architecture
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

Train the model with FUSION features:
```bash
# Train on EMODB
python -m src.training.main --mode train --data EMODB --feature_type FUSION --epoch 50 --batch_size 32

# Train on IEMOCAP
python -m src.training.main --mode train --data IEMOCAP --feature_type FUSION --epoch 50 --batch_size 32
```

### 3. Testing

Test the trained model:
```bash
# Same-corpus testing
python -m src.training.main --mode test --data EMODB --feature_type FUSION --test_path ./test_models/EMODB/FUSION

# Cross-corpus testing (train on EMODB, test on IEMOCAP)
python -m src.training.main --mode test --data IEMOCAP --feature_type FUSION --test_path ./test_models/EMODB/FUSION --source_data EMODB
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

Key observations:
- Best performance on "sad" emotions (95% F1-score)
- Strong performance across most emotions
- Lower precision for "happy" emotions despite high recall
- Balanced precision-recall trade-off for most emotions

## 📝 Notes

- The FUSION model combines LogMel spectrograms (128 features) and HuBERT embeddings (768 features) for a total of 896 features per sample
- Model architecture uses dilated convolutional layers for better temporal feature extraction
- K-fold cross-validation (k=2) is used during training
- Cross-corpus evaluation is supported between EMODB and IEMOCAP with emotion mapping
- Results are saved in both .h5 (model weights) and .xlsx (evaluation metrics) formats

## 🔄 Cross-Corpus Evaluation

Cross-corpus evaluation is a crucial aspect of our system that tests the model's ability to generalize across different datasets. This is important because:

1. **Real-world Applicability**: In practice, models will encounter data from various sources with different recording conditions
2. **Robustness Testing**: Helps identify if the model is overfitting to dataset-specific characteristics
3. **Generalization Assessment**: Measures how well the model performs on unseen data from different sources

### Supported Dataset Pairs
- EMODB ↔ IEMOCAP
- EMODB ↔ RAVDE
- IEMOCAP ↔ RAVDE

### Emotion Mapping
The system automatically maps emotions between datasets:
- EMODB "boredom" → IEMOCAP "neutral"
- IEMOCAP "excited" → EMODB "happy"
- IEMOCAP "frustrated" → EMODB "angry"
- IEMOCAP "surprised" → EMODB "neutral"

### Usage
To perform cross-corpus evaluation:
1. Train the model on one dataset (e.g., EMODB)
2. Test it on another dataset (e.g., IEMOCAP) using the `--source_data` parameter
3. The system will automatically map emotions between datasets

The FUSION approach (LogMel + HuBERT) is particularly effective for cross-corpus evaluation because:
- LogMel features capture acoustic characteristics that are consistent across datasets
- HuBERT embeddings provide robust semantic representations that generalize well
- The combination of both feature types helps the model learn more robust patterns

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

[@ali-adi]
[@YvesSamsonLi]
