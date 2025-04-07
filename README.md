# Speech Emotion Recognition with Domain Adaptation

## Table of Contents
1. [Introduction](#introduction)
2. [Research Motivation](#research-motivation)
3. [Technical Overview](#technical-overview)
4. [Model Architecture](#model-architecture)
5. [Design Rationale and Analysis](#design-rationale-and-analysis)
6. [Comparison with Existing Methods](#comparison-with-existing-methods)
7. [Experimental Setup](#experimental-setup)
8. [Performance Analysis](#performance-analysis)
9. [Project Structure](#project-structure)
10. [Setup and Usage](#setup-and-usage)
11. [Future Work](#future-work)
12. [Team and Academic Context](#team-and-academic-context)
13. [License and Acknowledgments](#license-and-acknowledgments)

## Introduction

This project implements a Speech Emotion Recognition (SER) system using deep learning, featuring domain adaptation capabilities for cross-corpus emotion recognition. The system addresses the challenge of speaker-invariant emotion recognition through a novel combination of Discriminant Temporal Pyramid Matching (DTPM) and Local Maximum Mean Discrepancy (LMMD) for domain adaptation.

### Key Features

- Multiple audio feature types support:
  - MFCC (Mel-Frequency Cepstral Coefficients)
  - LogMel Spectrograms
  - HuBERT (Self-supervised Learning Features)
  - Feature Fusion (combining multiple feature types)
- Domain Adaptation using LMMD (Local Maximum Mean Discrepancy)
- Cross-corpus emotion recognition
- Comprehensive hyperparameter tuning
- Detailed performance analysis and visualization
- Support for multiple datasets:
  - EMODB (Berlin Database of Emotional Speech)
  - RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)

## Research Motivation

Speech Emotion Recognition faces several challenges:
1. **Speaker Variability**: Emotions can be expressed differently across speakers of different genders, ages, and cultural backgrounds
2. **Domain Shift**: Performance degradation when testing on different datasets (cross-corpus evaluation)
3. **Feature Representation**: Finding robust features that capture emotional content while being invariant to speaker characteristics

Our approach addresses these challenges through:
- Multi-feature fusion combining traditional acoustic features (MFCC, LogMel) with self-supervised features (HuBERT)
- Domain adaptation using LMMD to reduce dataset-specific biases
- DTPM for better temporal modeling of emotional patterns

## Technical Overview

Our model consists of several key components:
1. Multi-stream Feature Processing
2. Temporal Feature Extraction
3. Domain Adaptation Module
4. Emotion Classification Head

```
Input Features
     ↓
[Feature Processing]
     ↓
[Temporal CNN + DTPM]  →  [LMMD Module]
     ↓                        ↓
[Attention Layer]    [Domain Adaptation Loss]
     ↓
[Classification Head]
     ↓
Emotion Predictions
```

## Model Architecture

### 1. Feature Processing Module

#### Multi-stream Architecture
- **MFCC Stream**:
  - Input shape: (T, 40) [Time steps × MFCC coefficients]
  - 1D Convolution layers for local feature extraction
  - Batch normalization and ReLU activation

- **LogMel Stream**:
  - Input shape: (T, 128) [Time steps × Mel bands]
  - 2D Convolution layers for spectral-temporal patterns
  - Max pooling for dimensionality reduction

- **HuBERT Stream**:
  - Input shape: (T, 768) [Time steps × HuBERT embedding]
  - Linear projection layers
  - Layer normalization

#### Feature Fusion
- Adaptive weighted combination of features
- Learnable weights adjusted during training
- Concatenation followed by dimension reduction

### 2. Temporal Modeling

#### Dilated Temporal CNN
- Multiple temporal blocks with increasing dilation rates
- Each block contains:
  ```
  Input
    ↓
  [Dilated Conv1D]
    ↓
  [BatchNorm + ReLU]
    ↓
  [Residual Connection]
  ```
- Dilation rates: [1, 2, 4, 8, 16]
- Kernel size: 3
- Filters: 128

#### DTPM (Discriminant Temporal Pyramid Matching)
- Hierarchical temporal feature extraction
- Multiple temporal resolutions:
  - Level 1: Global features
  - Level 2: Two temporal segments
  - Level 3: Four temporal segments
- Feature aggregation at each level
- Discriminative feature learning

### 3. Domain Adaptation

#### LMMD Module
- Calculates domain discrepancy at feature level
- Class-conditional alignment
- Gaussian kernel for similarity measurement
- Adaptive weighting based on class distribution

```python
def lmmd_loss(source_features, target_features, source_labels):
    # Local MMD calculation for each class
    loss = 0
    for class_idx in range(num_classes):
        source_class_features = select_class_features(source_features, source_labels, class_idx)
        target_class_features = estimate_target_features(target_features, class_idx)
        loss += calculate_mmd(source_class_features, target_class_features)
    return loss * lmmd_weight
```

### 4. Attention Mechanism

- Multi-head self-attention for temporal dependencies
- Position-wise feed-forward network
- Layer normalization and residual connections
```python
class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        
    def forward(self, Q, K, V):
        # Multi-head attention implementation
        attention_weights = softmax(Q @ K.transpose(-2, -1) / sqrt(self.d_k))
        return attention_weights @ V
```

### 5. Classification Head

- Global average pooling
- Dropout for regularization
- Fully connected layers:
  ```
  FC(1024) → ReLU → Dropout(0.4)
  FC(512) → ReLU → Dropout(0.4)
  FC(num_classes) → Softmax
  ```

### Training Process

1. **Forward Pass**:
   ```python
   # Feature extraction
   mfcc_features = mfcc_stream(mfcc_input)
   logmel_features = logmel_stream(logmel_input)
   hubert_features = hubert_stream(hubert_input)
   
   # Feature fusion
   fused_features = feature_fusion([mfcc_features, logmel_features, hubert_features])
   
   # Temporal modeling
   temporal_features = temporal_cnn(fused_features)
   dtpm_features = dtpm_module(temporal_features)
   
   # Domain adaptation
   if training_mode == 'adaptation':
       lmmd_loss = lmmd_module(source_features, target_features)
   
   # Classification
   emotion_predictions = classification_head(dtpm_features)
   ```

2. **Loss Computation**:
   ```python
   total_loss = classification_loss + lmmd_weight * lmmd_loss
   ```

3. **Optimization**:
   - Adam optimizer with custom beta values
   - Learning rate scheduling
   - Gradient clipping

## Design Rationale and Analysis

### Why These Approaches?

#### 1. Multi-stream Feature Processing

**Rationale:**
- **Complementary Information**: Each feature type captures different aspects of speech:
  - MFCC: Low-level acoustic features, good for speaker characteristics
  - LogMel: Spectral-temporal patterns, effective for emotion dynamics
  - HuBERT: High-level semantic features, robust to speaker variations

**Benefits:**
- **Robustness**: Different features have different sensitivities to noise and speaker variations
- **Rich Representation**: Combines both acoustic and semantic information
- **Adaptive Learning**: Model can learn which features are most relevant for each emotion

**Implementation Challenges:**
- Feature synchronization across streams
- Memory efficiency with multiple feature types
- Optimal fusion strategy selection

#### 2. Temporal Modeling with DTPM

**Why DTPM?**
- **Hierarchical Processing**: Emotions evolve over time with different temporal scales
- **Discriminative Learning**: Focuses on emotion-relevant temporal patterns
- **Memory Efficiency**: Reduces computational complexity through pyramid structure

**Advantages over Simple CNNs:**
- Better capture of long-term dependencies
- More interpretable feature hierarchy
- Reduced parameter count compared to recurrent architectures

**Limitations Addressed:**
- Original DTPM's fixed temporal resolution
- Lack of cross-scale feature interaction
- Insufficient attention to local patterns

#### 3. Domain Adaptation with LMMD

**Why LMMD over other methods?**
- **Class-conditional Alignment**: Preserves emotion-specific features
- **Local Feature Matching**: Better handles fine-grained emotion differences
- **Computational Efficiency**: More scalable than adversarial methods

**Comparison with Other Approaches:**
1. **MMD (Maximum Mean Discrepancy)**:
   - Original MMD: Global distribution alignment
   - Our LMMD: Local and class-specific alignment
   - Result: Better preservation of emotion-discriminative features

2. **Adversarial Domain Adaptation**:
   - GAN-based: Unstable training, mode collapse
   - Our LMMD: Stable optimization, explicit alignment
   - Result: More reliable cross-corpus performance

3. **Feature Alignment**:
   - Traditional: Global feature transformation
   - Our approach: Emotion-specific feature adaptation
   - Result: Better emotion recognition across domains

### Technical Deep Dive

#### 1. Feature Fusion Strategy

**Why This Approach:**
```python
class FeatureFusion:
    def __init__(self, feature_dims):
        self.attention = MultiHeadAttention(
            d_model=sum(feature_dims),
            num_heads=8
        )
        self.weights = nn.Parameter(torch.ones(len(feature_dims)))
        
    def forward(self, features):
        # Normalize weights
        weights = F.softmax(self.weights, dim=0)
        
        # Weighted concatenation
        fused = torch.cat([
            w * f for w, f in zip(weights, features)
        ], dim=-1)
        
        # Self-attention for feature interaction
        return self.attention(fused, fused, fused)
```

**Benefits:**
- Learnable feature weights
- Cross-feature interaction through attention
- Adaptive to different emotions

#### 2. Enhanced DTPM

**Improvements over Original:**
```python
class EnhancedDTPM:
    def __init__(self, num_levels, feature_dim):
        self.levels = nn.ModuleList([
            TemporalBlock(
                in_channels=feature_dim,
                dilation=2**i
            ) for i in range(num_levels)
        ])
        self.cross_scale_attention = CrossScaleAttention(
            feature_dim=feature_dim
        )
        
    def forward(self, x):
        features = []
        for level in self.levels:
            feat = level(x)
            features.append(feat)
        
        # Cross-scale feature interaction
        return self.cross_scale_attention(features)
```

**Key Enhancements:**
- Cross-scale feature interaction
- Dynamic temporal resolution
- Residual connections

#### 3. LMMD Implementation

**Technical Details:**
```python
class LMMD:
    def __init__(self, num_classes, kernel_mul=2.0, kernel_num=5):
        self.num_classes = num_classes
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        
    def gaussian_kernel(self, source, target):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size()[0]), 
            int(total.size()[0]), 
            int(total.size()[1])
        )
        L2_distance = ((total0-total).pow(2)).sum(2)
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)
        bandwidth_list = [bandwidth * (self.kernel_mul**i) for i in range(self.kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)
```

**Advantages:**
- Class-conditional alignment
- Adaptive kernel bandwidth
- Efficient computation

## Theoretical Foundation

### Mathematical Formulation

#### 1. Local Maximum Mean Discrepancy (LMMD)

The LMMD loss is formulated as:

\[ L_{LMMD} = \sum_{c=1}^{C} \left\| \frac{1}{n_c} \sum_{i=1}^{n_c} \phi(x_i^c) - \frac{1}{m_c} \sum_{j=1}^{m_c} \phi(y_j^c) \right\|^2_{\mathcal{H}} \]

where:
- \(x_i^c\) and \(y_j^c\) are features from source and target domains for class c
- \(\phi(\cdot)\) is the feature mapping function
- \(\|\cdot\|_{\mathcal{H}}\) is the norm in the reproducing kernel Hilbert space
- \(n_c\) and \(m_c\) are the number of samples for class c in source and target domains

#### 2. Discriminant Temporal Pyramid Matching (DTPM)

The DTPM feature extraction process:

1. **Temporal Pyramid Construction**:
   \[ P_l = \{B_{l,1}, B_{l,2}, ..., B_{l,2^l}\} \]
   where \(P_l\) is the pyramid at level l, and \(B_{l,i}\) are temporal blocks

2. **Feature Aggregation**:
   \[ F_l = \frac{1}{|P_l|} \sum_{i=1}^{|P_l|} f(B_{l,i}) \]
   where \(f(\cdot)\) is the feature extraction function

3. **Discriminative Learning**:
   \[ L_{DTPM} = \sum_{l=1}^{L} w_l \cdot D(F_l^s, F_l^t) \]
   where \(w_l\) are learnable weights for each pyramid level

#### 3. Multi-head Self-attention

The attention mechanism is defined as:

\[ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V \]

where:
- Q, K, V are query, key, and value matrices
- \(d_k\) is the dimension of the key vectors
- The multi-head version concatenates h attention heads:
  \[ \text{MultiHead}(Q, K, V) = [\text{head}_1; ...; \text{head}_h]W^O \]

## Implementation Details

### Data Processing Pipeline

1. **Audio Preprocessing**:
   ```python
   def preprocess_audio(audio_path, sample_rate=16000):
       # Load and resample audio
       waveform, _ = librosa.load(audio_path, sr=sample_rate)
       
       # Apply pre-emphasis filter
       pre_emphasis = 0.97
       emphasized_signal = np.append(
           waveform[0],
           waveform[1:] - pre_emphasis * waveform[:-1]
       )
       
       return emphasized_signal
   ```

2. **Feature Extraction**:
   ```python
   def extract_features(audio_path, feature_type):
       if feature_type == "mfcc":
           return extract_mfcc(audio_path, n_mfcc=96)
       elif feature_type == "logmel":
           return extract_logmel(audio_path, n_mels=128)
       elif feature_type == "hubert":
           return extract_hubert(audio_path)
   ```

### Model Architecture Implementation

1. **Temporal Convolutional Network**:
   ```python
   class TemporalConvNet:
       def __init__(self, nb_filters=64, kernel_size=2, nb_stacks=1, dilations=8):
           self.nb_filters = nb_filters
           self.kernel_size = kernel_size
           self.nb_stacks = nb_stacks
           self.dilations = dilations
           
       def __call__(self, inputs):
           # Bidirectional processing
           forward = inputs
           backward = tf.reverse(inputs, axis=[1])
           
           # Temporal blocks with increasing dilation
           for dilation_rate in [2**i for i in range(self.dilations)]:
               forward = temporal_block(forward, dilation_rate)
               backward = temporal_block(backward, dilation_rate)
   ```

2. **Domain Adaptation Module**:
   ```python
   class LMMDModule:
       def __init__(self, num_classes, kernel_mul=2.0, kernel_num=5):
           self.num_classes = num_classes
           self.kernel_mul = kernel_mul
           self.kernel_num = kernel_num
           
       def compute_lmmd(self, source_features, target_features):
           # Class-conditional alignment
           loss = 0
           for class_idx in range(self.num_classes):
               source_class = self.select_class_features(source_features, class_idx)
               target_class = self.estimate_target_features(target_features, class_idx)
               loss += self.calculate_mmd(source_class, target_class)
           return loss
   ```

## Experimental Results

### Performance Metrics

#### 1. Overall Performance
- Accuracy: X%
- F1 Score: Y%
- Precision: Z%
- Recall: W%

#### 2. Per-class Performance
| Emotion | Accuracy | F1 Score | Precision | Recall |
|---------|----------|----------|-----------|---------|
| Happy   | X%       | Y%       | Z%        | W%      |
| Sad     | X%       | Y%       | Z%        | W%      |
| Angry   | X%       | Y%       | Z%        | W%      |
| Neutral | X%       | Y%       | Z%        | W%      |
| Fear    | X%       | Y%       | Z%        | W%      |
| Disgust | X%       | Y%       | Z%        | W%      |
| Surprise| X%       | Y%       | Z%        | W%      |

#### 3. Cross-corpus Performance
- EMODB → RAVDESS: X% accuracy
- RAVDESS → EMODB: Y% accuracy
- Domain Adaptation Gain: Z%

## Visualization

### Feature Importance
```
[Placeholder for feature importance plot]
- X-axis: Feature types (MFCC, LogMel, HuBERT)
- Y-axis: Importance score (0-1)
- Bars: Different colors for each feature type
```

### Attention Heatmaps
```
[Placeholder for attention heatmap]
- X-axis: Time steps
- Y-axis: Attention weights
- Color intensity: Attention strength
```

### Domain Adaptation Visualization
```
[Placeholder for t-SNE plot]
- X-axis: First principal component
- Y-axis: Second principal component
- Colors: Different domains
- Shapes: Different emotions
```

### Confusion Matrices
```
[Placeholder for confusion matrix]
- X-axis: Predicted emotions
- Y-axis: True emotions
- Color intensity: Prediction confidence
```

## Ablation Studies

### Component Importance

#### 1. Feature Types
| Feature Combination | Accuracy | F1 Score |
|---------------------|----------|----------|
| MFCC only          | X%       | Y%       |
| LogMel only        | X%       | Y%       |
| HuBERT only        | X%       | Y%       |
| All features       | X%       | Y%       |

#### 2. Architecture Components
| Components         | Accuracy | F1 Score |
|--------------------|----------|----------|
| Base CNN          | X%       | Y%       |
| + DTPM            | X%       | Y%       |
| + Attention       | X%       | Y%       |
| + LMMD            | X%       | Y%       |
| Full Model        | X%       | Y%       |

### Hyperparameter Sensitivity
```
[Placeholder for hyperparameter sensitivity plot]
- X-axis: Hyperparameter values
- Y-axis: Model performance
- Lines: Different hyperparameters
```

## Real-world Applications

### 1. Healthcare
- **Mental Health Monitoring**: Track emotional patterns in patient speech
- **Autism Support**: Help individuals interpret emotional cues
- **Therapy Progress**: Monitor emotional changes during therapy sessions

### 2. Customer Service
- **Call Center Analytics**: Monitor agent-customer interactions
- **Service Quality**: Assess emotional satisfaction in customer interactions
- **Training**: Provide feedback to customer service representatives

### 3. Education
- **Student Engagement**: Monitor student emotional state during learning
- **Adaptive Learning**: Adjust content based on student emotional response
- **Teacher Training**: Help teachers understand student emotional needs

### 4. Accessibility
- **Emotion Recognition**: Assist visually impaired individuals
- **Social Interaction**: Help individuals with social communication challenges
- **Assistive Technology**: Enhance communication devices with emotion detection

## Challenges and Limitations

### 1. Technical Challenges
- **Computational Complexity**: High resource requirements for real-time processing
- **Feature Extraction**: Time-consuming process for HuBERT embeddings
- **Model Size**: Large parameter count due to multi-stream architecture

### 2. Data Challenges
- **Data Scarcity**: Limited emotion-labeled speech data
- **Class Imbalance**: Uneven distribution of emotions in datasets
- **Recording Quality**: Variations in audio quality affect performance

### 3. Performance Limitations
- **Cross-language**: Performance degradation across different languages
- **Noise Robustness**: Sensitivity to background noise
- **Speaker Variability**: Difficulty with unseen speakers

### 4. Ethical Considerations
- **Privacy**: Concerns about emotional data collection
- **Bias**: Potential biases in emotion recognition
- **Consent**: Need for explicit consent in data collection

## Technical Appendices

### A. Algorithm Pseudocode

#### 1. Feature Extraction Pipeline
```python
def extract_features(audio_path, feature_type):
    # Load and preprocess audio
    waveform = preprocess_audio(audio_path)
    
    # Extract features based on type
    if feature_type == "mfcc":
        features = extract_mfcc(waveform, n_mfcc=96)
    elif feature_type == "logmel":
        features = extract_logmel(waveform, n_mels=128)
    elif feature_type == "hubert":
        features = extract_hubert(waveform)
    
    return features
```

#### 2. DTPM Implementation
```python
def dtpm_feature_extraction(features, num_levels=3):
    pyramid_features = []
    
    for level in range(num_levels):
        # Split features into temporal blocks
        num_blocks = 2**level
        block_size = features.shape[0] // num_blocks
        blocks = [features[i:i+block_size] for i in range(0, features.shape[0], block_size)]
        
        # Extract features from each block
        block_features = [extract_block_features(block) for block in blocks]
        
        # Aggregate features at this level
        level_features = aggregate_features(block_features)
        pyramid_features.append(level_features)
    
    return pyramid_features
```

#### 3. LMMD Loss Calculation
```python
def compute_lmmd_loss(source_features, target_features, source_labels):
    total_loss = 0
    
    for class_idx in range(num_classes):
        # Select features for current class
        source_class_features = select_class_features(source_features, source_labels, class_idx)
        target_class_features = estimate_target_features(target_features, class_idx)
        
        # Calculate MMD for this class
        class_loss = calculate_mmd(source_class_features, target_class_features)
        total_loss += class_loss
    
    return total_loss * lmmd_weight
```

### B. Hyperparameter Search Space

#### 1. Model Architecture
```python
architecture_params = {
    'nb_filters': [32, 64, 128],
    'kernel_size': [2, 3, 4],
    'nb_stacks': [1, 2, 3],
    'dilations': [4, 8, 16],
    'dropout_rate': [0.1, 0.2, 0.3, 0.4]
}
```

#### 2. Training Parameters
```python
training_params = {
    'learning_rate': [0.0001, 0.0003, 0.001],
    'batch_size': [16, 32, 64],
    'epochs': [100, 200, 300],
    'lmmd_weight': [0.1, 0.3, 0.5, 0.7]
}
```

### C. Dataset Statistics

#### 1. EMODB Dataset
```python
emodb_stats = {
    'total_utterances': 535,
    'speakers': 10,
    'emotions': 7,
    'duration_range': '1-5 seconds',
    'sample_rate': 16000,
    'format': 'WAV',
    'language': 'German'
}
```

#### 2. RAVDESS Dataset
```python
ravdess_stats = {
    'total_utterances': 1440,
    'speakers': 24,
    'emotions': 8,
    'duration_range': '1-4 seconds',
    'sample_rate': 16000,
    'format': 'WAV',
    'language': 'English'
}
```

### D. Hardware Requirements

#### 1. Training Requirements
```python
training_requirements = {
    'GPU': 'NVIDIA GPU with 8GB+ VRAM',
    'CPU': '4+ cores',
    'RAM': '16GB+',
    'Storage': '50GB+ SSD',
    'Training Time': '4-8 hours per model'
}
```

#### 2. Inference Requirements
```python
inference_requirements = {
    'GPU': 'Optional, CPU sufficient',
    'CPU': '2+ cores',
    'RAM': '8GB+',
    'Storage': '5GB+',
    'Inference Time': '< 100ms per utterance'
}
```

## Comparison with Existing Methods

Our approach builds upon and extends previous work:

1. Previous work A introduced DTPM for SER, which we enhance with:
   - Multi-feature fusion strategy
   - Addition of self-supervised HuBERT features
   - Integration with LMMD for domain adaptation

2. Previous work B used DNNs with extreme learning, which we improve through:
   - More sophisticated temporal modeling
   - Modern self-supervised learning features
   - Cross-corpus adaptation capabilities

### Limitations of Reference Papers

#### Previous Work A

**Architectural Limitations:**
1. **Feature Representation**:
   - Single feature type (MFCC)
   - Limited temporal modeling
   - No consideration of speaker variations

2. **Temporal Modeling**:
   - Fixed pyramid structure
   - No cross-scale feature interaction
   - Limited long-term dependency modeling

3. **Cross-corpus Performance**:
   - No explicit domain adaptation
   - Poor generalization across datasets
   - Speaker-dependent features

**Our Improvements:**
1. **Feature Level**:
   - Multi-feature fusion
   - Self-supervised features
   - Adaptive feature weighting

2. **Temporal Level**:
   - Enhanced DTPM with cross-scale attention
   - Dilated convolutions for better long-term modeling
   - Dynamic temporal resolution

3. **Domain Level**:
   - LMMD for explicit adaptation
   - Class-conditional alignment
   - Speaker-invariant features

#### Previous Work B

**Methodological Limitations:**
1. **Feature Extraction**:
   - Traditional hand-crafted features
   - Limited feature types
   - No temporal modeling

2. **Model Architecture**:
   - Simple DNN + ELM
   - No explicit temporal modeling
   - Limited model capacity

3. **Training Strategy**:
   - Single-corpus training
   - No domain adaptation
   - Poor generalization

**Our Enhancements:**
1. **Feature Processing**:
   - Modern self-supervised features
   - Multi-stream architecture
   - Temporal feature extraction

2. **Model Design**:
   - Deep CNN with DTPM
   - Attention mechanisms
   - Domain adaptation

3. **Training Approach**:
   - Cross-corpus training
   - LMMD-based adaptation
   - Comprehensive evaluation

## Experimental Setup

### Datasets
1. **EMODB (Berlin Database of Emotional Speech)**
   - 535 utterances from 10 actors
   - 7 emotion categories
   - High-quality studio recordings
   - German language

2. **RAVDESS**
   - 1440 utterances from 24 professional actors
   - 8 emotion categories
   - Gender-balanced
   - English language

### Evaluation Metrics
- Weighted accuracy
- Per-class F1 scores
- Cross-corpus performance
- Domain adaptation gain
- Speaker-invariance measures

### Ablation Studies
1. Feature importance:
   - MFCC only
   - LogMel only
   - HuBERT only
   - Feature fusion

2. Architecture components:
   - With/without DTPM
   - With/without LMMD
   - Various temporal modeling approaches

## Performance Analysis

#### 1. Feature Importance

**Ablation Study Results:**
- MFCC: X% accuracy (baseline)
- LogMel: Y% accuracy
- HuBERT: Z% accuracy
- Fusion: W% accuracy
- Fusion + DTPM: V% accuracy
- Full Model: U% accuracy

#### 2. Temporal Modeling Impact

**DTPM vs. Alternatives:**
- Simple CNN: X% accuracy
- LSTM: Y% accuracy
- Original DTPM: Z% accuracy
- Our Enhanced DTPM: W% accuracy

#### 3. Domain Adaptation Effectiveness

**Cross-corpus Performance:**
- No Adaptation: X% accuracy
- MMD: Y% accuracy
- Adversarial: Z% accuracy
- Our LMMD: W% accuracy

## Project Structure

```
.
├── src/
│   ├── data_processing/    # Data preprocessing and feature extraction
│   ├── models/            # Model architecture and training logic
│   ├── training/         # Training scripts and configurations
│   └── utils/           # Utility functions
├── data/
│   ├── raw/            # Raw audio files
│   ├── processed/      # Preprocessed audio files
│   └── features/       # Extracted features
├── results/           # Training results and metrics
├── saved_models/     # Saved model checkpoints
├── test_models/     # Models for testing
├── analyze_results.py        # Script to analyze training results
├── analyze_cross_corpus.py   # Script to analyze cross-corpus performance
└── tune-hyperparameter.sh    # Script for hyperparameter tuning
```

## Setup and Usage

### Setup

1. Create and activate a virtual environment:
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare your data:
   - Place EMODB dataset in `data/raw/EMODB/`
   - Place RAVDESS dataset in `data/raw/RAVDESS/`

4. Process the data:
```bash
# For EMODB
python -m src.data_processing.run_data_processing --dataset EMODB

# For RAVDESS
python -m src.data_processing.run_data_processing --dataset RAVDESS
```

### Usage

#### Regular Training

```bash
# Train on EMODB with MFCC features
python -m src.training.main --mode train --data EMODB --feature_type MFCC --epoch 100 --batch_size 32

# Train on RAVDESS with fusion features
python -m src.training.main --mode train --data RAVDESS --feature_type FUSION --epoch 100 --batch_size 32
```

#### Domain Adaptation Training

```bash
# Train with domain adaptation (EMODB to RAVDESS)
python -m src.training.main --mode train-lmmd --data EMODB --feature_type FUSION --epoch 200 --lmmd_weight 0.5 --visualize

# Train with domain adaptation (RAVDESS to EMODB)
python -m src.training.main --mode train-lmmd --data RAVDESS --feature_type FUSION --epoch 200 --lmmd_weight 0.5 --visualize
```

#### Testing

```bash
# Same-corpus testing
python -m src.training.main --mode test --data EMODB --feature_type FUSION --test_path ./test_models/EMODB/FUSION

# Cross-corpus testing
python -m src.training.main --mode test-cross-corpus --data EMODB --feature_type FUSION --test_path ./test_models/EMODB/FUSION/ --visualize
```

#### Hyperparameter Tuning

To run comprehensive hyperparameter tuning:

```bash
bash tune-hyperparameter.sh
```

After tuning is complete, analyze the results:

```bash
python analyze_results.py
python analyze_cross_corpus.py
```

### Model Parameters

#### Training Parameters
- `epoch`: Number of training epochs (default: 300)
- `batch_size`: Training batch size (default: 32)
- `lr`: Learning rate (default: 0.0003)
- `beta1`: Adam optimizer beta1 (default: 0.93)
- `beta2`: Adam optimizer beta2 (default: 0.98)
- `split_fold`: Number of folds for cross-validation (default: 10)

#### Model Architecture Parameters
- `dropout`: Dropout rate (default: 0.4)
- `activation`: Activation function (default: "relu")
- `filter_size`: Number of filters (default: 128)
- `dilation_size`: Maximum power-of-two dilation size (default: 8)
- `kernel_size`: Kernel size for convolution layers (default: 3)
- `stack_size`: Number of temporal blocks to stack (default: 3)

#### Domain Adaptation Parameters
- `lmmd_weight`: Weight for LMMD loss (default: 0.5)
- `use_lmmd`: Whether to use LMMD loss
- `target_data`: Target domain dataset

## Future Work

Potential areas for improvement and extension:
1. Integration of more self-supervised learning features
2. Dynamic emotion detection within utterances
3. Unsupervised adaptation techniques
4. Multi-modal emotion recognition
5. Real-time processing capabilities

## Team and Academic Context

### Team Members

- [Member 1]
- [Member 2]
- [Member 3]

### Academic Context

This project was developed as part of the SC4001 Neural Networks and Deep Learning course at Nanyang Technological University. It addresses the challenge of developing speaker-invariant emotion recognition systems through deep learning techniques.

## License and Acknowledgments

### License

This project is licensed under the MIT License - see the LICENSE file for details.

### Acknowledgments

- EMODB dataset: [EmoDB](http://emodb.bilderbar.info/start.html)
- RAVDESS dataset: [RAVDESS](https://zenodo.org/record/1188976)
- HuBERT: [facebook/hubert-base-ls960](https://huggingface.co/facebook/hubert-base-ls960)
