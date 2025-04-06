# extract_feature.py
"""
extract_feature.py

DESCRIPTION:
------------
This script extracts audio features (MFCC, log-mel, or HuBERT) from .wav files that are
organized in emotion-labeled directories and converts them to single .npy files for training.

You can run this script manually or call its functions from another script
(e.g., as part of an automated feature extraction pipeline).

It does the following:
1. Reads each .wav file from an emotion-labeled folder
2. Extracts MFCC, log-mel, or HuBERT embeddings
3. Saves the feature matrix as a .npy file in a mirrored directory structure
4. Converts the extracted features into single .npy files for training

HOW TO RUN THIS SCRIPT:
------------------------
Manually (via CLI):

    python -m src.data_processing.extract_feature --dataset EMODB
    # or
    python -m src.data_processing.extract_feature --dataset RAVDESS

    # For specific feature type:
    python -m src.data_processing.extract_feature --dataset EMODB --feature_type mfcc --n_mfcc 96

FEATURE TYPES SUPPORTED:
-------------------------
- `mfcc`     → Mel-Frequency Cepstral Coefficients (e.g., n_mfcc=96)
- `logmel`   → Log-mel spectrograms (e.g., n_mels=128)
- `hubert`   → Self-supervised HuBERT embeddings
- `all`      → Extract all feature types (default)

DATASETS SUPPORTED:
------------------
- EMODB (German emotion dataset)
- RAVDESS (RAVDESS speech dataset)
"""

import os
import argparse
import numpy as np
import librosa
import torch
import torchaudio
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, HubertModel
from src.data_processing.load_dataset import convert_to_npy


# ======================
# 🎵 Feature Extractors
# ======================
def extract_mfcc(audio_path, n_mfcc=96, sample_rate=16000):
    waveform, _ = librosa.load(audio_path, sr=sample_rate)
    mfcc_features = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=n_mfcc)
    return mfcc_features.T

def extract_logmel(audio_path, sample_rate=16000, n_mels=128):
    waveform, _ = librosa.load(audio_path, sr=sample_rate)
    mel_spec = librosa.feature.melspectrogram(y=waveform, sr=sample_rate, n_mels=n_mels)
    log_mel = librosa.power_to_db(mel_spec)
    return log_mel.T

# Initialize HuBERT model only when needed
_hubert_model = None
_hubert_processor = None

def get_hubert_model():
    global _hubert_model, _hubert_processor
    if _hubert_model is None:
        # Suppress warnings
        import warnings
        import logging
        
        # Suppress FutureWarning about resume_download
        warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")
        
        # Suppress model initialization warnings
        logging.getLogger("transformers").setLevel(logging.ERROR)
        
        # Load model with proper configuration
        _hubert_processor = Wav2Vec2FeatureExtractor.from_pretrained(
            "facebook/hubert-base-ls960"
        )
        
        # Load model with proper configuration to avoid weight mismatch
        _hubert_model = HubertModel.from_pretrained(
            "facebook/hubert-base-ls960",
            ignore_mismatched_sizes=True  # Ignore size mismatches
        )
        _hubert_model.eval()
    return _hubert_model, _hubert_processor

def extract_hubert(audio_path):
    """Extract HuBERT features from audio file."""
    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000)
    
    # Get model and processor
    model, processor = get_hubert_model()
    
    # Process audio with HuBERT processor
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    
    # Extract features
    with torch.no_grad():
        # Reshape input to match expected dimensions [batch_size, sequence_length]
        input_values = inputs.input_values.squeeze(0)  # Remove batch dimension if present
        
        # Ensure input is 2D (unbatched) or 3D (batched)
        if input_values.dim() == 1:
            input_values = input_values.unsqueeze(0)  # Add batch dimension
        
        # Process through model
        outputs = model(input_values)
        
        # Get last hidden state
        features = outputs.last_hidden_state.numpy()
    
    return features

# ======================
# 🔁 AUTO MODE (DUAL) FOR PIPELINE
# ======================
def extract_all_features_dual(dataset="EMODB"):
    """
    Extract all feature types for a specific dataset.
    
    Args:
        dataset (str): Dataset name ("EMODB" or "RAVDESS")
    """
    input_dir = f"data/processed/{dataset}"

    print(f"🚀 Auto mode: Extracting features for {dataset}...")
    
    print("🚀 Extracting MFCC features...")
    process_directory(
        input_root=input_dir,
        output_root=f"data/features/MFCC/{dataset}_MFCC_96",
        feature_type="mfcc",
        n_mfcc=96
    )

    print("🚀 Extracting logmel features...")
    process_directory(
        input_root=input_dir,
        output_root=f"data/features/LOGMEL/{dataset}_LOGMEL_128",
        feature_type="logmel"
    )

    print("🚀 Extracting HuBERT features...")
    process_directory(
        input_root=input_dir,
        output_root=f"data/features/HUBERT/{dataset}_HUBERT",
        feature_type="hubert"
    )

    # Convert extracted features to single .npy files
    print("\n🔄 Converting extracted features to single .npy files...")
    feature_configs = [
        {
            'input_dir': f'data/features/MFCC/{dataset}_MFCC_96',
            'output_path': f'data/features/MFCC/{dataset}_MFCC_96/{dataset}.npy'
        },
        {
            'input_dir': f'data/features/LOGMEL/{dataset}_LOGMEL_128',
            'output_path': f'data/features/LOGMEL/{dataset}_LOGMEL_128/{dataset}.npy'
        },
        {
            'input_dir': f'data/features/HUBERT/{dataset}_HUBERT',
            'output_path': f'data/features/HUBERT/{dataset}_HUBERT/{dataset}.npy'
        }
    ]
    
    for config in feature_configs:
        if os.path.exists(config['input_dir']):
            print(f"\nProcessing {config['input_dir']}...")
            convert_to_npy(config['input_dir'], config['output_path'])
        else:
            print(f"\nSkipping {config['input_dir']} - directory not found")

# ======================
# 🚀 Directory Processor
# ======================
def process_directory(input_root, output_root, feature_type="mfcc", n_mfcc=96):
    os.makedirs(output_root, exist_ok=True)
    print(f"\n📂 Extracting {feature_type.upper()} features...")

    for emotion_label in os.listdir(input_root):
        input_emotion_path = os.path.join(input_root, emotion_label)
        if not os.path.isdir(input_emotion_path):
            continue

        output_emotion_path = os.path.join(output_root, emotion_label)
        os.makedirs(output_emotion_path, exist_ok=True)

        for file in tqdm(os.listdir(input_emotion_path), desc=f"{emotion_label}"):
            if not file.endswith(".wav"):
                continue

            audio_path = os.path.join(input_emotion_path, file)

            if feature_type == "mfcc":
                features = extract_mfcc(audio_path, n_mfcc=n_mfcc)
            elif feature_type == "logmel":
                features = extract_logmel(audio_path)
            elif feature_type == "hubert":
                features = extract_hubert(audio_path)
            else:
                raise ValueError(f"Unsupported feature type: {feature_type}")

            output_path = os.path.join(output_emotion_path, file.replace(".wav", ".npy"))
            np.save(output_path, features)

    print("✅ Feature extraction complete.")

# ======================
# 📌 CLI Entry Point
# ======================
def main():
    parser = argparse.ArgumentParser(description="Extract audio features from labeled .wav files.")
    parser.add_argument("--dataset", type=str, default="EMODB", choices=["EMODB", "RAVDESS"], help="Dataset to process (EMODB or RAVDESS)")
    parser.add_argument("--feature_type", type=str, default="all", choices=["mfcc", "logmel", "hubert", "all"], help="Feature type to extract")
    parser.add_argument("--n_mfcc", type=int, default=96, help="Number of MFCC coefficients (only for mfcc)")

    args = parser.parse_args()
    
    if args.feature_type == "all":
        extract_all_features_dual(args.dataset)
    else:
        input_dir = f"data/processed/{args.dataset}"
        output_dir = f"data/features/{args.feature_type.upper()}/{args.dataset}_{args.feature_type.upper()}_{args.n_mfcc if args.feature_type == 'mfcc' else '128'}"
        process_directory(input_dir, output_dir, args.feature_type, args.n_mfcc)

if __name__ == "__main__":
    main()
