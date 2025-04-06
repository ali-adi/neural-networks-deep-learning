# extract_feature.py
"""
extract_feature.py

DESCRIPTION:
------------
This script extracts audio features (MFCC, log-mel, or HuBERT) from .wav files that are
organized in emotion-labeled directories.

You can run this script manually or call its functions from another script
(e.g., as part of an automated feature extraction pipeline).

It does the following:
1. Reads each .wav file from an emotion-labeled folder
2. Extracts MFCC, log-mel, or HuBERT embeddings
3. Saves the feature matrix as a .npy file in a mirrored directory structure

HOW TO RUN THIS SCRIPT:
------------------------
Manually (via CLI):

    python -m src.data_processing.extract_feature \
        --input_dir datas/processed/EMODB \
        --output_dir datas/features/MFCC/EMODB_MFCC_96 \
        --feature_type mfcc \
        --n_mfcc 96

FEATURE TYPES SUPPORTED:
-------------------------
- `mfcc`     → Mel-Frequency Cepstral Coefficients (e.g., n_mfcc=96)
- `logmel`   → Log-mel spectrograms (e.g., n_mels=128)
- `hubert`   → Self-supervised HuBERT embeddings

"""

import os
import argparse
import numpy as np
import librosa
import torch
import torchaudio
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, HubertModel


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

# Load HuBERT model
processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
model.eval()

def extract_hubert(audio_path):
    waveform, sr = torchaudio.load(audio_path)
    waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)
    input_values = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt").input_values
    with torch.no_grad():
        outputs = model(input_values)
        return outputs.last_hidden_state.squeeze(0).numpy()

# ======================
# 🔁 AUTO MODE (DUAL) FOR PIPELINE
# ======================
def extract_all_features_dual():
    input_dir = "datas/processed/EMODB"

    print("🚀 Auto mode: Extracting MFCC features...")
    process_directory(
        input_root=input_dir,
        output_root="datas/features/MFCC/EMODB_MFCC_96",
        feature_type="mfcc",
        n_mfcc=96
    )

    print("🚀 Auto mode: Extracting logmel features...")
    process_directory(
        input_root=input_dir,
        output_root="datas/features/EMODB_LOGMEL",
        feature_type="logmel"
    )

    print("🚀 Auto mode: Extracting HuBERT features...")
    process_directory(
        input_root=input_dir,
        output_root="datas/features/EMODB_HUBERT",
        feature_type="hubert"
    )

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
# 🔁 AUTO MODE (for pipeline)
# ======================
def extract_all_features():
    input_dir = "datas/processed/EMODB"
    output_dir = "datas/features/MFCC/EMODB_MFCC_96"
    feature_type = "mfcc"
    n_mfcc = 96

    print(f"🚀 Auto mode: Extracting {feature_type.upper()} from {input_dir}")
    process_directory(input_dir, output_dir, feature_type, n_mfcc)

# ======================
# 📌 CLI Entry Point
# ======================
def main():
    parser = argparse.ArgumentParser(description="Extract audio features from labeled .wav files.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to input .wav file directory (organized by emotion)")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save .npy output features")
    parser.add_argument("--feature_type", type=str, default="mfcc", choices=["mfcc", "logmel", "hubert"], help="Feature type to extract")
    parser.add_argument("--n_mfcc", type=int, default=96, help="Number of MFCC coefficients (only for mfcc)")

    args = parser.parse_args()
    process_directory(args.input_dir, args.output_dir, args.feature_type, args.n_mfcc)

if __name__ == "__main__":
    main()
