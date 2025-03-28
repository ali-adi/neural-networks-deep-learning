# extract_feature.py
"""
extract_feature.py

HOW TO RUN THIS FILE:
---------------------
Run in terminal :
python -m src.data_processing.extract_feature --input_dir datas/processed/EMODB --output_dir datas/features/MFCC/EMODB_MFCC_96 --feature_type mfcc --n_mfcc 96

DESCRIPTION:
Extracts audio features (MFCC or log-mel) from .wav files organized into emotion folders.
Saves features in .npy format in a mirrored directory structure.
"""

import os
import argparse
import numpy as np
import librosa
from tqdm import tqdm

def extract_mfcc(audio_path, n_mfcc=96, sample_rate=16000):
    """
    Extract MFCC (Mel-frequency cepstral coefficients) features from an audio file.
    Args:
        audio_path (str): Path to the .wav audio file.
        n_mfcc (int): Number of MFCC coefficients to extract.
        sample_rate (int): Sampling rate for loading the audio.
    Returns:
        np.ndarray: 2D array of shape (time, n_mfcc)
    """
    waveform, _ = librosa.load(audio_path, sr=sample_rate)
    mfcc_features = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=n_mfcc)
    return mfcc_features.T  # Transpose so time is the first dimension

def extract_logmel(audio_path, sample_rate=16000, n_mels=128):
    """
    Extract log-mel spectrogram features from an audio file.
    Args:
        audio_path (str): Path to the .wav audio file.
        sample_rate (int): Sampling rate for loading the audio.
        n_mels (int): Number of mel filter banks.
    Returns:
        np.ndarray: 2D array of shape (time, n_mels)
    """
    waveform, _ = librosa.load(audio_path, sr=sample_rate)
    mel_spec = librosa.feature.melspectrogram(y=waveform, sr=sample_rate, n_mels=n_mels)
    log_mel = librosa.power_to_db(mel_spec)  # Convert power spectrogram to dB scale
    return log_mel.T  # Transpose so time is the first dimension

def process_directory(input_root, output_root, feature_type="mfcc", n_mfcc=96):
    """
    Processes a root directory of labeled audio folders, extracting features from each .wav file.
    Args:
        input_root (str): Root directory containing folders labeled by emotion.
        output_root (str): Destination directory to save .npy feature files.
        feature_type (str): Type of features to extract ('mfcc' or 'logmel').
        n_mfcc (int): Number of MFCC coefficients (used only for MFCC extraction).
    """
    os.makedirs(output_root, exist_ok=True)
    print(f"\n📂 Extracting {feature_type.upper()} features...")

    # Iterate over each emotion folder in the input directory
    for emotion_label in os.listdir(input_root):
        input_emotion_path = os.path.join(input_root, emotion_label)
        if not os.path.isdir(input_emotion_path):
            continue  # Skip non-folder files

        # Create corresponding output folder for saving features
        output_emotion_path = os.path.join(output_root, emotion_label)
        os.makedirs(output_emotion_path, exist_ok=True)

        # Iterate through each .wav file in the emotion-labeled folder
        for file in tqdm(os.listdir(input_emotion_path), desc=f"{emotion_label}"):
            if not file.endswith(".wav"):
                continue  # Skip non-wav files

            audio_path = os.path.join(input_emotion_path, file)

            # Extract features based on selected feature type
            if feature_type == "mfcc":
                features = extract_mfcc(audio_path, n_mfcc=n_mfcc)
            elif feature_type == "logmel":
                features = extract_logmel(audio_path)
            else:
                raise ValueError(f"Unsupported feature type: {feature_type}")

            # Save extracted features in .npy format
            output_path = os.path.join(output_emotion_path, file.replace(".wav", ".npy"))
            np.save(output_path, features)

    print("✅ Feature extraction complete.")

def main():
    # Set up argument parser for command-line usage
    parser = argparse.ArgumentParser(description="Extract audio features from labeled .wav files.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to input directory with emotion-labeled .wav files")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save output .npy feature files")
    parser.add_argument("--feature_type", type=str, default="mfcc", choices=["mfcc", "logmel"], help="Choose type of feature to extract")
    parser.add_argument("--n_mfcc", type=int, default=96, help="Number of MFCC coefficients (used only if feature_type is 'mfcc')")

    args = parser.parse_args()

    # Start processing based on user input
    process_directory(args.input_dir, args.output_dir, args.feature_type, args.n_mfcc)

if __name__ == "__main__":
    main()
