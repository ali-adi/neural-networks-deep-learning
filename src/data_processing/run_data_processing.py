"""
run_data_processing.py

DESCRIPTION:
------------
This script automates the full data processing pipeline for speech emotion recognition.

It does the following sequentially:
1. Reorganizes raw .wav files into emotion-labeled folders
2. Extracts MFCC, logmel, and HuBERT features
3. Converts the labeled feature directories into .npy datasets for TensorFlow models

USAGE:
------
Run this from the root project directory with:

    python -m src.data_processing.run_data_processing

REQUIREMENTS:
-------------
Ensure the following functions are implemented:
- `reorganize_features()` in `reorganize_data.py`
- `extract_all_features_dual()` in `extract_feature.py`
- `convert_to_npy()` in `load_dataset.py`
"""
# ================================
# 📦 Imports
# ================================
from src.data_processing.reorganize_data import reorganize_features
from src.data_processing.extract_feature import extract_all_features_dual
from src.data_processing.load_dataset import convert_to_npy

# ================================
# 🚀 Full Pipeline Runner
# ================================
def main():
    print("\n==============================")
    print("🔁 FULL DATA PROCESSING PIPELINE")
    print("==============================\n")

    # Step 1: Reorganize .wav files into labeled folders
    print("🗂️  [1/3] Reorganizing raw .wav files...")
    reorganize_features()

    # Step 2: Extract MFCC + LogMel + HuBERT features
    print("\n🎵 [2/3] Extracting MFCC, LogMel, and HuBERT features...")
    extract_all_features_dual()

    # Step 3: Convert all to .npy format
    print("\n📦 [3/3] Converting extracted features to .npy format...")

    convert_to_npy(
        input_dir="datas/features/MFCC/EMODB_MFCC_96",
        output_path="datas/features/MFCC/EMODB_MFCC_96/EMODB.npy"
    )

    convert_to_npy(
        input_dir="datas/features/EMODB_LOGMEL",
        output_path="datas/features/EMODB_LOGMEL/EMODB.npy"
    )

    convert_to_npy(
        input_dir="datas/features/EMODB_HUBERT",
        output_path="datas/features/EMODB_HUBERT/EMODB.npy"
    )

    print("\n✅ All done! MFCC, LogMel, and HuBERT features are ready for training.")

# ================================
# 🔧 CLI Entrypoint
# ================================
if __name__ == "__main__":
    main()
