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

    python -m src.data_processing.run_data_processing --dataset EMODB
    # or
    python -m src.data_processing.run_data_processing --dataset RAVDESS

REQUIREMENTS:
-------------
Ensure the following functions are implemented:
- `reorganize_emodb()` and `reorganize_ravdess()` in `reorganize_data.py`
- `extract_all_features_dual()` in `extract_feature.py`
- `convert_to_npy()` in `load_dataset.py`
"""

# ================================
# 📦 Imports
# ================================
import argparse
import os
from src.data_processing.reorganize_data import reorganize_emodb, reorganize_ravdess
from src.data_processing.extract_feature import extract_all_features_dual
from src.data_processing.load_dataset import convert_to_npy


# ================================
# 🚀 Full Pipeline Runner
# ================================
def main(dataset="EMODB"):
    print("\n==============================")
    print(f"🔁 FULL DATA PROCESSING PIPELINE FOR {dataset}")
    print("==============================\n")

    # Step 1: Reorganize .wav files into labeled folders
    print("🗂️  [1/3] Reorganizing raw .wav files...")
    source_dir = os.path.join("data", "raw", dataset)
    dest_dir = os.path.join("data", "processed", dataset)

    # Create destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)

    # Call appropriate reorganizer function
    if dataset == "EMODB":
        reorganize_emodb(source_dir, dest_dir)
    else:  # RAVDESS
        reorganize_ravdess(source_dir, dest_dir)

    # Step 2: Extract MFCC + LogMel + HuBERT features
    print("\n🎵 [2/3] Extracting MFCC, LogMel, and HuBERT features...")
    extract_all_features_dual(dataset)

    # Step 3: Convert all to .npy format
    print("\n📦 [3/3] Converting extracted features to .npy format...")

    convert_to_npy(
        input_dir=f"data/features/MFCC/{dataset}_MFCC_96",
        output_path=f"data/features/MFCC/{dataset}_MFCC_96/{dataset}.npy",
    )

    convert_to_npy(
        input_dir=f"data/features/LOGMEL/{dataset}_LOGMEL_128",
        output_path=f"data/features/LOGMEL/{dataset}_LOGMEL_128/{dataset}.npy",
    )

    convert_to_npy(
        input_dir=f"data/features/HUBERT/{dataset}_HUBERT",
        output_path=f"data/features/HUBERT/{dataset}_HUBERT/{dataset}.npy",
    )

    print(
        f"\n✅ All done! MFCC, LogMel, and HuBERT features for {dataset} are ready for training."
    )


# ================================
# 🔧 CLI Entrypoint
# ================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the full data processing pipeline for speech emotion recognition."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="EMODB",
        choices=["EMODB", "RAVDESS"],
        help="Dataset to process (EMODB or RAVDESS)",
    )

    args = parser.parse_args()
    main(args.dataset)
