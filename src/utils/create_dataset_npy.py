# create_dataset_npy.py

"""
create_dataset_npy.py

HOW TO RUN THIS FILE:
---------------------
Run in terminal:
python -m src.data_processing.create_dataset_npy --feature_dir datas/features/MFCC/EMODB_MFCC_96 --output_path datas/features/MFCC/EMODB.py

DESCRIPTION:
Converts a directory of extracted feature files (organized by emotion label folders)
into a single .npy dataset containing padded tensors and one-hot encoded labels.
Useful for training/testing deep learning models efficiently.
"""

import os
import numpy as np
from glob import glob
from tqdm import tqdm
import argparse

# Emotion label mapping (folder name to index)
LABEL_MAP = {
    "angry": 0,
    "boredom": 1,
    "disgust": 2,
    "fear": 3,
    "happy": 4,
    "neutral": 5,
    "sad": 6
}

def main(feature_folder, output_path):
    features = []
    labels = []

    print(f"\n📁 Reading features from: {feature_folder}")

    # Loop over each emotion label folder
    for label_name, label_index in LABEL_MAP.items():
        label_dir = os.path.join(feature_folder, label_name)
        if not os.path.exists(label_dir):
            continue

        print(f"🔍 Processing label: {label_name}")

        # Read all .npy feature files for this label
        for npy_file in tqdm(glob(os.path.join(label_dir, "*.npy"))):
            feat = np.load(npy_file)
            features.append(feat)
            labels.append(label_index)

    # Padding features to same length
    max_len = max([f.shape[0] for f in features])  # maximum time steps
    n_feat_dim = features[0].shape[1]              # feature dimension (e.g., 96)
    padded_features = np.zeros((len(features), max_len, n_feat_dim))

    for i, f in enumerate(features):
        padded_features[i, :f.shape[0], :] = f

    # One-hot encode labels
    labels = np.array(labels)
    one_hot_labels = np.eye(len(LABEL_MAP))[labels]

    print(f"\n📊 Final feature shape: {padded_features.shape}")
    print(f"🏷️ Final label shape: {one_hot_labels.shape}")

    # Save as dictionary
    np.save(output_path, {
        "x": padded_features,
        "y": one_hot_labels
    })

    print(f"\n✅ Dataset saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert extracted features into .npy dataset for model training.")
    parser.add_argument("--feature_dir", type=str, required=True, help="Path to the extracted feature folder")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output .npy file")
    args = parser.parse_args()

    main(args.feature_dir, args.output_path)
