# load_dataset.py
"""
load_dataset.py

HOW TO USE THIS MODULE:
------------------------
This module provides a PyTorch-compatible dataset loader for extracted speech emotion features.

Typical usage:

from src.data_processing.load_dataset import load_feature_dataset

train_loader, val_loader = load_feature_dataset(
    data_dir="datas/features/EMODB_MFCC_96",
    batch_size=32,
    val_split=0.2,
    shuffle=True,
    seed=42
)

Run in Terminal:
python -m src.data_processing.load_dataset

DESCRIPTION:
Loads .npy features extracted using extract_feature.py, assigns emotion labels,
and returns PyTorch DataLoader objects for training and validation.
All input tensors are padded to match the longest sample in each batch.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from glob import glob

# Dictionary to map emotion label folder names to numeric indices
LABEL_MAP = {
    "angry": 0,
    "boredom": 1,
    "disgust": 2,
    "fear": 3,
    "happy": 4,
    "sad": 5,
    "neutral": 6
}

class EmotionFeatureDataset(Dataset):
    def __init__(self, data_directory):
        """
        Initializes the dataset by reading all .npy feature files and mapping them to emotion labels.
        Args:
            data_directory (str): Root folder containing one subfolder per emotion label.
        """
        print(f"🔍 Scanning dataset directory: {data_directory}")
        self.samples = []

        # Iterate through each emotion-labeled subfolder
        for label_name in os.listdir(data_directory):
            label_path = os.path.join(data_directory, label_name)
            if not os.path.isdir(label_path):
                continue

            label_index = LABEL_MAP.get(label_name)
            if label_index is None:
                print(f"⚠️ Unknown label '{label_name}' skipped.")
                continue

            feature_files = glob(os.path.join(label_path, "*.npy"))
            for feature_path in feature_files:
                self.samples.append((feature_path, label_index))

            print(f"✅ Found {len(feature_files)} samples under label '{label_name}'")

        print(f"📦 Total samples collected: {len(self.samples)}\n")

    def __len__(self):
        # Return total number of samples
        return len(self.samples)

    def __getitem__(self, index):
        # Load the feature array and corresponding label
        feature_path, label = self.samples[index]
        features = np.load(feature_path)
        return torch.tensor(features, dtype=torch.float32), label

def collate_fn(batch):
    """
    Custom collate function to handle batches of variable-length sequences.
    Args:
        batch (List[Tuple[Tensor, int]]): List of (features, label) pairs
    Returns:
        Tuple[Tensor, Tensor]: Padded feature tensor and label tensor
    """
    features, labels = zip(*batch)
    padded_features = pad_sequence(features, batch_first=True)  # (batch_size, max_time, feature_dim)
    labels = torch.tensor(labels, dtype=torch.long)
    return padded_features, labels

def load_feature_dataset(data_directory, batch_size=32, val_split=0.2, shuffle=True, seed=42):
    """
    Loads emotion dataset and returns training and validation DataLoaders.
    Args:
        data_directory (str): Path to root directory containing labeled .npy feature folders
        batch_size (int): Batch size for training/validation
        val_split (float): Ratio of validation set
        shuffle (bool): Whether to shuffle data before splitting
        seed (int): Random seed for reproducibility
    Returns:
        Tuple[DataLoader, DataLoader]: train_loader and val_loader
    """
    print("📥 Loading dataset...")
    dataset = EmotionFeatureDataset(data_directory)

    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    print(f"🔀 Splitting dataset: {train_size} training / {val_size} validation")

    if shuffle:
        generator = torch.Generator().manual_seed(seed)
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    else:
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    print(f"✅ DataLoaders ready. Batch size: {batch_size}\n")
    return train_loader, val_loader

if __name__ == "__main__":
    print("\n============================")
    print("🧪 Testing load_feature_dataset")
    print("============================\n")

    # Run a test using the EMODB MFCC features
    train_loader, val_loader = load_feature_dataset(
        data_directory="datas/features/MFCC/EMODB_MFCC_96",
        batch_size=4,
        val_split=0.2
    )

    # Preview one batch of training data
    print("📊 Previewing first batch from training loader:")
    for features, labels in train_loader:
        print("📐 Feature Tensor Shape:", features.shape)
        print("🏷️ Labels:", labels)
        break

    print("\n✅ Dataset loading and batching test complete.\n")
