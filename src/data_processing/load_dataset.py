# load_dataset.py
"""
load_dataset.py

DESCRIPTION:
------------
This module provides utilities for loading emotion-labeled datasets extracted from .wav files
into PyTorch-compatible DataLoaders, as well as NumPy arrays for TensorFlow models.

It supports:
✔️ Loading MFCC and logmel features
✔️ Fusing HuBERT + logmel features (for TensorFlow)
✔️ Padding variable-length sequences
✔️ Splitting data into training and validation sets
✔️ Automatic testing of loaders for inspection

HOW TO USE THIS MODULE:
------------------------
1. Load PyTorch MFCC data:
    from src.data_processing.load_dataset import load_feature_dataset
    train_loader, val_loader = load_feature_dataset("datas/features/MFCC/EMODB_MFCC_96")

2. Load TensorFlow-compatible fused HuBERT + logmel:
    from src.data_processing.load_dataset import load_fused_tensorflow_dataset
    x_fused, y_fused = load_fused_tensorflow_dataset("EMODB")

3. Test all loaders from CLI:
    python -m src.data_processing.load_dataset

FUNCTION INDEX:
---------------
- EmotionFeatureDataset: PyTorch-compatible dataset wrapper
- collate_fn: Custom batch collator to pad sequences
- load_feature_dataset(): Loads .npy files into PyTorch DataLoader
- load_fused_tensorflow_dataset(): Fuses HuBERT + logmel features for TensorFlow
- load_all_feature_datasets(): Utility to test all loaders at once (MFCC + Fusion)

"""

# ====================
# 📦 Core Imports
# ====================
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from torch.nn.utils.rnn import pad_sequence
from glob import glob

# ====================
# 🔖 Label Mapping
# ====================
LABEL_MAP = {
    "angry": 0,
    "boredom": 1,
    "disgust": 2,
    "fear": 3,
    "happy": 4,
    "sad": 5,
    "neutral": 6
}

# ==============================
# 📂 PyTorch Dataset Definition
# ==============================
class EmotionFeatureDataset(Dataset):
    def __init__(self, data_directory):
        print(f"🔍 Scanning dataset directory: {data_directory}")
        self.samples = []

        # Traverse each emotion-labeled subfolder
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
        return len(self.samples)

    def __getitem__(self, index):
        feature_path, label = self.samples[index]
        features = np.load(feature_path)
        return torch.tensor(features, dtype=torch.float32), label

# ==============================
# 🧩 Sequence Padding for Batches
# ==============================
def collate_fn(batch):
    features, labels = zip(*batch)
    padded_features = pad_sequence(features, batch_first=True)
    labels = torch.tensor(labels, dtype=torch.long)
    return padded_features, labels

# ===================================
# 🧪 Load PyTorch Dataset for Training
# ===================================
def load_feature_dataset(data_directory, batch_size=32, val_split=0.2, shuffle=True, seed=42):
    """
    Loads a PyTorch DataLoader from .npy feature files.

    Args:
        data_directory (str): Path to dataset directory
        batch_size (int): Batch size for training
        val_split (float): Validation split ratio
        shuffle (bool): Shuffle before splitting
        seed (int): Random seed for reproducibility

    Returns:
        Tuple[DataLoader, DataLoader]: (train_loader, val_loader)
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

# ======================================
# 🔗 Load and Fuse HuBERT + Logmel Arrays
# ======================================
def load_fused_tensorflow_dataset(dataset_name):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    """
    Fuses HuBERT and logmel .npy features for TensorFlow training.

    Args:
        dataset_name (str): e.g., "EMODB"

    Returns:
        Tuple[np.ndarray, np.ndarray]: x_fused, y_labels
    """
    logmel_path = f"datas/features/{dataset_name}_LOGMEL/{dataset_name}.npy"
    hubert_path = f"datas/features/{dataset_name}_HUBERT/{dataset_name}.npy"

    logmel = np.load(logmel_path, allow_pickle=True).item()
    hubert = np.load(hubert_path, allow_pickle=True).item()

    x_logmel, y_logmel = logmel['x'], logmel['y']
    x_hubert, y_hubert = hubert['x'], hubert['y']

    assert len(x_logmel) == len(x_hubert),
    assert np.array_equal(y_logmel, y_hubert),

    # ⚠️ Pad each feature set separately to the same time dimension
    x_logmel = pad_sequences(x_logmel, padding="post", dtype="float32")
    x_hubert = pad_sequences(x_hubert, padding="post", dtype="float32")

    # ⚠️ Ensure same time steps
    min_timesteps = min(x_logmel.shape[1], x_hubert.shape[1])
    x_logmel = x_logmel[:, :min_timesteps, :]
    x_hubert = x_hubert[:, :min_timesteps, :]

    # ✅ Concatenate along feature dimension
    x_fused = np.concatenate([x_logmel, x_hubert], axis=-1)

    return x_fused, y_logmel

# ===============================
# 🚀 Auto Test All Feature Loaders
# ===============================
def load_all_feature_datasets(dataset_name="EMODB", batch_size=64):
    """
    Tests MFCC DataLoader + HuBERT+logmel fusion dataset for TensorFlow.
    """
    mfcc_path = f"datas/features/MFCC/{dataset_name}_MFCC_96"
    print("🧪 Loading MFCC DataLoader...")
    mfcc_train, mfcc_val = load_feature_dataset(mfcc_path, batch_size=batch_size)

    print("🧪 Loading fused logmel + HuBERT...")
    x_fused, y_fused = load_fused_tensorflow_dataset(dataset_name)
    print("🧪 Fused TensorFlow dataset shape:", x_fused.shape)

    return mfcc_train, mfcc_val, x_fused, y_fused

# 🔄 CONVERT FOLDER TO .NPY FORMAT FOR TENSORFLOW (SINGLE FILE)
def convert_to_npy(input_dir="datas/features/MFCC/EMODB_MFCC_96", output_path="datas/features/MFCC/EMODB_MFCC_96/EMODB.npy"):
    print(f"📤 Converting from folder '{input_dir}' to single .npy file → {output_path}")
    data = []
    labels = []
    for label in os.listdir(input_dir):
        label_dir = os.path.join(input_dir, label)
        if not os.path.isdir(label_dir):
            continue
        label_index = LABEL_MAP.get(label)
        for file in os.listdir(label_dir):
            if file.endswith(".npy"):
                feat = np.load(os.path.join(label_dir, file))
                data.append(feat)
                labels.append(label_index)
    data = np.array(data, dtype=object)
    labels = np.array(labels)
    np.save(output_path, {"x": data, "y": labels})
    print("✅ Dataset conversion to .npy complete! Saved to:", output_path)


# ===============================
# 🧪 CLI TEST MODE
# ===============================
if __name__ == "__main__":
    print("\n============================")
    print("🧪 Testing load_feature_dataset")
    print("============================\n")

    train_loader, val_loader = load_feature_dataset(
        data_directory="datas/features/MFCC/EMODB_MFCC_96",
        batch_size=4,
        val_split=0.2
    )

    print("📊 Previewing one batch:")
    for features, labels in train_loader:
        print("📐 Feature Tensor Shape:", features.shape)
        print("🏷️ Labels:", labels)
        break
