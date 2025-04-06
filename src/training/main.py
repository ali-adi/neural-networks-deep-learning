# main.py
"""
# main.py

DESCRIPTION:
------------
This is the main entry point for training or testing the Speech Emotion Recognition (SER) model
based on Discriminant Temporal Pyramid Matching (DTPM) + Attention.

It supports training on various feature types like MFCC, Log-Mel, HuBERT, or a fusion of them,
and saves model checkpoints and evaluation metrics accordingly.

It does the following:
1. Loads preprocessed .npy feature files (or fused features)
2. One-hot encodes emotion labels
3. Initializes and trains the SpeechEmotionModel using k-fold cross-validation
4. Optionally evaluates the model on test data and saves results in Excel

HOW TO RUN THIS SCRIPT:
------------------------
For training mode (example):

    python -m src.training.main --mode train --data EMODB --epoch 50

For testing mode (example):
     python -m src.training.main --mode test --data EMODB --feature_type HUBERT --test_path test_models/EMODB/HUBERT/

This will automatically search for the latest model checkpoint in the test path and evaluate it.

SUPPORTED DATASETS:
-------------------
- EMODB (German emotion dataset)
- RAVDE (RAVDESS speech dataset)

SUPPORTED FEATURE TYPES:
-------------------------
- `MFCC`     → Mel-Frequency Cepstral Coefficients (e.g., 96-dim)
- `LOGMEL`   → Log-mel spectrograms
- `HUBERT`   → Self-supervised HuBERT features
- `FUSION`   → Fusion of multiple features (e.g., MFCC + HuBERT)

KEY PARAMETERS:
---------------
--mode           : Whether to train or test the model
--data           : Dataset name (EMODB or RAVDE)
--feature_type   : Feature type to use (MFCC, LOGMEL, HUBERT, FUSION)
--epoch          : Number of training epochs
--batch_size     : Training batch size
--model_path     : Where to save model weights
--result_path    : Where to save evaluation results
--test_path      : Path to .h5 weights for testing (used only in test mode)
--split_fold     : Number of folds for k-fold cross-validation

NOTES:
------
- Uses categorical_crossentropy → Ensure labels are one-hot encoded
- Automatically adjusts input shape and validates GPU availability
"""

import os
import numpy as np
import argparse
import shutil

# Suppress warnings and TensorFlow logs for clean CLI experience
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from src.models.model import SpeechEmotionModel
from tensorflow.keras.utils import to_categorical
from src.data_processing.load_dataset import load_fused_tensorflow_dataset

# Label sets for supported datasets
EMODB_LABELS = ("angry","boredom","disgust","fear","happy","neutral","sad")
RAVDE_LABELS = ("angry","calm","disgust","fear","happy","neutral","sad","surprise")

LABEL_DICT = {
    "EMODB": EMODB_LABELS,
    "RAVDE": RAVDE_LABELS,
}

def load_data_by_type(args):
    if args.feature_type == 'FUSION':
        x_source, y_source = load_fused_tensorflow_dataset(args.data)
    else:
        if args.feature_type == 'MFCC':
            subfolder = f"{args.data}_MFCC_96"
            data_path = os.path.join("datas", "features", "MFCC", subfolder, f"{args.data}.npy")
        elif args.feature_type == 'LOGMEL':
            subfolder = f"{args.data}_LOGMEL"
            data_path = os.path.join("datas", "features", subfolder, f"{args.data}.npy")
        elif args.feature_type == 'HUBERT':
            subfolder = f"{args.data}_HUBERT"
            data_path = os.path.join("datas", "features", subfolder, f"{args.data}.npy")
        else:
            raise ValueError("Unsupported feature type. Must be MFCC, LOGMEL, HUBERT, or FUSION.")

        print(f"\U0001F4E6 Data source: {data_path}")
        loaded_data = np.load(data_path, allow_pickle=True).item()
        if not isinstance(loaded_data, dict) or "x" not in loaded_data or "y" not in loaded_data:
            raise ValueError(f"Invalid data format in {data_path}")
        x_source = loaded_data["x"]
        y_source = loaded_data["y"]
    return x_source, y_source

def main():
    parser = argparse.ArgumentParser(description="Train or test a temporal conv-based speech emotion model.")
    parser.add_argument('--mode', type=str, default="train", choices=["train", "test"], help="Run mode: train or test.")
    parser.add_argument('--model_path', type=str, default='./saved_models/', help="Root directory to save model checkpoints.")
    parser.add_argument('--result_path', type=str, default='./results/', help="Root directory to save evaluation results.")
    parser.add_argument('--test_path', type=str, default='./test_models/EMODB', help="Path to load test model weights or folder containing .h5 files.")
    parser.add_argument('--data', type=str, default='EMODB', help="Dataset name: EMODB or RAVDE")
    parser.add_argument('--feature_type', type=str, default='MFCC', choices=['MFCC', 'LOGMEL', 'HUBERT', 'FUSION'], help="Type of audio feature used.")
    parser.add_argument('--lr', type=float, default=0.0005, help="Lower learning rate for finer convergence.")
    parser.add_argument('--beta1', type=float, default=0.93, help="Adam beta1.")
    parser.add_argument('--beta2', type=float, default=0.98, help="Adam beta2.")
    parser.add_argument('--batch_size', type=int, default=32, help="Smaller batch size to help generalization.")
    parser.add_argument('--epoch', type=int, default=300, help="Fewer epochs to avoid overfitting.")
    parser.add_argument('--dropout', type=float, default=0.3, help="More dropout to reduce overfitting.")
    parser.add_argument('--random_seed', type=int, default=46, help="Seed for reproducibility.")
    parser.add_argument('--activation', type=str, default='relu', help="Activation function.")
    parser.add_argument('--filter_size', type=int, default=64, help="More filters for richer features")
    parser.add_argument('--dilation_size', type=int, default=8, help="Maximum power-of-two dilation size.")
    parser.add_argument('--kernel_size', type=int, default=2, help="Kernel size for convolution layers.")
    parser.add_argument('--stack_size', type=int, default=2, help="Deeper temporal blocks.")
    parser.add_argument('--split_fold', type=int, default=2, help="More cross-validation folds for stability.")
    parser.add_argument('--gpu', type=str, default='0', help="GPU device index to use.")

    args = parser.parse_args()

    print("\n==============================")
    print("🎛️  SPEECH EMOTION RECOGNITION")
    print("==============================\n")
    print(f"🧪 Mode: {args.mode.upper()}")
    print(f"📁 Dataset: {args.data}")

    x_source, y_source = load_data_by_type(args)


    y_source = to_categorical(y_source, num_classes=len(LABEL_DICT[args.data]))
    print(f"📊 Loaded {x_source.shape[0]} samples for training/testing.")

    args.model_path = os.path.join(args.model_path, args.feature_type.upper())
    args.result_path = os.path.join(args.result_path, args.feature_type.upper())
    if os.path.exists(args.model_path) and not os.path.isdir(args.model_path):
        os.remove(args.model_path)
    os.makedirs(args.model_path, exist_ok=True)
    if os.path.exists(args.result_path) and not os.path.isdir(args.result_path):
        os.remove(args.result_path)
    os.makedirs(args.result_path, exist_ok=True)

    print(f"📂 Model Save Path: {args.model_path}")
    print(f"📈 Result Save Path: {args.result_path}")
    if args.mode == "test":
        print(f"📅 Test model weights path: {args.test_path}")

    print("\n⚙️  Model Configuration:")
    for arg in vars(args):
        print(f"   {arg}: {getattr(args, arg)}")

    devices = tf.config.list_physical_devices()
    selected_device = next((d for d in devices if 'GPU' in d.device_type), devices[0])
    print(f"\n🖥️ Selected Device: {selected_device}\n")

    class_labels = LABEL_DICT[args.data]
    input_shape = x_source.shape[1:]
    print(f"✅ Data loaded successfully! Shape: {x_source.shape}")

    print("\n🧠 Initializing model...\n")
    model = SpeechEmotionModel(input_shape=input_shape, class_labels=class_labels, args=args)

    if args.mode == "train":
        print("🚀 Starting training...\n")
        model.train(x_source, y_source)
        print("\n✅ Training complete!\n")

    elif args.mode == "test":
        print("🧪 Starting testing...\n")
        if os.path.isdir(args.test_path):
            h5_files = [os.path.join(args.test_path, f) for f in os.listdir(args.test_path) if f.endswith(".h5")]
            if not h5_files:
                raise FileNotFoundError(f"No .h5 files found in {args.test_path}")
            args.test_path = max(h5_files, key=os.path.getmtime)
            print(f"📌 Latest model found: {args.test_path}")


        model.evaluate_test(x_source, y_source, path=args.test_path)
        print("\n✅ Testing complete!\n")

if __name__ == "__main__":
    main()
