# main.py
"""
main.py

HOW TO RUN THIS FILE:
---------------------
Example (train):
    python -m src.training.main --mode train --data EMODB --epoch 50
Example (test):
    python -m src.training.main --mode test --data EMODB --test_path test_models/EMODB/  will search for the latest model

DESCRIPTION:
Entry point for training or testing the SpeechEmotionModel using a temporal
convolution-based architecture for emotion recognition.
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

# Label sets for supported datasets
EMODB_LABELS = ("angry","boredom","disgust","fear","happy","neutral","sad")
RAVDE_LABELS = ("angry","calm","disgust","fear","happy","neutral","sad","surprise")

LABEL_DICT = {
    "EMODB": EMODB_LABELS,
    "RAVDE": RAVDE_LABELS,
}

def main():
    parser = argparse.ArgumentParser(description="Train or test a temporal conv-based speech emotion model.")
    parser.add_argument('--mode', type=str, default="train", choices=["train", "test"], help="Run mode: train or test.")
    parser.add_argument('--model_path', type=str, default='./saved_models/', help="Root directory to save model checkpoints.")
    parser.add_argument('--result_path', type=str, default='./results/', help="Root directory to save evaluation results.")
    parser.add_argument('--test_path', type=str, default='./test_models/EMODB', help="Path to load test model weights or folder containing .h5 files.")
    parser.add_argument('--data', type=str, default='EMODB', help="Dataset name: EMODB or RAVDE")
    parser.add_argument('--feature_type', type=str, default='MFCC', choices=['MFCC', 'LOGMEL', 'HUBERT'], help="Type of audio feature used.")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate.")
    parser.add_argument('--beta1', type=float, default=0.93, help="Adam beta1.")
    parser.add_argument('--beta2', type=float, default=0.98, help="Adam beta2.")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size.")
    parser.add_argument('--epoch', type=int, default=500, help="Training epochs.")
    parser.add_argument('--dropout', type=float, default=0.1, help="Dropout rate.")
    parser.add_argument('--random_seed', type=int, default=46, help="Seed for reproducibility.")
    parser.add_argument('--activation', type=str, default='relu', help="Activation function.")
    parser.add_argument('--filter_size', type=int, default=39, help="Number of filters.")
    parser.add_argument('--dilation_size', type=int, default=8, help="Maximum power-of-two dilation size.")
    parser.add_argument('--kernel_size', type=int, default=2, help="Kernel size for convolution layers.")
    parser.add_argument('--stack_size', type=int, default=1, help="Number of stacked temporal blocks.")
    parser.add_argument('--split_fold', type=int, default=2, help="Cross-validation fold count.")
    parser.add_argument('--gpu', type=str, default='0', help="GPU device index to use.")

    args = parser.parse_args()

    print("\n==============================")
    print("🎛️  SPEECH EMOTION RECOGNITION")
    print("==============================\n")
    print(f"🧪 Mode: {args.mode.upper()}")
    print(f"📁 Dataset: {args.data}")

    # Determine the dataset path based on feature type and dataset
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
        raise ValueError("Unsupported feature type. Must be MFCC, LOGMEL, or HUBERT.")

    print(f"📦 Data source: {data_path}")
    print(f"📂 Model Save Path: {os.path.join(args.model_path, args.data)}")
    print(f"📈 Result Save Path: {os.path.join(args.result_path, args.data)}")
    if args.mode == "test":
        print(f"📥 Test model weights path: {args.test_path}")

    print("\n⚙️  Model Configuration:")
    for arg in vars(args):
        print(f"   {arg}: {getattr(args, arg)}")

    # Flexible device selection: prioritize CUDA → GPU → MPS → CPU
    cuda_gpus = tf.config.list_physical_devices('GPU') if tf.test.is_built_with_cuda() else []
    gpu_devices = tf.config.list_physical_devices('GPU')
    mps_devices = tf.config.list_physical_devices('MPS')
    cpu_devices = tf.config.list_physical_devices('CPU')

    if cuda_gpus:
        device = cuda_gpus[0]
    elif gpu_devices:
        device = gpu_devices[0]
    elif mps_devices:
        device = mps_devices[0]
    else:
        device = cpu_devices[0]

    print(f"\n🖥️ Selected Device: {device}\n")

    if args.data == "IEMOCAP" and args.dilation_size != 10:
        args.dilation_size = 10
        print("⚠️ Dilation size adjusted to 10 for IEMOCAP dataset.\n")

    print(f"📤 Loading dataset from: {data_path}\n")
    loaded_data = np.load(data_path, allow_pickle=True).item()
    x_source = loaded_data["x"]
    y_source = loaded_data["y"]

    class_labels = LABEL_DICT[args.data]
    input_shape = x_source.shape[1:]
    print(f"✅ Data loaded successfully! Shape: {x_source.shape}")

    print("\n🧠 Initializing model...\n")
    model = SpeechEmotionModel(input_shape=input_shape, class_labels=class_labels, args=args)

    args.model_path = os.path.join(args.model_path, args.data)
    args.result_path = os.path.join(args.result_path, args.data)
    os.makedirs(args.model_path, exist_ok=True)
    os.makedirs(args.result_path, exist_ok=True)

    if args.mode == "train":
        print("🚀 Starting training...\n")
        model.train(x_source, y_source)
        print("\n✅ Training complete!\n")

    elif args.mode == "test":
        print("🧪 Starting testing...\n")

        # Dynamically select latest .h5 file if folder path is provided
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
