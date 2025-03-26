# main.py

"""
main.py

HOW TO RUN THIS FILE:
---------------------
Example (train):
    python -m src.training.main --mode train --data EMODB --epoch 2
Example (test):
    python -m src.training.main --mode test --data EMODB --test_path ./test_models/EMODB_46

DESCRIPTION:
Entry point for training or testing the SpeechEmotionModel using a temporal
convolution-based architecture for emotion recognition.
"""

import os
import numpy as np
import argparse

# 1) Suppress TensorFlow and Python warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Hides all TF logs except errors
import warnings
warnings.filterwarnings("ignore")         # Hides Python-level warnings

import tensorflow as tf
from src.models.model import SpeechEmotionModel

# Define label sets here
EMODB_LABELS = ("angry","boredom","disgust","fear","happy","neutral","sad")
RAVDE_LABELS = ("angry","calm","disgust","fear","happy","neutral","sad","surprise")

LABEL_DICT = {
    "EMODB": EMODB_LABELS,
    "RAVDE": RAVDE_LABELS,
}

def main():
    parser = argparse.ArgumentParser(description="Train or test a temporal conv-based speech emotion model.")
    parser.add_argument('--mode', type=str, default="train", choices=["train", "test"],
                        help="Run mode: train or test.")
    parser.add_argument('--model_path', type=str, default='./saved_models/',
                        help="Directory to save model checkpoints.")
    parser.add_argument('--result_path', type=str, default='./results/',
                        help="Directory to save results (metrics, confusion matrices).")
    parser.add_argument('--test_path', type=str, default='./test_models/EMODB_46',
                        help="Directory for loading test model weights.")
    parser.add_argument('--data', type=str, default='EMODB',
                        help="Which dataset to use (e.g., EMODB).")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate.")
    parser.add_argument('--beta1', type=float, default=0.93, help="Adam beta1.")
    parser.add_argument('--beta2', type=float, default=0.98, help="Adam beta2.")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size.")
    parser.add_argument('--epoch', type=int, default=500, help="Number of epochs.")
    parser.add_argument('--dropout', type=float, default=0.1, help="Dropout rate.")
    parser.add_argument('--random_seed', type=int, default=46, help="Random seed for reproducibility.")
    parser.add_argument('--activation', type=str, default='relu', help="Activation function.")
    parser.add_argument('--filter_size', type=int, default=39, help="Number of filters in the model.")
    parser.add_argument('--dilation_size', type=int, default=8,
                        help="Max power-of-2 dilation. (For IEMOCAP, might need 10.)")
    parser.add_argument('--kernel_size', type=int, default=2, help="Convolution kernel size.")
    parser.add_argument('--stack_size', type=int, default=1, help="Number of stacked blocks.")
    parser.add_argument('--split_fold', type=int, default=10, help="Number of folds for cross-validation.")
    parser.add_argument('--gpu', type=str, default='0', help="Which GPU to use.")

    args = parser.parse_args()

    print("\n==============================")
    print("🎛️  SPEECH EMOTION RECOGNITION")
    print("==============================\n")
    print(f"🧪 Mode: {args.mode.upper()}")
    print(f"📁 Dataset: {args.data}")
    print(f"🎯 Output model path: {args.model_path}")
    print(f"📊 Results will be saved to: {args.result_path}")
    if args.mode == "test":
        print(f"📥 Test model weights path: {args.test_path}")
    print("\n⚙️ Model Settings:")
    print(f"   Epochs: {args.epoch}")
    print(f"   Batch Size: {args.batch_size}")
    print(f"   Learning Rate: {args.lr}")
    print(f"   Dropout: {args.dropout}")
    print(f"   Filter Size: {args.filter_size}, Kernel Size: {args.kernel_size}")
    print(f"   Dilation Size: {args.dilation_size}, Stacks: {args.stack_size}")
    print(f"   Activation: {args.activation}")
    print(f"   Cross-Validation Folds: {args.split_fold}")
    print("\n📦 Loading data...\n")

    # GPU setup
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    print(f"🖥️ Available GPUs: {gpus}\n")

    if args.data == "IEMOCAP" and args.dilation_size != 10:
        args.dilation_size = 10
        print("⚠️ Dilation size adjusted to 10 for IEMOCAP dataset.\n")

    # Load data (adjust path if necessary)
    data_path = os.path.join("data", "MFCC", f"{args.data}.npy")  
    loaded_data = np.load(data_path, allow_pickle=True).item()
    x_source = loaded_data["x"]
    y_source = loaded_data["y"]

    class_labels = LABEL_DICT[args.data]
    input_shape = x_source.shape[1:]

    print(f"✅ Data loaded successfully! Shape: {x_source.shape}")
    print("🧠 Initializing model...\n")

    model = SpeechEmotionModel(
        input_shape=input_shape,
        class_labels=class_labels,
        args=args
    )

    if args.mode == "train":
        print("🚀 Starting training...\n")
        model.train(x_source, y_source)
        print("\n✅ Training complete!\n")
    elif args.mode == "test":
        print("🧪 Starting testing...\n")
        x_feats, y_labels = model.test(x_source, y_source, path=args.test_path)
        print("\n✅ Testing complete!\n")

if __name__ == "__main__":
    main()
