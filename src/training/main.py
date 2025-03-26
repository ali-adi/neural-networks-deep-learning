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

import numpy as np
import os
import argparse
import tensorflow as tf

from src.models.model import SpeechEmotionModel

# Define label sets here
CASIA_LABELS = ("angry","fear","happy","neutral","sad","surprise")
EMODB_LABELS = ("angry","boredom","disgust","fear","happy","neutral","sad")
SAVEE_LABELS = ("angry","disgust","fear","happy","neutral","sad","surprise")
RAVDE_LABELS = ("angry","calm","disgust","fear","happy","neutral","sad","surprise")
IEMOCAP_LABELS = ("angry","happy","neutral","sad")
EMOVO_LABELS = ("angry","disgust","fear","happy","neutral","sad","surprise")

LABEL_DICT = {
    "CASIA": CASIA_LABELS,
    "EMODB": EMODB_LABELS,
    "SAVEE": SAVEE_LABELS,
    "RAVDE": RAVDE_LABELS,
    "IEMOCAP": IEMOCAP_LABELS,
    "EMOVO": EMOVO_LABELS
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

    # GPU setup
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    print(f"### gpus: {gpus} ###")

    # Auto-adjust for IEMOCAP if needed
    if args.data == "IEMOCAP" and args.dilation_size != 10:
        args.dilation_size = 10

    # Load data (adjust path if necessary)
    data_path = os.path.join("data", "MFCC", f"{args.data}.npy")  
    loaded_data = np.load(data_path, allow_pickle=True).item()
    x_source = loaded_data["x"]
    y_source = loaded_data["y"]

    class_labels = LABEL_DICT[args.data]
    input_shape = x_source.shape[1:]  # (time_steps, feature_dims)

    # Instantiate the model
    model = SpeechEmotionModel(
        input_shape=input_shape,
        class_labels=class_labels,
        args=args
    )

    if args.mode == "train":
        model.train(x_source, y_source)
    elif args.mode == "test":
        x_feats, y_labels = model.test(x_source, y_source, path=args.test_path)
        # x_feats, y_labels are lists containing extracted features/labels per fold

if __name__ == "__main__":
    main()
