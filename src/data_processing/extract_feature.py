"""
extract_feature.py

HOW TO RUN THIS FILE:
---------------------
Example:
    python extract_feature.py --data_name EMODB --mean_signal_length 96000 --output_dir ./EMODB_MFCC_96

DESCRIPTION:
Extracts MFCC features from audio files (e.g., EMODB), saves them as CSV,
and compiles them into .npy files for model training.
"""

import numpy as np
import os
import sys
from typing import Tuple
from tqdm import tqdm
import librosa
from tensorflow.keras.utils import to_categorical
import argparse
from natsort import ns, natsorted

def get_feature(file_path: str,
                feature_type:str="MFCC",
                mean_signal_length:int=96000,
                embed_len:int=39) -> np.ndarray:
    """
    Loads audio from file_path, pads/crops to fixed length,
    and extracts the chosen feature (MFCC by default).

    Args:
        file_path (str): Path to the audio file.
        feature_type (str): Currently supports "MFCC".
        mean_signal_length (int): Desired fixed length for the audio signal.
        embed_len (int): Number of MFCC coefficients.

    Returns:
        np.ndarray: Feature matrix of shape (time_steps, embed_len).
    """
    signal, fs = librosa.load(file_path, sr=None)
    s_len = len(signal)

    # Pad or crop signal
    if s_len < mean_signal_length:
        pad_len = mean_signal_length - s_len
        pad_rem = pad_len % 2
        pad_len //= 2
        signal = np.pad(signal, (pad_len, pad_len + pad_rem), 'constant', constant_values=0)
    else:
        pad_len = s_len - mean_signal_length
        pad_len //= 2
        signal = signal[pad_len:pad_len + mean_signal_length]

    if feature_type == "MFCC":
        mfcc_feat = librosa.feature.mfcc(y=signal, sr=fs, n_mfcc=embed_len)
        return np.transpose(mfcc_feat)
    else:
        raise ValueError(f"Unsupported feature_type: {feature_type}")

def generate_csv(csv_save: str,
                 data_name: str="EMODB",
                 feature_type: str="MFCC",
                 embed_len: int=39,
                 mean_signal_length: int=96000,
                 class_labels: Tuple=("angry","boredom","disgust","fear","happy","neutral","sad")):
    """
    Iterates over emotion folders, extracts features from audio .wav files,
    and saves them as CSV files in labeled subdirectories under csv_save.
    """
    current_dir = os.getcwd()
    if not os.path.exists(csv_save):
        print(f"{csv_save} created.")
        os.makedirs(csv_save)

    for label_dir in class_labels:
        label_path = os.path.join(csv_save, label_dir)
        if not os.path.exists(label_path):
            os.makedirs(label_path)
            print(f"{label_path} created.")

    datapath, labels = [], []
    dataset_dir = os.path.join("data", "processed", data_name)
    sys.stderr.write(f'Current Folder: {current_dir}\n')
    sys.stderr.write(f'Looking in folder: {dataset_dir}\n')

    # Gather all .wav files and labels
    for i, directory in enumerate(class_labels):
        emotion_dir = os.path.join(dataset_dir, directory)
        sys.stderr.write(f"Start to Read {directory}\n")
        filelist = os.listdir(emotion_dir)
        for f in tqdm(filelist):
            if f.endswith('.wav'):
                filepath = os.path.join(emotion_dir, f)
                datapath.append(filepath)
                labels.append(i)
        sys.stderr.write(f"End to Read {directory}\n")

    # Extract features and save CSV
    for (video_path, label) in tqdm(zip(datapath, labels), total=len(datapath)):
        filename = os.path.splitext(os.path.basename(video_path))[0]
        feature_vector = get_feature(file_path=video_path,
                                     feature_type=feature_type,
                                     mean_signal_length=mean_signal_length,
                                     embed_len=embed_len)
        csv_filename = f"{filename}_raw.csv"
        csv_filepath = os.path.join(csv_save, class_labels[label], csv_filename)
        np.savetxt(csv_filepath, feature_vector, delimiter=',')

def process_csv(data_path: str,
                mfcc_len: int=39,
                class_labels: Tuple=("angry","boredom","disgust","fear","happy","neutral","sad"),
                flatten: bool=False):
    """
    Reads CSV files (previously generated) in data_path into NumPy arrays.
    """
    x, y = [], []
    current_dir = os.getcwd()
    sys.stderr.write(f'Current Folder: {current_dir}\n')
    os.chdir(data_path)

    for i, directory in enumerate(class_labels):
        sys.stderr.write(f"Start to Read {directory}\n")
        os.chdir(directory)
        file_list = natsorted(os.listdir('.'), alg=ns.PATH)
        for filename in tqdm(file_list):
            if filename.endswith('.csv') and not filename.endswith('time.csv'):
                filepath = os.path.join(os.getcwd(), filename)
                feature_vector = np.loadtxt(filepath, delimiter=",", dtype=np.float32)
                x.append(feature_vector)
                y.append(i)
        sys.stderr.write(f"End to Read {directory}\n")
        os.chdir('..')

    os.chdir(current_dir)
    return np.array(x), np.array(y)

def main():
    parser = argparse.ArgumentParser(description="Extract features from audio dataset.")
    parser.add_argument("--data_name", type=str, default="EMODB",
                        help="Name of the dataset, e.g., EMODB, CASIA, etc.")
    parser.add_argument("--mean_signal_length", type=int, default=96000,
                        help="Length (in samples) to pad/crop each audio signal.")
    parser.add_argument("--feature_type", type=str, default="MFCC",
                        help="Type of audio feature to extract (e.g., MFCC).")
    parser.add_argument("--embed_len", type=int, default=39,
                        help="Number of feature coefficients (e.g., MFCC).")
    parser.add_argument("--output_dir", type=str, default="./EMODB_MFCC_96",
                        help="Where to save CSV files of extracted features.")
    args = parser.parse_args()

    # Example label set for EMODB
    EMODB_LABELS = ("angry","boredom","disgust","fear","happy","neutral","sad")

    # Generate CSV
    generate_csv(
        csv_save=args.output_dir,
        data_name=args.data_name,
        feature_type=args.feature_type,
        embed_len=args.embed_len,
        mean_signal_length=args.mean_signal_length,
        class_labels=EMODB_LABELS
    )

    # Convert CSVs to .npy
    x, y = process_csv(args.output_dir, mfcc_len=args.embed_len, class_labels=EMODB_LABELS, flatten=False)
    y = to_categorical(y, num_classes=len(EMODB_LABELS))
    data_dict = {"x": x, "y": y}
    np.save(f"{args.data_name}.npy", data_dict)
    print(f"Feature arrays saved to {args.data_name}.npy")

if __name__ == "__main__":
    main()
