# extract_feature.py

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
    print("📁 STEP 1: Creating output directories...")
    current_dir = os.getcwd()
    if not os.path.exists(csv_save):
        print(f"📂 Creating folder: {csv_save}")
        os.makedirs(csv_save)

    for label_dir in class_labels:
        label_path = os.path.join(csv_save, label_dir)
        if not os.path.exists(label_path):
            os.makedirs(label_path)
            print(f"📂 Creating label subfolder: {label_path}")

    datapath, labels = [], []
    dataset_dir = os.path.join("data", "processed", data_name)
    print("\n🔍 STEP 2: Scanning for .wav files...\n")
    print(f"📌 Current directory: {current_dir}")
    print(f"📌 Looking in dataset folder: {dataset_dir}\n")

    for i, directory in enumerate(class_labels):
        print(f"▶️ Reading label '{directory}'...")
        emotion_dir = os.path.join(dataset_dir, directory)
        filelist = os.listdir(emotion_dir)
        for f in tqdm(filelist, desc=f"📦 {directory}"):
            if f.endswith('.wav'):
                filepath = os.path.join(emotion_dir, f)
                datapath.append(filepath)
                labels.append(i)
        print(f"✅ Finished reading '{directory}'\n")

    print("🎧 STEP 3: Extracting features and saving CSV files...\n")
    for (video_path, label) in tqdm(zip(datapath, labels), total=len(datapath), desc="🚀 Extracting"):
        filename = os.path.splitext(os.path.basename(video_path))[0]
        feature_vector = get_feature(file_path=video_path,
                                     feature_type=feature_type,
                                     mean_signal_length=mean_signal_length,
                                     embed_len=embed_len)
        csv_filename = f"{filename}_raw.csv"
        csv_filepath = os.path.join(csv_save, class_labels[label], csv_filename)
        np.savetxt(csv_filepath, feature_vector, delimiter=',')

    print("\n✅ All features extracted and saved as CSV!\n")

def process_csv(data_path: str,
                mfcc_len: int=39,
                class_labels: Tuple=("angry","boredom","disgust","fear","happy","neutral","sad"),
                flatten: bool=False):
    """
    Reads CSV files (previously generated) in data_path into NumPy arrays.
    """
    print("🧾 STEP 4: Compiling CSVs into .npy arrays...\n")
    x, y = [], []
    current_dir = os.getcwd()
    print(f"📌 Current directory: {current_dir}")
    os.chdir(data_path)

    for i, directory in enumerate(class_labels):
        print(f"🔄 Reading CSVs from: {directory}")
        os.chdir(directory)
        file_list = natsorted(os.listdir('.'), alg=ns.PATH)
        for filename in tqdm(file_list, desc=f"📄 {directory}"):
            if filename.endswith('.csv') and not filename.endswith('time.csv'):
                filepath = os.path.join(os.getcwd(), filename)
                feature_vector = np.loadtxt(filepath, delimiter=",", dtype=np.float32)
                x.append(feature_vector)
                y.append(i)
        os.chdir('..')
        print(f"✅ Done with '{directory}'\n")

    os.chdir(current_dir)
    print("✅ CSV compilation complete!\n")
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

    print("\n==============================")
    print("🎙️  STARTING FEATURE EXTRACTION")
    print("==============================\n")

    EMODB_LABELS = ("angry","boredom","disgust","fear","happy","neutral","sad")

    generate_csv(
        csv_save=args.output_dir,
        data_name=args.data_name,
        feature_type=args.feature_type,
        embed_len=args.embed_len,
        mean_signal_length=args.mean_signal_length,
        class_labels=EMODB_LABELS
    )

    x, y = process_csv(args.output_dir, mfcc_len=args.embed_len, class_labels=EMODB_LABELS, flatten=False)
    y = to_categorical(y, num_classes=len(EMODB_LABELS))
    data_dict = {"x": x, "y": y}

    # 1) Construct the output path under data/MFCC
    output_npy_path = os.path.join("data", "MFCC", f"{args.data_name}.npy")

    # 2) Save to the new location
    np.save(output_npy_path, data_dict)

    print(f"📦 Feature arrays saved to `{output_npy_path}`\n")
    print("✅ FEATURE EXTRACTION COMPLETE!\n")

if __name__ == "__main__":
    main()
