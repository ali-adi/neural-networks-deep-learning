"""
reorganize_data.py

HOW TO RUN THIS FILE:
---------------------
Example:
    python reorganize_data.py --source_dir /path/to/raw/EMODB --dest_dir /path/to/processed/EMODB

DESCRIPTION:
Converts unorganized EMODB .wav files into a directory structure labeled by emotion.
Originally created from a one-cell Jupyter notebook (preprocess.ipynb).
"""

import os
import shutil
import argparse

# Mapping from sixth character of the filename to the emotion label
EMOTION_MAP = {
    'W': 'angry',    # e.g., Wut
    'L': 'boredom',  # e.g., Langeweile
    'E': 'disgust',  # e.g., Ekel
    'A': 'fear',     # e.g., Angst
    'F': 'happy',    # e.g., Freude
    'T': 'sad',      # e.g., Trauer
    'N': 'neutral'   # neutral
}

def reorganize_emodb(source_dir: str, dest_dir: str):
    """
    Copies EMODB .wav files from source_dir into labeled subfolders under dest_dir
    based on the sixth character in the filename.

    Args:
        source_dir (str): Path to the unorganized .wav files.
        dest_dir (str): Destination directory where subfolders (by emotion) are created.
    """
    # Create subdirectories for each emotion label
    for label in set(EMOTION_MAP.values()):
        os.makedirs(os.path.join(dest_dir, label), exist_ok=True)

    # Iterate over .wav files in the source directory
    for filename in os.listdir(source_dir):
        if filename.endswith('.wav'):
            emotion_code = filename[5].upper()  # Sixth character
            label = EMOTION_MAP.get(emotion_code)
            if label is not None:
                src_path = os.path.join(source_dir, filename)
                dst_path = os.path.join(dest_dir, label, filename)
                shutil.copy(src_path, dst_path)
            else:
                print(f"Unknown emotion code in file: {filename}")

def main():
    parser = argparse.ArgumentParser(
        description="Reorganize EMODB .wav files into emotion-labeled subdirectories."
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        required=True,
        help="Path to the folder containing original .wav files."
    )
    parser.add_argument(
        "--dest_dir",
        type=str,
        required=True,
        help="Destination path where labeled subfolders will be created."
    )
    args = parser.parse_args()

    reorganize_emodb(args.source_dir, args.dest_dir)

if __name__ == "__main__":
    main()
