# reorganize_data.py

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

import argparse
import os
import shutil

# Mapping from sixth character of the filename to the emotion label
EMOTION_MAP = {
    "W": "angry",  # e.g., Wut
    "L": "boredom",  # e.g., Langeweile
    "E": "disgust",  # e.g., Ekel
    "A": "fear",  # e.g., Angst
    "F": "happy",  # e.g., Freude
    "T": "sad",  # e.g., Trauer
    "N": "neutral",  # neutral
}


def reorganize_emodb(source_dir: str, dest_dir: str):
    """
    Copies EMODB .wav files from source_dir into labeled subfolders under dest_dir
    based on the sixth character in the filename.
    """
    print("\n==============================")
    print("📂 REORGANIZING EMODB DATA")
    print("==============================\n")

    print(f"🔍 Source Directory: {source_dir}")
    print(f"📁 Destination Directory: {dest_dir}\n")

    print("📦 STEP 1: Creating label folders...")
    for label in set(EMOTION_MAP.values()):
        path = os.path.join(dest_dir, label)
        os.makedirs(path, exist_ok=True)
        print(f"✅ Created folder: {path}")

    print("\n🔄 STEP 2: Sorting files into emotion folders...\n")
    total_files = 0
    unknown_labels = 0

    for filename in os.listdir(source_dir):
        if filename.endswith(".wav"):
            emotion_code = filename[5].upper()  # Sixth character
            label = EMOTION_MAP.get(emotion_code)
            if label is not None:
                src_path = os.path.join(source_dir, filename)
                dst_path = os.path.join(dest_dir, label, filename)
                shutil.copy(src_path, dst_path)
                total_files += 1
            else:
                print(f"⚠️ Unknown emotion code in file: {filename}")
                unknown_labels += 1

    print(f"\n✅ Done! {total_files} files reorganized successfully.")
    if unknown_labels > 0:
        print(f"⚠️ {unknown_labels} files had unknown emotion codes and were skipped.")
    print("\n🎉 REORGANIZATION COMPLETE!\n")


def main():
    parser = argparse.ArgumentParser(
        description="Reorganize EMODB .wav files into emotion-labeled subdirectories."
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        required=True,
        help="Path to the folder containing original .wav files.",
    )
    parser.add_argument(
        "--dest_dir",
        type=str,
        required=True,
        help="Destination path where labeled subfolders will be created.",
    )
    args = parser.parse_args()

    reorganize_emodb(args.source_dir, args.dest_dir)


if __name__ == "__main__":
    main()
