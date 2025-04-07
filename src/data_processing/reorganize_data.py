# reorganize_data.py
"""
reorganize_data.py

DESCRIPTION:
------------
This script helps reorganize EMODB and RAVDESS `.wav` audio files into subfolders labeled by emotion.

EMODB File Naming Convention:
    The 6th character indicates the emotion category.
    Example:
        - "03a01Fa.wav" → emotion code "F" → "fear"
        - "10b03Na.wav" → emotion code "N" → "neutral"

RAVDESS File Naming Convention:
    7-part numerical identifier (e.g., 03-01-06-01-02-01-12.wav)
    The 3rd part indicates the emotion:
        01 = neutral
        02 = calm
        03 = happy
        04 = sad
        05 = angry
        06 = fearful
        07 = disgust
        08 = surprised

It does the following:
1. Scans the raw folder of .wav files
2. Detects the emotion label based on the dataset's naming convention
3. Copies the file into a subdirectory based on its emotion

HOW TO RUN THIS SCRIPT:
------------------------
In your terminal or PowerShell:

    python -m src.data_processing.reorganize_data --dataset EMODB
    # or
    python -m src.data_processing.reorganize_data --dataset RAVDESS

This will organize files like:

    data/processed/EMODB/angry/
    data/processed/EMODB/fear/
    ...
    # or
    data/processed/RAVDESS/angry/
    data/processed/RAVDESS/fear/
    ...

MAPPING TABLES:
--------------
EMODB Emotion Code → Emotion Label:
    W → angry
    L → boredom
    E → disgust
    A → fear
    F → happy
    T → sad
    N → neutral

RAVDESS Emotion Code → Emotion Label:
    01 → neutral
    02 → calm
    03 → happy
    04 → sad
    05 → angry
    06 → fearful
    07 → disgust
    08 → surprised
"""

import argparse

# ======================
# 📦 Core Python Imports
# ======================
import os
import shutil

# ======================
# 🔖 Emotion Mappings
# ======================
EMODB_EMOTION_CODE_TO_LABEL = {
    "W": "angry",
    "L": "boredom",
    "E": "disgust",
    "A": "fear",
    "F": "happy",
    "T": "sad",
    "N": "neutral",
}

RAVDESS_EMOTION_CODE_TO_LABEL = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}


# ======================
# 🧠 Reorganizer Functions
# ======================
def reorganize_emodb(source_directory: str, destination_directory: str):
    """
    Copies EMODB .wav files into labeled subdirectories.
    Emotion is inferred from the 6th character in the filename.

    Args:
        source_directory (str): Directory containing original .wav files
        destination_directory (str): Where to create labeled emotion folders
    """
    print("\n==============================")
    print("📂 REORGANIZING EMODB DATA")
    print("==============================\n")

    print(f"🔍 Source Directory: {source_directory}")
    print(f"📁 Destination Directory: {destination_directory}\n")

    # Step 1: Create label folders
    print("📦 STEP 1: Creating label folders...")
    for emotion_label in set(EMODB_EMOTION_CODE_TO_LABEL.values()):
        emotion_folder_path = os.path.join(destination_directory, emotion_label)
        os.makedirs(emotion_folder_path, exist_ok=True)
        print(f"✅ Created folder: {emotion_folder_path}")

    # Step 2: Move files to folders
    print("\n🔄 STEP 2: Sorting files into emotion folders...\n")
    total_processed = 0
    total_unknown = 0
    skipped_files = []

    for file in os.listdir(source_directory):
        if file.endswith(".wav"):
            if len(file) > 5:
                emotion_code = file[5].upper()
            else:
                emotion_code = None
            emotion_label = EMODB_EMOTION_CODE_TO_LABEL.get(emotion_code)

            if emotion_label:
                source_path = os.path.join(source_directory, file)
                target_path = os.path.join(destination_directory, emotion_label, file)
                shutil.copy2(source_path, target_path)
                print(f"[✓] {file} → {emotion_label}/")
                total_processed += 1
            else:
                print(f"⚠️ Unknown emotion code in file: {file} | Length: {len(file)} | Char[5]: {file[5] if len(file) > 5 else 'N/A'}")
                skipped_files.append(file)
                total_unknown += 1

    # Log skipped files for inspection
    if skipped_files:
        with open("skipped_files_emodb.txt", "w") as f:
            for filename in skipped_files:
                f.write(f"{filename}\n")

    # Summary
    print(f"\n✅ Done! {total_processed} files reorganized successfully.")
    if total_unknown > 0:
        print(f"⚠️ {total_unknown} files had unknown emotion codes and were skipped.")
        print("📄 See 'skipped_files_emodb.txt' for full list.")
    print("\n🎉 REORGANIZATION COMPLETE!\n")


def reorganize_ravdess(source_directory: str, destination_directory: str):
    """
    Copies RAVDESS .wav files into labeled subdirectories.
    Emotion is inferred from the 3rd part of the filename (e.g., 03-01-06-01-02-01-12.wav → 06 → fearful).
    Walks through actor subdirectories (Actor_01, Actor_02, etc.).

    Args:
        source_directory (str): Directory containing actor subdirectories
        destination_directory (str): Where to create labeled emotion folders
    """
    print("\n==============================")
    print("📂 REORGANIZING RAVDESS DATA")
    print("==============================\n")

    print(f"🔍 Source Directory: {source_directory}")
    print(f"📁 Destination Directory: {destination_directory}\n")

    # Step 1: Create label folders
    print("📦 STEP 1: Creating label folders...")
    for emotion_label in set(RAVDESS_EMOTION_CODE_TO_LABEL.values()):
        emotion_folder_path = os.path.join(destination_directory, emotion_label)
        os.makedirs(emotion_folder_path, exist_ok=True)
        print(f"✅ Created folder: {emotion_folder_path}")

    # Step 2: Move files to folders
    print("\n🔄 STEP 2: Sorting files into emotion folders...\n")
    total_processed = 0
    total_unknown = 0
    skipped_files = []

    # Walk through all actor directories
    for actor_dir in os.listdir(source_directory):
        actor_path = os.path.join(source_directory, actor_dir)
        if not os.path.isdir(actor_path):
            continue

        print(f"\n👤 Processing {actor_dir}...")

        for file in os.listdir(actor_path):
            if file.endswith(".wav"):
                try:
                    # Split filename by '-' and get the emotion code (3rd part)
                    parts = file.split("-")
                    if len(parts) >= 3:
                        emotion_code = parts[2]
                        emotion_label = RAVDESS_EMOTION_CODE_TO_LABEL.get(emotion_code)

                        if emotion_label:
                            source_path = os.path.join(actor_path, file)
                            target_path = os.path.join(destination_directory, emotion_label, file)
                            shutil.copy2(source_path, target_path)
                            print(f"[✓] {file} → {emotion_label}/")
                            total_processed += 1
                        else:
                            print(f"⚠️ Unknown emotion code in file: {file} | Code: {emotion_code}")
                            skipped_files.append(os.path.join(actor_dir, file))
                            total_unknown += 1
                    else:
                        print(f"⚠️ Invalid filename format: {file}")
                        skipped_files.append(os.path.join(actor_dir, file))
                        total_unknown += 1
                except Exception as e:
                    print(f"⚠️ Error processing file {file}: {str(e)}")
                    skipped_files.append(os.path.join(actor_dir, file))
                    total_unknown += 1

    # Log skipped files for inspection
    if skipped_files:
        with open("skipped_files_ravdess.txt", "w") as f:
            for filename in skipped_files:
                f.write(f"{filename}\n")

    # Summary
    print(f"\n✅ Done! {total_processed} files reorganized successfully.")
    if total_unknown > 0:
        print(f"⚠️ {total_unknown} files had unknown emotion codes and were skipped.")
        print("📄 See 'skipped_files_ravdess.txt' for full list.")
    print("\n🎉 REORGANIZATION COMPLETE!\n")


# ======================
# 🚀 CLI Entrypoint
# ======================
def main():
    parser = argparse.ArgumentParser(description="Reorganize EMODB or RAVDESS .wav files into emotion-labeled subdirectories.")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["EMODB", "RAVDESS"],
        required=True,
        help="Dataset to process (EMODB or RAVDESS)",
    )
    args = parser.parse_args()

    # Set source and destination directories based on dataset
    source_dir = os.path.join("data", "raw", args.dataset)
    dest_dir = os.path.join("data", "processed", args.dataset)

    # Create destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)

    # Call appropriate reorganizer function
    if args.dataset == "EMODB":
        reorganize_emodb(source_dir, dest_dir)
    else:  # RAVDESS
        reorganize_ravdess(source_dir, dest_dir)


if __name__ == "__main__":
    main()
