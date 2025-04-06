# reorganize_data.py
"""
reorganize_data.py

DESCRIPTION:
------------
This script helps reorganize EMODB `.wav` audio files into subfolders labeled by emotion.
The EMODB filenames follow a naming convention where the **6th character** indicates the emotion category.

For example:
    - "03a01Fa.wav" → emotion code "F" → "fear"
    - "10b03Na.wav" → emotion code "N" → "neutral"

It does the following:
1. Scans the raw folder of EMODB `.wav` files
2. Detects the emotion label from the 6th character of the filename
3. Copies the file into a subdirectory based on its emotion

HOW TO RUN THIS SCRIPT:
------------------------
In your terminal or PowerShell:

    python -m src.data_processing.reorganize_data \
        --source_dir datas/raw/EMODB/wav \
        --dest_dir datas/processed/EMODB

This will organize files like:

    datas/processed/EMODB/angry/
    datas/processed/EMODB/fear/
    ...

MAPPING TABLE:
--------------
Emotion Code → Emotion Label

    W → angry
    L → boredom
    E → disgust
    A → fear
    F → happy
    T → sad
    N → neutral

"""

# ======================
# 📦 Core Python Imports
# ======================
import os
import shutil
import argparse

# ======================
# 🔖 Emotion Mapping (EMODB code to label)
# ======================
EMOTION_CODE_TO_LABEL = {
    'W': 'angry',
    'L': 'boredom',
    'E': 'disgust',
    'A': 'fear',
    'F': 'happy',
    'T': 'sad',
    'N': 'neutral'
}

# ======================
# 🧠 Reorganizer Function
# ======================
def reorganize_emodb(source_directory: str, destination_directory: str):
    """
    Copies EMODB .wav files into labeled subdirectories under destination_directory.
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
    for emotion_label in set(EMOTION_CODE_TO_LABEL.values()):
        emotion_folder_path = os.path.join(destination_directory, emotion_label)
        os.makedirs(emotion_folder_path, exist_ok=True)
        print(f"✅ Created folder: {emotion_folder_path}")

    # Step 2: Move files to folders
    print("\n🔄 STEP 2: Sorting files into emotion folders...\n")
    total_processed = 0
    total_unknown = 0
    skipped_files = []

    for file in os.listdir(source_directory):
        if file.endswith('.wav'):
            if len(file) > 5:
                emotion_code = file[5].upper()
            else:
                emotion_code = None
            emotion_label = EMOTION_CODE_TO_LABEL.get(emotion_code)

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
        with open("skipped_files.txt", "w") as f:
            for filename in skipped_files:
                f.write(f"{filename}\n")

    # Summary
    print(f"\n✅ Done! {total_processed} files reorganized successfully.")
    if total_unknown > 0:
        print(f"⚠️ {total_unknown} files had unknown emotion codes and were skipped.")
        print("📄 See 'skipped_files.txt' for full list.")
    print("\n🎉 REORGANIZATION COMPLETE!\n")

def reorganize_features():
    """
    Default wrapper to run EMODB reorganization for automated pipelines.
    """
    source_dir = "datas/raw/EMODB/wav"
    dest_dir = "datas/processed/EMODB"
    reorganize_emodb(source_dir, dest_dir)

# ======================
# 🚀 CLI Entrypoint
# ======================
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
