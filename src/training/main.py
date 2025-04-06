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
import datetime
from sklearn.metrics import confusion_matrix, classification_report

# Suppress warnings and TensorFlow logs for clean CLI experience
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf

# Enable Metal for Apple Silicon or CUDA for NVIDIA GPUs
if tf.test.is_built_with_cuda():
    os.environ['DEVICE'] = 'cuda'
else:
    os.environ['DEVICE'] = 'metal'

print("\n🖥️  Device Configuration:")

# Try to enable GPU (either Metal or CUDA)
try:
    # List available devices
    physical_devices = tf.config.list_physical_devices()
    print("   Available physical devices:", [device.name for device in physical_devices])
    
    # Try to create a simple operation on GPU
    with tf.device('/device:GPU:0'):
        # Test if GPU is working
        test_tensor = tf.zeros((1, 1))
        if tf.test.is_built_with_cuda():
            print("   ✅ CUDA GPU is available and configured")
            print("   🚀 Using CUDA GPU for computation")
        else:
            print("   ✅ Metal GPU is available and configured")
            print("   📱 Using Metal GPU for computation")
except Exception as e:
    print(f"   ⚠️ Could not configure GPU: {e}")
    print("   ℹ️ Falling back to CPU")

from src.models.model import SpeechEmotionModel
from tensorflow.keras.utils import to_categorical
from src.data_processing.load_dataset import load_fused_tensorflow_dataset

# Label sets for supported datasets
EMODB_LABELS = ("angry","boredom","disgust","fear","happy","neutral","sad")
RAVDESS_LABELS = ("calm","angry","disgust","fear","happy","neutral","sad","surprised")

LABEL_DICT = {
    "EMODB": EMODB_LABELS,
    "RAVDESS": RAVDESS_LABELS,
}

# Emotion mapping for cross-corpus testing
EMOTION_MAPPING = {
    # EMODB → RAVDESS mapping
    "EMODB_TO_RAVDESS": {
        "angry": ["angry"],
        "boredom": ["neutral", "calm"],  # Map boredom to both neutral and calm
        "disgust": ["disgust"],
        "fear": ["fear"],
        "happy": ["happy"],
        "neutral": ["neutral"],
        "sad": ["sad"],
        # No mapping for RAVDESS "surprised"
    },
    # RAVDESS → EMODB mapping
    "RAVDESS_TO_EMODB": {
        "angry": ["angry"],
        "calm": ["neutral", "boredom"],  # Map calm to both neutral and boredom
        "disgust": ["disgust"],
        "fear": ["fear"],
        "happy": ["happy"],
        "neutral": ["neutral"],
        "sad": ["sad"],
        "surprised": ["fear"]  # Map surprised to fear (closest match)
    }
}

def load_data_by_type(args):
    if args.feature_type == 'FUSION':
        x_source, y_source = load_fused_tensorflow_dataset(args.data)
    else:
        if args.feature_type == 'MFCC':
            data_path = os.path.join("data", "features", "MFCC", "EMODB_MFCC_96", f"{args.data}.npy")
        elif args.feature_type == 'LOGMEL':
            data_path = os.path.join("data", "features", "LOGMEL", "EMODB_LOGMEL_128", f"{args.data}.npy")
        elif args.feature_type == 'HUBERT':
            data_path = os.path.join("data", "features", "HUBERT", "EMODB_HUBERT", f"{args.data}.npy")
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
    parser.add_argument('--mode', type=str, default="train", choices=["train", "test", "test-cross-corpus"], help="Run mode: train, test, or test-cross-corpus.")
    parser.add_argument('--model_path', type=str, default='./saved_models/', help="Root directory to save model checkpoints.")
    parser.add_argument('--result_path', type=str, default='./results/', help="Root directory to save evaluation results.")
    parser.add_argument('--test_path', type=str, default='./test_models/EMODB', help="Path to load test model weights or folder containing .h5 files.")
    parser.add_argument('--data', type=str, default='EMODB', help="Dataset name: EMODB or RAVDESS")
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
    parser.add_argument('--visualize', action='store_true', help="Whether to visualize confusion matrix (requires matplotlib)")

    args = parser.parse_args()

    print("\n==============================")
    print("🎛️  SPEECH EMOTION RECOGNITION")
    print("==============================\n")
    print(f"🧪 Mode: {args.mode.upper()}")
    print(f"📁 Dataset: {args.data}")

    x_source, y_source = load_data_by_type(args)

    # Get the maximum label index in the data
    max_label = np.max(y_source)
    num_classes = max_label + 1
    print(f"📊 Data contains labels from 0 to {max_label}, using {num_classes} classes for one-hot encoding")
    
    # One-hot encode the labels
    y_source = to_categorical(y_source, num_classes=num_classes)
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

    # Configure device for training
    try:
        with tf.device('/device:GPU:0'):
            tf.zeros((1, 1))
            print("\n💻 Using Metal GPU for training")
    except:
        print("\n💻 Using CPU for training")

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

    elif args.mode == "test-cross-corpus":
        print("🧪 Starting cross-corpus validation...\n")
        
        # Automatically determine the test dataset based on the source dataset
        if args.data == "EMODB":
            args.test_data = "RAVDESS"
        elif args.data == "RAVDESS":
            args.test_data = "EMODB"
        else:
            raise ValueError(f"Unsupported dataset: {args.data}. Must be either EMODB or RAVDESS.")
            
        print(f"\n🔄 Cross-corpus validation: Training on {args.data}, testing on {args.test_data}")
        
        # Load the SOURCE dataset (what the model was trained on)
        print(f"📝 SOURCE dataset: {args.data} (training data)")
        source_class_labels = LABEL_DICT[args.data]
        
        # Load the TARGET dataset (what we'll test on)
        print(f"📝 TARGET dataset: {args.test_data} (testing data)")
        target_class_labels = LABEL_DICT[args.test_data]
        
        # Load target dataset for testing
        x_target, y_target = load_data_by_type(argparse.Namespace(**{**vars(args), "data": args.test_data}))
        
        # Determine emotion mapping key
        mapping_key = f"{args.data}_TO_{args.test_data}"
        if mapping_key not in EMOTION_MAPPING:
            raise ValueError(f"No emotion mapping defined for {mapping_key}")
        
        emotion_map = EMOTION_MAPPING[mapping_key]
        print("\n🗺️ Emotion mapping:")
        for source, targets in emotion_map.items():
            print(f"   {source} → {targets}")
            
        # Map indices to emotion names for source dataset
        source_idx_to_emotion = {i: emotion for i, emotion in enumerate(source_class_labels)}
        
        # Map emotion names to indices for target dataset
        target_emotion_to_idx = {emotion: i for i, emotion in enumerate(target_class_labels)}
        
        # Create mapping visualization
        print("\n🔀 Cross-corpus emotion mappings:")
        mapping_rows = []
        for source_idx, source_emotion in source_idx_to_emotion.items():
            target_emotions = emotion_map.get(source_emotion, [])
            target_emotion_str = ", ".join(target_emotions) if target_emotions else "N/A"
            target_indices = [target_emotion_to_idx.get(e, "N/A") for e in target_emotions]
            target_idx_str = ", ".join(str(idx) for idx in target_indices) if target_indices else "N/A"
            mapping_rows.append(f"   {source_idx} ({source_emotion}) → {target_idx_str} ({target_emotion_str})")
        
        # Sort and print for readability
        mapping_rows.sort()
        for row in mapping_rows:
            print(row)
        
        # Map numerical labels
        print("\n🔄 Mapping numerical labels...")
        y_source_emotions = np.argmax(y_target, axis=1) if len(y_target.shape) > 1 else y_target
        
        # Create a lookup for emotion names based on indices in target dataset
        target_idx_to_emotion = {i: emotion for i, emotion in enumerate(target_class_labels)}
        
        # Apply mapping and count valid/invalid mappings
        valid_count = 0
        invalid_count = 0
        invalid_indices = []
        
        # We'll create one-hot encoded vectors for multiple mappings
        y_mapped_multi = np.zeros((len(y_source_emotions), len(target_class_labels)))
        
        for i, label_idx in enumerate(y_source_emotions):
            # Get the emotion name for this index in the target dataset
            if label_idx < len(target_class_labels):
                target_emotion = target_idx_to_emotion[label_idx]
                
                # Find all source emotions that map to this target emotion
                mappings_found = False
                
                # Check each source emotion and its mappings
                for source_emotion, mapped_emotions in emotion_map.items():
                    if target_emotion in mapped_emotions:
                        # Found a mapping, get the index in source dataset
                        if source_emotion in target_emotion_to_idx:
                            # Set 1 for this mapping in the one-hot encoded vector
                            mapped_idx = target_emotion_to_idx[source_emotion]
                            y_mapped_multi[i, mapped_idx] = 1
                            mappings_found = True
                
                if mappings_found:
                    valid_count += 1
                else:
                    invalid_count += 1
                    invalid_indices.append(i)
            else:
                invalid_count += 1
                invalid_indices.append(i)
        
        print(f"   ✅ {valid_count} samples mapped successfully")
        if invalid_count > 0:
            print(f"   ⚠️ {invalid_count} samples had no valid mapping and will be excluded")
            
            # Remove samples with invalid mappings
            if invalid_count > 0:
                valid_indices = [i for i in range(len(y_mapped_multi)) if i not in invalid_indices]
                x_target = x_target[valid_indices]
                y_mapped_multi = y_mapped_multi[valid_indices]
        
        # Convert multi-hot to single class for each sample (take the max probability)
        y_mapped = np.zeros(len(y_mapped_multi), dtype=int)
        for i in range(len(y_mapped_multi)):
            if np.sum(y_mapped_multi[i]) > 0:
                # If there are multiple 1s, choose randomly among them
                possible_indices = np.where(y_mapped_multi[i] == 1)[0]
                y_mapped[i] = np.random.choice(possible_indices)
        
        # One-hot encode for the model
        y_target_mapped = to_categorical(y_mapped, num_classes=len(target_class_labels))
        
        print("\n🧠 Initializing model for evaluation...\n")
        model = SpeechEmotionModel(
            input_shape=x_target.shape[1:], 
            class_labels=target_class_labels, 
            args=args
        )
        
        # Generate unique result folder/file name
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        result_filename = f"cross_corpus_{args.data}_to_{args.test_data}_{timestamp}"
        result_dir = os.path.join(args.result_path, f"CrossCorpus/{args.data}_TO_{args.test_data}")
        os.makedirs(result_dir, exist_ok=True)
        
        print(f"\n🚀 Starting cross-corpus validation ({args.data} → {args.test_data})...\n")
        y_pred, accuracy = model.evaluate_test(
            x_target, 
            y_target_mapped, 
            path=args.test_path,
            result_filename=result_filename,
            result_dir=result_dir
        )
        
        # Optionally visualize confusion matrix
        if args.visualize:
            try:
                import matplotlib.pyplot as plt
                import seaborn as sns
                
                print("\n📊 Generating confusion matrix visualization...")
                
                # Get the number of unique classes in the predicted data
                unique_classes = np.unique(np.concatenate([y_mapped, y_pred]))
                num_unique_classes = len(unique_classes)
                
                # Determine which labels to use
                if num_unique_classes != len(target_class_labels):
                    print(f"⚠️ Using {num_unique_classes} labels for visualization instead of {len(target_class_labels)}")
                    present_classes = sorted(unique_classes)
                    display_labels = [target_class_labels[i] if i < len(target_class_labels) else f"Class {i}" 
                                     for i in present_classes]
                else:
                    display_labels = target_class_labels
                
                # Convert y_mapped to emotion names for better visualization
                y_true = y_mapped
                
                # Calculate confusion matrix
                cm = confusion_matrix(y_true, y_pred)
                cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                
                # Create figure
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                           xticklabels=display_labels,
                           yticklabels=display_labels)
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title(f'Cross-Corpus Validation: {args.data} → {args.test_data}\nAccuracy: {accuracy:.2f}')
                
                # Save figure
                viz_path = os.path.join(model.result_dir, f"{result_filename}_cm.png")
                plt.savefig(viz_path)
                print(f"✅ Visualization saved to: {viz_path}")
                
            except ImportError:
                print("⚠️ Visualization requires matplotlib and seaborn. Install with 'pip install matplotlib seaborn'")
        
        print("\n✅ Cross-corpus validation complete!\n")

if __name__ == "__main__":
    main()
