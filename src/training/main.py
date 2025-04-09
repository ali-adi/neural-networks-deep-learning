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

import argparse
import datetime
import os
import shutil
import sys
import json
import gc
import time
import threading

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Suppress warnings and TensorFlow logs for clean CLI experience
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings

warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras.utils import to_categorical

# Function to monitor GPU memory usage
def monitor_gpu_memory():
    """Thread function to periodically check and report GPU memory usage"""
    if tf.test.is_built_with_cuda():
        # Set memory limit to 64GB (in bytes)
        memory_limit_gb = 64
        memory_limit_bytes = memory_limit_gb * 1024 * 1024 * 1024
        
        while True:
            try:
                # Print current memory usage
                try:
                    # Get memory usage using TensorFlow
                    devices = tf.config.list_physical_devices('GPU')
                    if devices:
                        # Try different methods to get memory info
                        memory_info = None
                        
                        # Method 1: Using experimental.get_memory_info
                        try:
                            memory_info = tf.config.experimental.get_memory_info('GPU:0')
                        except:
                            pass
                            
                        # Method 2: Using tf.config.experimental.get_memory_growth
                        if not memory_info:
                            try:
                                # Create a small tensor to force memory allocation
                                with tf.device('/GPU:0'):
                                    test_tensor = tf.zeros((1000, 1000))
                                    memory_info = tf.config.experimental.get_memory_info('GPU:0')
                            except:
                                pass
                                
                        # Method 3: Using nvidia-smi via subprocess
                        if not memory_info:
                            try:
                                import subprocess
                                result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'])
                                used_mb, total_mb = map(int, result.decode('utf-8').strip().split(','))
                                print(f"\n💾 GPU Memory Usage: {used_mb} MB / {total_mb} MB ({used_mb/total_mb*100:.1f}%)")
                                
                                # Check if memory usage exceeds the limit
                                if used_mb * 1024 * 1024 > memory_limit_bytes:
                                    print(f"⚠️ WARNING: Memory usage exceeds {memory_limit_gb}GB limit!")
                                    print(f"   Consider reducing batch size or model complexity")
                            except:
                                pass
                                
                        # If we got memory info from TensorFlow, use it
                        if memory_info:
                            used_mb = memory_info.get('current', 0) / (1024 * 1024)
                            used_gb = used_mb / 1024
                            print(f"\n💾 GPU Memory Usage: {used_mb:.2f} MB ({used_gb:.2f} GB)")
                            
                            # Check if memory usage exceeds the limit
                            if memory_info.get('current', 0) > memory_limit_bytes:
                                print(f"⚠️ WARNING: Memory usage exceeds {memory_limit_gb}GB limit!")
                                print(f"   Consider reducing batch size or model complexity")
                except Exception as e:
                    print(f"\n⚠️ Could not get GPU memory info: {e}")
                
                time.sleep(30)  # Check every 30 seconds
            except Exception as e:
                print(f"Error in GPU monitor: {e}")
                break
            except KeyboardInterrupt:
                break

print("\n🖥️  Device Configuration:")

# Try a more direct approach to detect NVIDIA GPUs
nvidia_gpu_available = False
try:
    import subprocess
    result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode == 0:
        nvidia_gpu_output = result.stdout.decode('utf-8')
        print(f"   ✅ NVIDIA GPU detected via nvidia-smi")
        print(f"   GPU info: {nvidia_gpu_output.split('|')[0].strip().split('GPU')[0].strip()}")
        nvidia_gpu_available = True
    else:
        print(f"   ❌ nvidia-smi command failed: {result.stderr.decode('utf-8')}")
except Exception as e:
    print(f"   ❌ Error checking for NVIDIA GPU: {e}")

# Check for CUDA availability
cuda_available = tf.test.is_built_with_cuda()
print(f"   CUDA built with TensorFlow: {cuda_available}")

# Force TensorFlow to see CUDA device if available
if nvidia_gpu_available and cuda_available:
    # Set visible devices explicitly
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print(f"   ✅ Set CUDA_VISIBLE_DEVICES=0 to force GPU visibility")
    
    # Try to force-enable GPU in TensorFlow
    physical_devices = tf.config.list_physical_devices()
    print("   Available physical devices before configuration:", [device.name for device in physical_devices])
    
    # Try alternate ways to get GPU devices
    try:
        # Try with experimental.get_visible_devices()
        visible_devices = tf.config.experimental.get_visible_devices()
        print("   Visible devices:", [device.name for device in visible_devices])
        
        # Try listing CUDA devices directly
        cuda_devices = tf.config.experimental.list_physical_devices('GPU')
        if cuda_devices:
            print(f"   Found {len(cuda_devices)} CUDA device(s) with experimental API")
        
        # Force device visibility if needed
        if not cuda_devices:
            print("   Attempting to force GPU visibility...")
            # Create a dummy CUDA session to initialize GPU
            import os
            os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
            
            # Try to create a simple CUDA operation
            with tf.device('/device:GPU:0'):
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
                c = tf.matmul(a, b)
                print(f"   Test CUDA operation result: {c}")
                print("   ✅ Successfully executed operation on CUDA GPU")
    except Exception as e:
        print(f"   ⚠️ Error during advanced GPU detection: {e}")

# List physical devices after configuration attempts
physical_devices = tf.config.list_physical_devices()
print("   Available physical devices after configuration:", [device.name for device in physical_devices])

# Configure GPU if found
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"   Found {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")
    
    # Set memory growth to avoid allocating all memory at once
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"   ✅ Memory growth enabled for {gpu.name}")
        except RuntimeError as e:
            print(f"   ⚠️ Could not set memory growth for {gpu.name}: {e}")
    
    # Set memory limit to 64GB
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=64 * 1024)]  # 64GB in MB
        )
        print(f"   ✅ Memory limit set to 64GB for {gpus[0].name}")
    except RuntimeError as e:
        print(f"   ⚠️ Could not set memory limit: {e}")
    
    # Determine which GPU backend to use
    if cuda_available:
        os.environ["DEVICE"] = "cuda"
        print("   ✅ Using CUDA GPU for computation")
        
        # Start a background thread to monitor GPU memory
        try:
            gpu_monitor = threading.Thread(target=monitor_gpu_memory, daemon=True)
            gpu_monitor.start()
            print("   🔍 GPU memory monitoring thread started")
        except Exception as e:
            print(f"   ⚠️ Could not start GPU monitoring: {e}")
    else:
        os.environ["DEVICE"] = "metal"
        print("   ✅ Using Metal GPU for computation")
elif nvidia_gpu_available:
    # If nvidia-smi shows a GPU but TensorFlow doesn't see it, there might be a configuration issue
    print("   ⚠️ NVIDIA GPU detected with nvidia-smi but not recognized by TensorFlow")
    print("   🔧 Setting environment variables for CUDA visibility")
    
    # Try to force GPU to be visible
    os.environ["DEVICE"] = "cuda"
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print("   ✅ Forced CUDA environment variables for computation")
else:
    print("   ℹ️ No GPU found, falling back to CPU")
    os.environ["DEVICE"] = "cpu"

# Try to create a simple operation on GPU if available
if nvidia_gpu_available or gpus:
    try:
        with tf.device("/device:GPU:0"):
            test_tensor = tf.zeros((1, 1))
            print(f"   ✅ Successfully created tensor on GPU: {os.environ['DEVICE']}")
    except Exception as e:
        print(f"   ⚠️ Could not create tensor on GPU: {e}")
        print("   ℹ️ Falling back to CPU")
        os.environ["DEVICE"] = "cpu"

from src.data_processing.load_dataset import load_fused_tensorflow_dataset
from src.models.model import SpeechEmotionModel

# Label sets for supported datasets
EMODB_LABELS = ("angry", "boredom", "disgust", "fear", "happy", "neutral", "sad")
RAVDESS_LABELS = (
    "calm",
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprised",
)

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
        "surprised": ["fear"],  # Map surprised to fear (closest match)
    },
}


def load_data_by_type(args):
    if args.feature_type == "FUSION":
        x_source, y_source = load_fused_tensorflow_dataset(args.data)
    else:
        feature_type_map = {
            "MFCC": f"{args.data}_MFCC_96",
            "LOGMEL": f"{args.data}_LOGMEL_128",
            "HUBERT": f"{args.data}_HUBERT",
        }

        if args.feature_type not in feature_type_map:
            raise ValueError("Unsupported feature type. Must be MFCC, LOGMEL, HUBERT, or FUSION.")

        dir_name = feature_type_map[args.feature_type]
        data_path = os.path.join("data", "features", args.feature_type, dir_name, f"{args.data}.npy")

        print(f"\U0001F4E6 Data source: {data_path}")
        loaded_data = np.load(data_path, allow_pickle=True).item()
        if not isinstance(loaded_data, dict) or "x" not in loaded_data or "y" not in loaded_data:
            raise ValueError(f"Invalid data format in {data_path}")
        x_source = loaded_data["x"]
        y_source = loaded_data["y"]

        # Special handling for HUBERT features to ensure they have the correct shape
        if args.feature_type == "HUBERT" and len(x_source.shape) == 1 and x_source.dtype == object:
            # For HUBERT, if it's a 1D object array of features, we can use it directly
            # The model will handle reshaping in the create_model function
            print(f"✅ Using HUBERT features as 1D embeddings")

    return x_source, y_source


def main():
    import numpy as np  # Add numpy import inside the function
    parser = argparse.ArgumentParser(description="Train or test a temporal conv-based speech emotion model.")
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "test", "test-cross-corpus", "train-lmmd"],
        help="Run mode: train, test, test-cross-corpus, or train-lmmd.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./saved_models/",
        help="Root directory to save model checkpoints.",
    )
    parser.add_argument(
        "--result_path",
        type=str,
        default="./results/",
        help="Root directory to save evaluation results.",
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default="./test_models/EMODB",
        help="Path to load test model weights or folder containing .h5 files.",
    )
    parser.add_argument("--data", type=str, default="EMODB", help="Dataset name: EMODB or RAVDESS")
    parser.add_argument(
        "--feature_type",
        type=str,
        default="FUSION",
        choices=["MFCC", "LOGMEL", "HUBERT", "FUSION"],
        help="Type of audio feature used.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0003,
        help="Lower learning rate for finer convergence.",
    )
    parser.add_argument("--beta1", type=float, default=0.93, help="Adam beta1.")
    parser.add_argument("--beta2", type=float, default=0.98, help="Adam beta2.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Smaller batch size to help generalization.",
    )
    parser.add_argument("--epoch", type=int, default=300, help="Number of epochs for training.")
    parser.add_argument("--dropout", type=float, default=0.4, help="Dropout rate to reduce overfitting.")
    parser.add_argument("--random_seed", type=int, default=46, help="Seed for reproducibility.")
    parser.add_argument("--activation", type=str, default="relu", help="Activation function.")
    parser.add_argument(
        "--filter_size",
        type=int,
        default=128,
        help="Number of filters for richer features",
    )
    parser.add_argument(
        "--dilation_size",
        type=int,
        default=8,
        help="Maximum power-of-two dilation size.",
    )
    parser.add_argument(
        "--kernel_size",
        type=int,
        default=3,
        help="Size of convolutional kernel.",
    )
    parser.add_argument(
        "--stack_size",
        type=int,
        default=3,
        help="Number of temporal blocks to stack.",
    )
    parser.add_argument(
        "--split_fold",
        type=int,
        default=10,
        help="Number of folds for k-fold cross-validation.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID to use (0 for first GPU).",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enable visualization of training metrics.",
    )
    parser.add_argument(
        "--lmmd_weight",
        type=float,
        default=0.5,
        help="Weight for LMMD loss in domain adaptation.",
    )
    parser.add_argument(
        "--target_data",
        type=str,
        default=None,
        help="Target dataset for domain adaptation (EMODB or RAVDESS).",
    )

    args = parser.parse_args()

    print("\n==============================")
    print("🎛️  SPEECH EMOTION RECOGNITION")
    print("==============================\n")
    print(f"🧪 Mode: {args.mode.upper()}")
    print(f"📁 Dataset: {args.data}")

    x_source, y_source = load_data_by_type(args)

    # CRITICAL FIX: Remap labels to consecutive integers for each dataset
    if args.data == "EMODB":
        # For EMODB, map to 0-6
        num_classes = 7
        # Define a mapping table from existing indices to sequential ones
        # EMODB labels can be [0,1,2,3,4,5,6] but could have other values from LABEL_MAP
        unique_labels = np.unique(y_source)
        print(f"🔄 Original unique labels found: {unique_labels}")

        # Create a mapping dictionary where each original label maps to a position from 0-6
        label_mapping = {label: idx for idx, label in enumerate(sorted(unique_labels))}
        print(f"🔄 Remapping labels to: {label_mapping}")

        # Apply the mapping to all labels
        y_source_remapped = np.array([label_mapping[label] for label in y_source])
        y_source = y_source_remapped

    elif args.data == "RAVDESS":
        # For RAVDESS, map to 0-7
        num_classes = 8
        unique_labels = np.unique(y_source)
        print(f"🔄 Original unique labels found: {unique_labels}")

        # Create a mapping dictionary for RAVDESS (could have labels like [0,2,4,5,6,7,8,9])
        label_mapping = {label: idx for idx, label in enumerate(sorted(unique_labels))}
        print(f"🔄 Remapping labels to: {label_mapping}")

        # Apply the mapping
        y_source_remapped = np.array([label_mapping[label] for label in y_source])
        y_source = y_source_remapped
    else:
        # For other datasets, just use the number of unique labels
        num_classes = len(np.unique(y_source))

    print(f"📊 Using {num_classes} classes for one-hot encoding")

    # One-hot encode the labels with the remapped values
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
        with tf.device("/device:GPU:0"):
            tf.zeros((1, 1))
            if os.environ["DEVICE"] == "cuda":
                print("\n💻 Using CUDA GPU for training")
            elif os.environ["DEVICE"] == "metal":
                print("\n💻 Using Metal GPU for training")
            else:
                print("\n💻 Using CPU for training")
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
        
        # Initialize model first
        print("\n🧠 Initializing model...")
        model = SpeechEmotionModel(input_shape=x_source.shape[1:], class_labels=class_labels, args=args)
        
        # Create model architecture explicitly
        print("\n🔄 Creating model architecture...")
        model.create_model()
        
        # Find latest model weights
        if os.path.isdir(args.test_path):
            # Check for both .h5 files and TensorFlow checkpoints
            h5_files = [os.path.join(args.test_path, f) for f in os.listdir(args.test_path) if f.endswith(".h5")]
            checkpoint_files = [os.path.join(args.test_path, f) for f in os.listdir(args.test_path) if f.endswith(".index")]
            
            if not h5_files and not checkpoint_files:
                raise FileNotFoundError(f"No .h5 or checkpoint files found in {args.test_path}")
            
            if h5_files:
                args.test_path = max(h5_files, key=os.path.getmtime)
                print(f"📌 Latest H5 model found: {args.test_path}")
            else:
                # For checkpoint files, remove the .index extension to get the base path
                checkpoint_base = max(checkpoint_files, key=os.path.getmtime)[:-6]  # Remove .index
                args.test_path = checkpoint_base
                print(f"📌 Latest checkpoint model found: {args.test_path}")
        
        # Load weights with expect_partial() to suppress optimizer variable warnings
        print(f"📋 Loading model weights from: {args.test_path}")
        try:
            model.model.load_weights(args.test_path).expect_partial()
            print("✅ Model weights loaded successfully")
            model.evaluate_test(x_source, y_source, path=args.test_path)
            print("\n✅ Testing complete!\n")
        except Exception as e:
            print(f"❌ Error loading model weights: {str(e)}")
            raise

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

        # Validate datasets
        print("\n🔍 Validating datasets:")
        print(f"Source dataset: {x_source.shape} samples, {y_source.shape} labels")
        print(f"Target dataset: {x_target.shape} samples, {y_target.shape} labels")
        
        # Check feature dimensions match
        if x_source.shape[2] != x_target.shape[2]:
            raise ValueError(f"Feature dimensions mismatch: source has {x_source.shape[2]} features, target has {x_target.shape[2]} features")
        
        # Remap target labels to consecutive integers starting from 0
        unique_target_labels = np.unique(y_target)
        print(f"\n🔄 Original target labels found: {unique_target_labels}")
        target_label_mapping = {label: idx for idx, label in enumerate(sorted(unique_target_labels))}
        print(f"🔄 Remapping target labels to: {target_label_mapping}")
        y_target_remapped = np.array([target_label_mapping[label] for label in y_target])
        
        # One-hot encode with the correct number of classes
        num_target_classes = len(target_class_labels)
        print(f"📊 Using {num_target_classes} classes for target one-hot encoding")
        y_target = to_categorical(y_target_remapped, num_classes=num_target_classes)
        print(f"✅ Target labels shape after one-hot encoding: {y_target.shape}")
        
        # Ensure source labels are also properly one-hot encoded
        if len(y_source.shape) == 1:
            num_source_classes = len(source_class_labels)
            print(f"\n🔄 Converting source labels to one-hot encoding with {num_source_classes} classes...")
            y_source = to_categorical(y_source, num_classes=num_source_classes)
            print(f"✅ Source labels shape after one-hot encoding: {y_source.shape}")
            
        if y_source.shape[1] != len(source_class_labels) or y_target.shape[1] != len(target_class_labels):
            raise ValueError("Label dimensions don't match number of classes")

        # Handle sequence length differences
        max_seq_len = max(x_source.shape[1], x_target.shape[1])
        feature_dim = x_source.shape[2]
        
        print(f"\n📊 Sequence length adjustment:")
        print(f"   Source length: {x_source.shape[1]}")
        print(f"   Target length: {x_target.shape[1]}")
        print(f"   Using max length: {max_seq_len}")
        
        # Pad both source and target data to match sequence length
        if x_source.shape[1] < max_seq_len:
            pad_length = max_seq_len - x_source.shape[1]
            x_source = np.pad(
                x_source,
                ((0, 0), (0, pad_length), (0, 0)),
                mode='constant'
            )
            print(f"   Padded source data: {x_source.shape}")
            
        if x_target.shape[1] < max_seq_len:
            pad_length = max_seq_len - x_target.shape[1]
            x_target = np.pad(
                x_target,
                ((0, 0), (0, pad_length), (0, 0)),
                mode='constant'
            )
            print(f"   Padded target data: {x_target.shape}")
            
        print(f"\n✅ Final shapes:")
        print(f"   Source: {x_source.shape}")
        print(f"   Target: {x_target.shape}")

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
        model = SpeechEmotionModel(input_shape=x_target.shape[1:], class_labels=target_class_labels, args=args)

        # Generate unique result folder/file name
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        result_filename = f"cross_corpus_{args.data}_to_{args.test_data}_{timestamp}"
        result_dir = os.path.join(args.result_path, f"CrossCorpus/{args.data}_TO_{args.test_data}")
        os.makedirs(result_dir, exist_ok=True)

        print(f"\n🚀 Starting cross-corpus validation ({args.data} → {args.test_data})...\n")
        y_pred, accuracy = model.evaluate_test(
            x_target,
            y_target_mapped,
            path=args.test_path,
            result_filename=result_filename,
            result_dir=result_dir,
        )

        # Optionally visualize confusion matrix and training metrics
        if args.visualize:
            try:
                import matplotlib.pyplot as plt
                import seaborn as sns
                import pandas as pd
                import numpy as np

                print("\n📊 Generating visualizations...")
                os.makedirs(model.result_dir, exist_ok=True)

                # 1. Confusion Matrix Visualization
                print("   → Creating confusion matrix...")
                unique_classes = np.unique(np.concatenate([y_mapped, y_pred]))
                num_unique_classes = len(unique_classes)

                if num_unique_classes != len(target_class_labels):
                    print(f"   ⚠️ Using {num_unique_classes} labels instead of {len(target_class_labels)}")
                    present_classes = sorted(unique_classes)
                    display_labels = [(target_class_labels[i] if i < len(target_class_labels) else f"Class {i}") 
                                    for i in present_classes]
                else:
                    display_labels = target_class_labels

                y_true = y_mapped
                cm = confusion_matrix(y_true, y_pred)
                cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

                plt.figure(figsize=(12, 8))
                sns.heatmap(
                    cm_norm,
                    annot=True,
                    fmt=".2f",
                    cmap="Blues",
                    xticklabels=display_labels,
                    yticklabels=display_labels,
                )
                plt.xlabel("Predicted")
                plt.ylabel("True")
                plt.title(f"Cross-Corpus Validation: {args.data} → {args.test_data}\nAccuracy: {accuracy:.2f}")

                viz_path = os.path.join(model.result_dir, f"{result_filename}_confusion_matrix.png")
                plt.savefig(viz_path, bbox_inches='tight', dpi=300)
                plt.close()

                # 2. Training History Visualization (if available)
                if hasattr(model, 'history') and model.history is not None:
                    print("   → Creating training history plots...")
                    
                    # Convert history to DataFrame for easier manipulation
                    history_df = pd.DataFrame(model.history.history)
                    
                    # Apply rolling smoothing (window of 5 epochs)
                    smoothing_window = 5
                    smoothed_metrics = {}
                    
                    for metric in history_df.columns:
                        if len(history_df[metric]) >= smoothing_window:
                            smoothed_metrics[f"{metric}_smooth"] = history_df[metric].rolling(
                                window=smoothing_window, center=True
                            ).mean()
                        else:
                            smoothed_metrics[f"{metric}_smooth"] = history_df[metric]
                    
                    # Create subplots for different metrics
                    fig, axes = plt.subplots(2, 1, figsize=(12, 12))
                    
                    # Plot accuracy metrics
                    ax1 = axes[0]
                    if 'accuracy' in history_df:
                        ax1.plot(history_df.index, history_df['accuracy'], 
                                'b-', alpha=0.3, label='Training Accuracy')
                        ax1.plot(history_df.index, smoothed_metrics['accuracy_smooth'], 
                                'b-', label='Training Accuracy (Smoothed)')
                    if 'val_accuracy' in history_df:
                        ax1.plot(history_df.index, history_df['val_accuracy'], 
                                'r-', alpha=0.3, label='Validation Accuracy')
                        ax1.plot(history_df.index, smoothed_metrics['val_accuracy_smooth'], 
                                'r-', label='Validation Accuracy (Smoothed)')
                    ax1.set_title('Model Accuracy over Epochs')
                    ax1.set_xlabel('Epoch')
                    ax1.set_ylabel('Accuracy')
                    ax1.legend()
                    ax1.grid(True)
                    
                    # Plot loss metrics
                    ax2 = axes[1]
                    if 'loss' in history_df:
                        ax2.plot(history_df.index, history_df['loss'], 
                                'b-', alpha=0.3, label='Training Loss')
                        ax2.plot(history_df.index, smoothed_metrics['loss_smooth'], 
                                'b-', label='Training Loss (Smoothed)')
                    if 'val_loss' in history_df:
                        ax2.plot(history_df.index, history_df['val_loss'], 
                                'r-', alpha=0.3, label='Validation Loss')
                        ax2.plot(history_df.index, smoothed_metrics['val_loss_smooth'], 
                                'r-', label='Validation Loss (Smoothed)')
                    ax2.set_title('Model Loss over Epochs')
                    ax2.set_xlabel('Epoch')
                    ax2.set_ylabel('Loss')
                    ax2.legend()
                    ax2.grid(True)
                    
                    # Adjust layout and save
                    plt.tight_layout()
                    history_path = os.path.join(model.result_dir, f"{result_filename}_training_history.png")
                    plt.savefig(history_path, bbox_inches='tight', dpi=300)
                    plt.close()
                    
                    print(f"✅ Visualizations saved to: {model.result_dir}")
                else:
                    print("   ⚠️ No training history available for visualization")

            except ImportError as e:
                print(f"⚠️ Visualization requires additional packages: {str(e)}")
                print("   Install with 'pip install matplotlib seaborn pandas'")
            except Exception as e:
                print(f"⚠️ Error during visualization: {str(e)}")

        print("\n✅ Cross-corpus validation complete!\n")

    elif args.mode == "train-lmmd":
        # Domain adaptation training with LMMD
        print("🚀 Starting domain adaptation training with LMMD loss")

        # Set the target dataset - if not specified, use the opposite of source
        target_dataset = args.target_data if args.target_data else ("RAVDESS" if args.data == "EMODB" else "EMODB")
        print(f"📊 Source domain: {args.data}")
        print(f"📊 Target domain: {target_dataset}")

        # Save the original data args
        source_dataset = args.data

        # Load target domain data
        args.data = target_dataset
        x_target, y_target = load_data_by_type(args)

        # Restore original args
        args.data = source_dataset

        # Check shapes and adjust if needed
        print(f"Source data shape: {x_source.shape}")
        print(f"Target data shape: {x_target.shape}")

        # Get labels for both datasets
        source_class_labels = LABEL_DICT[args.data]
        target_class_labels = LABEL_DICT[target_dataset]

        print(f"Source classes: {source_class_labels}")
        print(f"Target classes: {target_class_labels}")

        # Initialize the model with source domain class labels
        ser_model = SpeechEmotionModel(input_shape=x_source.shape[1:], class_labels=source_class_labels, args=args)

        # Train with domain adaptation
        # Initialize LMMD loss by setting the flag
        args.use_lmmd = True
        ser_model.train_with_domain_adaptation(x_source, y_source, x_target, y_target)

        print("\n✅ Domain adaptation training complete!")
        print(f"🏆 Best model saved to: {ser_model.best_fold_weight_path}")

        # Get classification report
        if ser_model.trained:
            # Evaluate on source validation data
            y_source_pred = ser_model.model.predict(x_source)
            y_source_pred = np.argmax(y_source_pred, axis=1)
            y_source_true = y_source if len(y_source.shape) == 1 else np.argmax(y_source, axis=1)

            source_report = classification_report(
                y_source_true,
                y_source_pred,
                target_names=source_class_labels,
                digits=4,
                zero_division=0,
            )

            print("\n📊 Source Domain Classification Report:")
            print(source_report)

            # Also evaluate on target data
            y_target_pred = ser_model.model.predict(x_target)
            y_target_pred = np.argmax(y_target_pred, axis=1)
            y_target_true = y_target if len(y_target.shape) == 1 else np.argmax(y_target, axis=1)

            # Adjust target class names if needed
            if len(target_class_labels) != max(y_target_true) + 1:
                target_names = [f"Class {i}" for i in range(max(y_target_true) + 1)]
            else:
                target_names = target_class_labels

            target_report = classification_report(
                y_target_true,
                y_target_pred,
                target_names=target_names,
                digits=4,
                zero_division=0,
            )

            print("\n📊 Target Domain Classification Report:")
            print(target_report)

        # Generate and save confusion matrix if requested
        if args.visualize:
            try:
                import matplotlib.pyplot as plt
                import seaborn as sns

                print("\n📊 Generating confusion matrix visualizations...")

                # Convert labels to appropriate format if they're not already
                if not "y_source_true" in locals() or not "y_source_pred" in locals():
                    # If we haven't evaluated on source data yet, do it now
                    y_source_pred = ser_model.model.predict(x_source)
                    y_source_pred = np.argmax(y_source_pred, axis=1)
                    y_source_true = y_source if len(y_source.shape) == 1 else np.argmax(y_source, axis=1)

                # Similarly ensure target predictions are available
                if not "y_target_true" in locals() or not "y_target_pred" in locals():
                    y_target_pred = ser_model.model.predict(x_target)
                    y_target_pred = np.argmax(y_target_pred, axis=1)
                    y_target_true = y_target if len(y_target.shape) == 1 else np.argmax(y_target, axis=1)

                    # Adjust target class names if needed
                    if len(target_class_labels) != max(y_target_true) + 1:
                        target_names = [f"Class {i}" for i in range(max(y_target_true) + 1)]
                    else:
                        target_names = target_class_labels

                # Source domain visualization
                cm_source = confusion_matrix(y_source_true, y_source_pred)
                cm_source_norm = cm_source.astype("float") / cm_source.sum(axis=1)[:, np.newaxis]

                plt.figure(figsize=(10, 8))
                sns.heatmap(
                    cm_source_norm,
                    annot=True,
                    fmt=".2f",
                    cmap="Blues",
                    xticklabels=source_class_labels,
                    yticklabels=source_class_labels,
                )
                plt.xlabel("Predicted")
                plt.ylabel("True")
                plt.title(f"Source Domain ({args.data}) Confusion Matrix")

                # Create results directory if needed
                os.makedirs(args.result_path, exist_ok=True)
                source_viz_path = os.path.join(args.result_path, f"lmmd_{args.data}_source_cm.png")
                plt.savefig(source_viz_path)

                # Target domain visualization
                cm_target = confusion_matrix(y_target_true, y_target_pred)
                cm_target_norm = cm_target.astype("float") / cm_target.sum(axis=1)[:, np.newaxis]

                plt.figure(figsize=(10, 8))
                sns.heatmap(
                    cm_target_norm,
                    annot=True,
                    fmt=".2f",
                    cmap="Blues",
                    xticklabels=target_names,
                    yticklabels=target_names,
                )
                plt.xlabel("Predicted")
                plt.ylabel("True")
                plt.title(f"Target Domain ({target_dataset}) Confusion Matrix")

                target_viz_path = os.path.join(args.result_path, f"lmmd_{target_dataset}_target_cm.png")
                plt.savefig(target_viz_path)

                print(f"✅ Visualizations saved to: {args.result_path}")

            except ImportError:
                print("⚠️ Visualization requires matplotlib and seaborn. Install with 'pip install matplotlib seaborn'")

        # Domain adaptation visualizations
        print("\n📊 Generating domain adaptation visualizations...")

        # 1. Source domain confusion matrix
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            cm_source_norm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=source_class_labels,
            yticklabels=source_class_labels,
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Source Domain ({args.data}) Confusion Matrix")
        source_viz_path = os.path.join(args.result_path, f"lmmd_{args.data}_source_cm.png")
        plt.savefig(source_viz_path, bbox_inches='tight', dpi=300)
        plt.close()

        # 2. Target domain confusion matrix
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            cm_target_norm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=target_names,
            yticklabels=target_names,
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Target Domain ({target_dataset}) Confusion Matrix")
        target_viz_path = os.path.join(args.result_path, f"lmmd_{target_dataset}_target_cm.png")
        plt.savefig(target_viz_path, bbox_inches='tight', dpi=300)
        plt.close()

        # 3. Training history with smoothing (if available)
        if hasattr(ser_model, 'history') and ser_model.history is not None:
            print("   → Creating domain adaptation training history plots...")
            
            # Convert history to DataFrame
            history_df = pd.DataFrame(ser_model.history.history)
            
            # Apply rolling smoothing
            smoothing_window = 5
            smoothed_metrics = {}
            
            for metric in history_df.columns:
                if len(history_df[metric]) >= smoothing_window:
                    smoothed_metrics[f"{metric}_smooth"] = history_df[metric].rolling(
                        window=smoothing_window, center=True
                    ).mean()
                else:
                    smoothed_metrics[f"{metric}_smooth"] = history_df[metric]
            
            # Create subplots for different metrics
            fig, axes = plt.subplots(2, 1, figsize=(12, 12))
            
            # Plot accuracy metrics
            ax1 = axes[0]
            metrics_to_plot = [
                ('accuracy', 'Training Accuracy'),
                ('val_accuracy', 'Validation Accuracy'),
                ('target_accuracy', 'Target Domain Accuracy')
            ]
            
            colors = ['blue', 'red', 'green']
            for (metric, label), color in zip(metrics_to_plot, colors):
                if metric in history_df:
                    ax1.plot(history_df.index, history_df[metric], 
                            color=color, alpha=0.3, label=label)
                    ax1.plot(history_df.index, smoothed_metrics[f"{metric}_smooth"], 
                            color=color, label=f"{label} (Smoothed)")
            
            ax1.set_title('Model Accuracy over Epochs')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy')
            ax1.legend()
            ax1.grid(True)
            
            # Plot loss metrics
            ax2 = axes[1]
            loss_metrics = [
                ('loss', 'Training Loss'),
                ('val_loss', 'Validation Loss'),
                ('lmmd_loss', 'LMMD Loss')
            ]
            
            for (metric, label), color in zip(loss_metrics, colors):
                if metric in history_df:
                    ax2.plot(history_df.index, history_df[metric], 
                            color=color, alpha=0.3, label=label)
                    ax2.plot(history_df.index, smoothed_metrics[f"{metric}_smooth"], 
                            color=color, label=f"{label} (Smoothed)")
            
            ax2.set_title('Model Loss over Epochs')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.legend()
            ax2.grid(True)
            
            # Adjust layout and save
            plt.tight_layout()
            history_path = os.path.join(args.result_path, f"lmmd_training_history.png")
            plt.savefig(history_path, bbox_inches='tight', dpi=300)
            plt.close()
            
            print(f"✅ Domain adaptation visualizations saved to: {args.result_path}")
        else:
            print("   ⚠️ No training history available for visualization")

    # Final cleanup
    print("\n🧹 Final memory cleanup...")
    tf.keras.backend.clear_session()
    gc.collect()
    print("✅ Done!")


if __name__ == "__main__":
    main()
