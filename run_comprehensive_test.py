#!/usr/bin/env python3
"""
Comprehensive Test Script for Speech Emotion Recognition

This script runs a complete evaluation of the SER system:
1. First runs the full data processing pipeline for both datasets
2. Then runs a series of experiments to test all combinations of:
   - Datasets (EMODB, RAVDESS)
   - Feature types (MFCC, LOGMEL, HUBERT, FUSION)
   - Training modes (standard, LMMD domain adaptation)
   - Testing modes (standard testing, cross-corpus validation)

It's designed to be run overnight as it may take several hours to complete all tests.
Results will be saved in organized folders with timestamps for easy comparison.
"""

import os
import sys
import subprocess
import time
import datetime
from itertools import product

def log(message):
    """Log a message with timestamp"""
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {message}")
    # Also save to log file
    with open("comprehensive_test_log.txt", "a") as f:
        f.write(f"[{timestamp}] {message}\n")

# Create a master results directory with timestamp
TIMESTAMP = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
MASTER_RESULTS_DIR = f"comprehensive_results_{TIMESTAMP}"
os.makedirs(MASTER_RESULTS_DIR, exist_ok=True)

# Configuration parameters
DATASETS = ["EMODB", "RAVDESS"]
FEATURE_TYPES = ["MFCC", "LOGMEL", "HUBERT", "FUSION"]
EPOCHS_TRAIN = 500  # Increased from 300 for more comprehensive testing
EPOCHS_LMMD = 700   # Increased from 400 for more thorough domain adaptation
BATCH_SIZE = 32
SPLIT_FOLD = 15     # Increased from 10 for more robust cross-validation
DROPOUT = 0.5       # Increased from 0.4 for stronger regularization
FILTER_SIZE = 256   # Increased from 128 for more model capacity
KERNEL_SIZE = 5     # Increased from 3 for wider convolution receptive field
STACK_SIZE = 4      # Increased from 3 for deeper networks
LR = 0.0001         # Reduced from 0.0003 for more precise convergence

# Standard project directories
MODEL_PATH = "./saved_models"
RESULT_PATH = "./results"
TEST_MODELS_PATH = "./test_models"

# Tracking variables
start_time = time.time()
completed_tests = 0
total_tests = 0

# Calculate total number of tests
# Data processing steps
total_tests += len(DATASETS)  # Data processing
# Model training and testing
total_tests += len(DATASETS) * len(FEATURE_TYPES)  # Standard training
total_tests += len(DATASETS) * len(FEATURE_TYPES)  # Standard testing
total_tests += len(DATASETS) * len(FEATURE_TYPES)  # Cross-corpus testing
total_tests += len(DATASETS) * len(FEATURE_TYPES)  # LMMD training

log(f"Starting comprehensive testing suite. Total tests to run: {total_tests}")
log(f"Results will be saved in standard project directories")
log(f"Logs and summary will be saved in: {MASTER_RESULTS_DIR}")

# Create directory for logs only
logs_dir = os.path.join(MASTER_RESULTS_DIR, "logs")
os.makedirs(logs_dir, exist_ok=True)

try:
    # 0. Data Processing Phase
    log("=== PHASE 0: DATA PROCESSING ===")
    
    # Check if features already exist
    def check_features_exist(dataset):
        """Check if dataset features already exist to avoid reprocessing"""
        mfcc_path = f"data/features/MFCC/{dataset}_MFCC_96/{dataset}.npy"
        logmel_path = f"data/features/LOGMEL/{dataset}_LOGMEL_128/{dataset}.npy"
        hubert_path = f"data/features/HUBERT/{dataset}_HUBERT/{dataset}.npy"
        
        if os.path.exists(mfcc_path) and os.path.exists(logmel_path) and os.path.exists(hubert_path):
            return True
        return False

    for dataset in DATASETS:
        if check_features_exist(dataset):
            log(f"Features for {dataset} already exist. Skipping data processing.")
            completed_tests += 1
            log(f"Progress: {completed_tests}/{total_tests} completed")
            continue
        
        log(f"Running full data processing pipeline for {dataset}...")
        
        # Check if raw data exists
        raw_data_path = os.path.join("data", "raw", dataset)
        if not os.path.exists(raw_data_path):
            log(f"ERROR: Raw data for {dataset} not found at {raw_data_path}")
            log(f"Skipping data processing for {dataset}")
            completed_tests += 1
            continue
        
        command = [
            "python", "-m", "src.data_processing.run_data_processing",
            "--dataset", dataset
        ]
        
        # Execute command with timeout
        log(f"COMMAND: {' '.join(command)}")
        try:
            # Set a timeout of 30 minutes per dataset processing
            result = subprocess.run(command, capture_output=True, text=True, timeout=1800)
            
            # Save outputs to log directory
            output_file = os.path.join(logs_dir, f"data_processing_{dataset}_output.txt")
            with open(output_file, "w") as f:
                f.write(result.stdout)
                if result.stderr:
                    f.write("\n\n=== ERRORS ===\n")
                    f.write(result.stderr)
            
            if result.returncode != 0:
                log(f"WARNING: Data processing for {dataset} may have encountered issues (exit code {result.returncode})")
        except subprocess.TimeoutExpired:
            log(f"ERROR: Data processing for {dataset} timed out after 30 minutes.")
        except Exception as e:
            log(f"ERROR during data processing for {dataset}: {str(e)}")
        
        completed_tests += 1
        log(f"Progress: {completed_tests}/{total_tests} completed")

    # 1. Standard Training Phase
    log("=== PHASE 1: STANDARD TRAINING ===")
    for dataset, feature_type in product(DATASETS, FEATURE_TYPES):
        log(f"Training standard model: {dataset} with {feature_type} features")
        
        command = [
            "python", "-m", "src.training.main",
            "--mode", "train",
            "--data", dataset,
            "--feature_type", feature_type,
            "--epoch", str(EPOCHS_TRAIN),
            "--batch_size", str(BATCH_SIZE),
            "--split_fold", str(SPLIT_FOLD),
            "--dropout", str(DROPOUT),
            "--filter_size", str(FILTER_SIZE),
            "--kernel_size", str(KERNEL_SIZE),
            "--stack_size", str(STACK_SIZE),
            "--lr", str(LR),
            "--model_path", MODEL_PATH,
            "--result_path", RESULT_PATH,
            "--visualize"
        ]
        
        # Execute command
        log(f"COMMAND: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True)
        
        # Save outputs to log directory
        output_file = os.path.join(logs_dir, f"train_{dataset}_{feature_type}_output.txt")
        with open(output_file, "w") as f:
            f.write(result.stdout)
            if result.stderr:
                f.write("\n\n=== ERRORS ===\n")
                f.write(result.stderr)
                
        completed_tests += 1
        log(f"Progress: {completed_tests}/{total_tests} completed")

    # 2. Standard Testing Phase
    log("=== PHASE 2: STANDARD TESTING ===")
    for dataset, feature_type in product(DATASETS, FEATURE_TYPES):
        log(f"Testing standard model: {dataset} with {feature_type} features")
        
        # Define paths
        test_path = os.path.join(TEST_MODELS_PATH, dataset, feature_type)
        
        command = [
            "python", "-m", "src.training.main",
            "--mode", "test",
            "--data", dataset,
            "--feature_type", feature_type,
            "--split_fold", str(SPLIT_FOLD),
            "--dropout", str(DROPOUT),
            "--filter_size", str(FILTER_SIZE),
            "--kernel_size", str(KERNEL_SIZE),
            "--stack_size", str(STACK_SIZE),
            "--lr", str(LR),
            "--test_path", test_path,
            "--result_path", RESULT_PATH,
            "--visualize"
        ]
        
        # Execute command
        log(f"COMMAND: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True)
        
        # Save outputs to log directory
        output_file = os.path.join(logs_dir, f"test_{dataset}_{feature_type}_output.txt")
        with open(output_file, "w") as f:
            f.write(result.stdout)
            if result.stderr:
                f.write("\n\n=== ERRORS ===\n")
                f.write(result.stderr)
                
        completed_tests += 1
        log(f"Progress: {completed_tests}/{total_tests} completed")

    # 3. Cross-Corpus Testing Phase
    log("=== PHASE 3: CROSS-CORPUS TESTING ===")
    for dataset, feature_type in product(DATASETS, FEATURE_TYPES):
        log(f"Cross-corpus testing: {dataset} with {feature_type} features")
        
        # Define paths
        test_path = os.path.join(TEST_MODELS_PATH, dataset, feature_type)
        
        command = [
            "python", "-m", "src.training.main",
            "--mode", "test-cross-corpus",
            "--data", dataset,
            "--feature_type", feature_type,
            "--split_fold", str(SPLIT_FOLD),
            "--dropout", str(DROPOUT),
            "--filter_size", str(FILTER_SIZE),
            "--kernel_size", str(KERNEL_SIZE),
            "--stack_size", str(STACK_SIZE),
            "--lr", str(LR),
            "--test_path", test_path,
            "--result_path", RESULT_PATH,
            "--visualize"
        ]
        
        # Execute command
        log(f"COMMAND: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True)
        
        # Save outputs to log directory
        output_file = os.path.join(logs_dir, f"cross_corpus_{dataset}_{feature_type}_output.txt")
        with open(output_file, "w") as f:
            f.write(result.stdout)
            if result.stderr:
                f.write("\n\n=== ERRORS ===\n")
                f.write(result.stderr)
                
        completed_tests += 1
        log(f"Progress: {completed_tests}/{total_tests} completed")

    # 4. LMMD Domain Adaptation Training Phase
    log("=== PHASE 4: LMMD DOMAIN ADAPTATION ===")
    for dataset, feature_type in product(DATASETS, FEATURE_TYPES):
        log(f"LMMD training: {dataset} with {feature_type} features")
        
        # For LMMD, the target dataset is the opposite of the source
        target_dataset = "RAVDESS" if dataset == "EMODB" else "EMODB"
        
        command = [
            "python", "-m", "src.training.main",
            "--mode", "train-lmmd",
            "--data", dataset,
            "--target_data", target_dataset,
            "--feature_type", feature_type,
            "--epoch", str(EPOCHS_LMMD),
            "--batch_size", str(BATCH_SIZE),
            "--split_fold", str(SPLIT_FOLD),
            "--dropout", str(DROPOUT),
            "--filter_size", str(FILTER_SIZE),
            "--kernel_size", str(KERNEL_SIZE),
            "--stack_size", str(STACK_SIZE),
            "--lr", str(LR),
            "--model_path", MODEL_PATH,
            "--result_path", RESULT_PATH,
            "--visualize",
            "--use_lmmd"
        ]
        
        # Execute command
        log(f"COMMAND: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True)
        
        # Save outputs to log directory
        output_file = os.path.join(logs_dir, f"lmmd_{dataset}_to_{target_dataset}_{feature_type}_output.txt")
        with open(output_file, "w") as f:
            f.write(result.stdout)
            if result.stderr:
                f.write("\n\n=== ERRORS ===\n")
                f.write(result.stderr)
                
        completed_tests += 1
        log(f"Progress: {completed_tests}/{total_tests} completed")

    # Calculate total execution time
    end_time = time.time()
    total_time = end_time - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    log(f"All tests completed successfully!")
    log(f"Total execution time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    log(f"Results saved in: {MASTER_RESULTS_DIR}")
    
    # Generate summary report
    summary_path = os.path.join(MASTER_RESULTS_DIR, "summary_report.txt")
    with open(summary_path, "w") as f:
        f.write("=== SPEECH EMOTION RECOGNITION COMPREHENSIVE TEST REPORT ===\n\n")
        f.write(f"Test Date: {TIMESTAMP}\n")
        f.write(f"Total Tests: {total_tests}\n")
        f.write(f"Total Time: {int(hours)}h {int(minutes)}m {int(seconds)}s\n\n")
        
        f.write("Test Configurations:\n")
        f.write(f"- Datasets: {', '.join(DATASETS)}\n")
        f.write(f"- Feature Types: {', '.join(FEATURE_TYPES)}\n")
        f.write(f"- Standard Training Epochs: {EPOCHS_TRAIN}\n")
        f.write(f"- LMMD Training Epochs: {EPOCHS_LMMD}\n")
        f.write(f"- Batch Size: {BATCH_SIZE}\n")
        f.write(f"- Cross-Validation Folds: {SPLIT_FOLD}\n\n")
        
        f.write("Test Phases:\n")
        f.write("1. Standard Training\n")
        f.write("2. Standard Testing\n")
        f.write("3. Cross-Corpus Testing\n")
        f.write("4. LMMD Domain Adaptation\n\n")
        
        f.write("For detailed results, see the individual test output files in each directory.\n")
    
    log(f"Summary report written to {summary_path}")

except Exception as e:
    log(f"ERROR: Test suite encountered an error: {str(e)}")
    import traceback
    log(traceback.format_exc())
    sys.exit(1) 