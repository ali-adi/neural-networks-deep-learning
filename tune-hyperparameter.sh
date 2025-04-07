#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=20
#SBATCH --time=06:00:00
#SBATCH --job-name=SER_full_pipeline
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

# Load environment
module load anaconda
source activate /tc1apps/2_conda_env/Jupyter

# BEGIN ACTUAL SCRIPT
# ----------------------------------------
# Function to print section headers
print_header() {
    echo -e "\n=== $1 ==="
    echo "Started at: $(date)"
}

# Function to run a command and check its status
run_command() {
    echo "Running: $1"
    eval "$1"
    if [ $? -ne 0 ]; then
        echo "Error: Command failed: $1"
        exit 1
    fi
}

# Create necessary directories
mkdir -p test_models/EMODB/{MFCC,LOGMEL,HUBERT,FUSION}
mkdir -p test_models/RAVDESS/{MFCC,LOGMEL,HUBERT,FUSION}

# Feature Extraction
print_header "Starting Feature Extraction"
run_command "python -m src.data_processing.extract_feature --dataset EMODB"
run_command "python -m src.data_processing.extract_feature --dataset RAVDESS"

# Define parameter ranges - Reduced set
FEATURES=("MFCC" "LOGMEL" "HUBERT" "FUSION")
DATASETS=("EMODB" "RAVDESS")
EPOCHS=(100 200)
BATCH_SIZES=(32 64)
LEARNING_RATES=(0.0001 0.0003 0.001)
BETA1_VALUES=(0.9 0.93)
BETA2_VALUES=(0.98 0.99)
DROPOUT_RATES=(0.3 0.4)
ACTIVATIONS=("relu" "elu")
FILTER_SIZES=(64 128)
DILATION_SIZES=(4 8)
KERNEL_SIZES=(3 5)
STACK_SIZES=(2 3)
SPLIT_FOLDS=(10 15)
LMMD_WEIGHTS=(0.1 0.3 0.5 0.7 0.9)

# Regular Training
print_header "Starting Regular Training with Parameter Tuning"
for dataset in "${DATASETS[@]}"; do
    for feature in "${FEATURES[@]}"; do
        for epoch in "${EPOCHS[@]}"; do
            for batch_size in "${BATCH_SIZES[@]}"; do
                for lr in "${LEARNING_RATES[@]}"; do
                    for beta1 in "${BETA1_VALUES[@]}"; do
                        for beta2 in "${BETA2_VALUES[@]}"; do
                            for dropout in "${DROPOUT_RATES[@]}"; do
                                for activation in "${ACTIVATIONS[@]}"; do
                                    for filter_size in "${FILTER_SIZES[@]}"; do
                                        for dilation_size in "${DILATION_SIZES[@]}"; do
                                            for kernel_size in "${KERNEL_SIZES[@]}"; do
                                                for stack_size in "${STACK_SIZES[@]}"; do
                                                    for split_fold in "${SPLIT_FOLDS[@]}"; do
                                                        print_header "Training $dataset with $feature"
                                                        run_command "python -m src.training.main --mode train \
                                                            --data $dataset \
                                                            --feature_type $feature \
                                                            --epoch $epoch \
                                                            --batch_size $batch_size \
                                                            --lr $lr \
                                                            --beta1 $beta1 \
                                                            --beta2 $beta2 \
                                                            --dropout $dropout \
                                                            --activation $activation \
                                                            --filter_size $filter_size \
                                                            --dilation_size $dilation_size \
                                                            --kernel_size $kernel_size \
                                                            --stack_size $stack_size \
                                                            --split_fold $split_fold"
                                                    done
                                                done
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

# Domain Adaptation Training
print_header "Starting Domain Adaptation Training with Parameter Tuning"
for dataset in "${DATASETS[@]}"; do
    for feature in "${FEATURES[@]}"; do
        for epoch in "${EPOCHS[@]}"; do
            for batch_size in "${BATCH_SIZES[@]}"; do
                for lr in "${LEARNING_RATES[@]}"; do
                    for beta1 in "${BETA1_VALUES[@]}"; do
                        for beta2 in "${BETA2_VALUES[@]}"; do
                            for dropout in "${DROPOUT_RATES[@]}"; do
                                for activation in "${ACTIVATIONS[@]}"; do
                                    for filter_size in "${FILTER_SIZES[@]}"; do
                                        for dilation_size in "${DILATION_SIZES[@]}"; do
                                            for kernel_size in "${KERNEL_SIZES[@]}"; do
                                                for stack_size in "${STACK_SIZES[@]}"; do
                                                    for split_fold in "${SPLIT_FOLDS[@]}"; do
                                                        for lmmd_weight in "${LMMD_WEIGHTS[@]}"; do
                                                            print_header "DA: $dataset, $feature, lmmd_weight=$lmmd_weight"
                                                            run_command "python -m src.training.main --mode train-lmmd \
                                                                --data $dataset \
                                                                --feature_type $feature \
                                                                --epoch $epoch \
                                                                --batch_size $batch_size \
                                                                --lr $lr \
                                                                --beta1 $beta1 \
                                                                --beta2 $beta2 \
                                                                --dropout $dropout \
                                                                --activation $activation \
                                                                --filter_size $filter_size \
                                                                --dilation_size $dilation_size \
                                                                --kernel_size $kernel_size \
                                                                --stack_size $stack_size \
                                                                --split_fold $split_fold \
                                                                --lmmd_weight $lmmd_weight \
                                                                --visualize"
                                                        done
                                                    done
                                                done
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

# Same-Corpus Testing
print_header "Starting Same-Corpus Testing"
for dataset in "${DATASETS[@]}"; do
    for feature in "${FEATURES[@]}"; do
        run_command "python -m src.training.main --mode test --data $dataset --feature_type $feature --test_path ./test_models/$dataset/$feature"
    done
done

# Cross-Corpus Testing
print_header "Starting Cross-Corpus Testing"
for feature in "${FEATURES[@]}"; do
    run_command "python -m src.training.main --mode test-cross-corpus --data EMODB --feature_type $feature --test_path ./test_models/EMODB/$feature/ --visualize"
    run_command "python -m src.training.main --mode test-cross-corpus --data RAVDESS --feature_type $feature --test_path ./test_models/RAVDESS/$feature/ --visualize"
done

print_header "All experiments completed"
echo "Finished at: $(date)"

# ----------------------------------------
# END OF SCRIPT
