#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=20
#SBATCH --time=06:00:00
#SBATCH --job-name=SER_job3
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

module load anaconda
eval "$(conda shell.bash hook)"
conda activate /home/FYP/muha0262/.conda/envs/ser_env_new

echo "Running with Python: $(which python)"
python --version

print_header() {
    echo -e "\n=== $1 ==="
    echo "Started at: $(date)"
}

run_command() {
    echo "Running: $1"
    eval "$1"
    if [ $? -ne 0 ]; then
        echo "Error: Command failed: $1"
        exit 1
    fi
}

print_header "Domain Adaptation EMODB (Optimized)"
print_header "Memory Status Before Training"
free -h

print_header "Top Processes by Memory"
top -b -o +%MEM | head -n 20

run_command "python -m src.training.main --mode train-lmmd \
    --data EMODB \
    --feature_type FUSION \
    --epoch 50 \
    --batch_size 8 \
    --lmmd_weight 0.3 \
    --split_fold 10"

print_header "Memory Status After Training"
free -h
