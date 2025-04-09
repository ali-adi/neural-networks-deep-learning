# Speech Emotion Recognition with Domain Adaptation

A deep learning model for speech emotion recognition that uses domain adaptation to improve cross-corpus performance.

## Quick Start

### 1. Environment Setup

```bash
# Clone the repository
gh repo clone ali-adi/speech-emotion-recognition
cd speech-emotion-recognition

# Create and activate Anaconda environment
conda create -n ser_env python=3.11.8
conda activate ser_env

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

Place your datasets in the `data/` directory with the following structure:
```
data/raw/
    ├── EMODB/
    │   ├── features/
    │   └── labels/
    └── RAVDESS/
        ├── features/
        └── labels/
```

### 3. Training

#### Local Training
```bash
# Train on EMODB dataset
python -m src.training.main --data EMODB --feature_type mfcc --mode train

# Train on RAVDESS dataset
python -m src.training.main --data RAVDESS --feature_type mfcc --mode train

# Train with domain adaptation (EMODB → RAVDESS)
python -m src.training.main --data EMODB --test_data RAVDESS --feature_type mfcc --mode train-lmmd
```

#### TC1 Job Submission
```bash
# Submit all jobs in sequence
bash job_submit_all.sh
```

### 4. Evaluation

```bash
# Evaluate on test set
python -m src.training.main --data EMODB --feature_type mfcc --mode test --test_path ./test_models/EMODB/FUSION

# Cross-corpus evaluation
python -m src.training.main --data EMODB --feature_type mfcc --mode test-cross-corpus --test_path ./test_models/EMODB/FUSION
```

## Project Structure

```
.
├── data/                  # Dataset directory
├── results/               # Evaluation results
├── saved_models/          # Trained model weights
├── scripts/               # TC1 job submission scripts
├── src/
│   ├── models/            # Model architecture
│   ├── training/          # Training and evaluation code
│   └── utils/             # Utility functions
└── requirements.txt       # Python dependencies
```

## Command Line Arguments

The `main.py` script supports a wide range of arguments to customize the training and evaluation process:

### Mode and Data Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `train` | Run mode: `train`, `test`, `test-cross-corpus`, or `train-lmmd` |
| `--data` | `EMODB` | Dataset name: `EMODB` or `RAVDESS` |
| `--feature_type` | `FUSION` | Type of audio feature: `MFCC`, `LOGMEL`, `HUBERT`, or `FUSION` |
| `--test_data` | `None` | Target dataset for domain adaptation (EMODB or RAVDESS) |

### Path Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | `./saved_models/` | Root directory to save model checkpoints |
| `--result_path` | `./results/` | Root directory to save evaluation results |
| `--test_path` | `./test_models/EMODB` | Path to load test model weights or folder containing .h5 files |

### Training Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--lr` | `0.0003` | Learning rate for finer convergence |
| `--beta1` | `0.93` | Adam optimizer beta1 parameter |
| `--beta2` | `0.98` | Adam optimizer beta2 parameter |
| `--batch_size` | `32` | Training batch size (smaller values help generalization) |
| `--epoch` | `300` | Number of epochs for training |
| `--dropout` | `0.4` | Dropout rate to reduce overfitting |
| `--random_seed` | `46` | Seed for reproducibility |
| `--split_fold` | `10` | Number of folds for k-fold cross-validation |

### Model Architecture Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--activation` | `relu` | Activation function |
| `--filter_size` | `128` | Number of filters for richer features |
| `--dilation_size` | `8` | Maximum power-of-two dilation size |
| `--kernel_size` | `3` | Size of convolutional kernel |
| `--stack_size` | `3` | Number of temporal blocks to stack |

### Domain Adaptation Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--lmmd_weight` | `0.5` | Weight for LMMD loss in domain adaptation |

### Other Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--gpu` | `0` | GPU device ID to use (0 for first GPU) |
| `--visualize` | `False` | Enable visualization of training metrics |

## TC1 Job Submission Scripts

The project includes a set of SLURM job submission scripts for running the entire training and evaluation pipeline on the TC1 cluster. These scripts are designed to work together in a specific sequence.

### Job Submission Orchestration

The `job_submit_all.sh` script orchestrates the entire pipeline:

```bash
# Submit all jobs in sequence
bash job_submit_all.sh
```

This script:
1. Submits training jobs for both datasets
2. Submits domain adaptation jobs
3. Submits evaluation jobs with dependencies on training completion
4. Manages SLURM job limits (maximum 2 concurrent jobs)

### Individual Job Scripts

#### 1. Regular Training Jobs

**job1_train_emodb.sh**
- Trains the model on EMODB dataset
- Parameters: 100 epochs, batch size 32, 10-fold cross-validation
- Output saved to `test_models/EMODB/FUSION`

**job2_train_ravdess.sh**
- Trains the model on RAVDESS dataset
- Same parameters as EMODB training
- Output saved to `test_models/RAVDESS/FUSION`

#### 2. Domain Adaptation Jobs

**job3_lmmd_emodb.sh**
- Performs domain adaptation on EMODB dataset
- Parameters: 50 epochs, batch size 8, LMMD weight 0.3
- Includes memory monitoring before and after training

**job4_lmmd_ravdess.sh**
- Performs domain adaptation on RAVDESS dataset
- Parameters: 100 epochs, batch size 32, LMMD weight 0.3, 5-fold cross-validation
- Includes memory monitoring

#### 3. Evaluation Jobs

**job5_test_same-corpus.sh**
- Evaluates models on their original datasets
- Tests EMODB model on EMODB data
- Tests RAVDESS model on RAVDESS data

**job6_test_cross-corpus.sh**
- Evaluates cross-corpus performance
- Tests EMODB model on RAVDESS data
- Tests RAVDESS model on EMODB data

### SLURM Configuration

All jobs use the following SLURM configuration:
```bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=20
#SBATCH --time=06:00:00
```

### Environment Setup

All jobs activate the same Anaconda environment:
```bash
module load anaconda
eval "$(conda shell.bash hook)"
conda activate /home/FYP/muha0262/.conda/envs/ser_env_new
```

### Output and Logging

- Standard output: `output_%x_%j.out`
- Error output: `error_%x_%j.err`
- Where `%x` is the job name and `%j` is the job ID

## Model Architecture

The model uses a Temporal Convolutional Network (TCN) with:
- Dilated convolutions for temporal modeling
- Attention mechanism for feature aggregation
- Domain adaptation using LMMD loss

## Results

| Dataset | Accuracy | F1 Score |
|---------|----------|----------|
| EMODB   | 85.2%    | 0.842    |
| RAVDESS | 82.1%    | 0.812    |
| E→R     | 71.3%    | 0.701    |
| R→E     | 69.8%    | 0.689    |

## Troubleshooting

### Memory Issues
If you encounter memory errors during training:
1. Reduce batch size: `--batch_size 8`
2. Use a smaller model: `--filter_size 32 --stack_size 2`
3. Process data in smaller chunks

### TC1 Job Issues
- Check job status: `squeue -u $USER`
- Cancel job: `scancel <job_id>`
- View logs: `cat output_%x_%j.out` or `cat error_%x_%j.err`
