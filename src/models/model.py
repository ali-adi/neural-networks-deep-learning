# model.py
"""
model.py

DESCRIPTION:
------------
Defines the core architecture and training logic for the Speech Emotion Recognition (SER) model
using a combination of Discriminant Temporal Pyramid Matching (DTPM) and an attention-based
feature aggregation mechanism.

This model is designed to extract temporal and discriminative patterns from speech signals,
enhanced with bidirectional temporal modeling and label smoothing for better generalization.

WHAT THIS SCRIPT DOES:
-----------------------
1. Builds a deep temporal convolutional network with dilated convolutions (TCN-style)
2. Incorporates bidirectional streams (forward and reversed input)
3. Applies gated residual connections for better feature retention
4. Aggregates features using a learnable attention layer
5. Supports training using k-fold cross-validation
6. Saves performance metrics (confusion matrix + classification report) as Excel files

MODEL COMPONENTS:
------------------
- `TemporalConvNet`   : Stacked temporal blocks with increasing dilation for long-range temporal modeling
- `WeightLayer`       : Learnable attention weights for weighted feature aggregation
- `smooth_labels()`   : Applies label smoothing to enhance generalization
- `SpeechEmotionModel`: High-level model wrapper that handles training, evaluation, and I/O

KEY FEATURES:
-------------
- Uses `categorical_crossentropy` for multi-class classification
- Implements gated skip connections (similar to TCN/ResNet)
- Output metrics include accuracy, confusion matrix, and classification report
- Saves best-performing fold weights in `test_models/` for later evaluation

HOW TO USE THIS FILE:
----------------------
This script is not meant to be executed directly.
It is imported and used by `main.py` during both training and testing:

    from src.models.model import SpeechEmotionModel

SUPPORTED INPUT SHAPES:
------------------------
- Input: 3D tensors shaped as (time_steps, feature_dim)
  e.g., (300, 96) for MFCC, or (300, 768) for HuBERT

NOTES:
------
- Works with both standard features (MFCC, Log-Mel) and self-supervised embeddings (HuBERT)
- Automatically handles varying sequence lengths using `pad_sequences`
- Can be extended with custom loss functions like LMMD for domain adaptation (future scope)
"""


# Suppress unnecessary warnings for cleaner output
import os
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import copy
import datetime
import shutil

# Add visualization imports
import matplotlib.pyplot as plt

# Standard libraries
import numpy as np
import pandas as pd
import gc  # Add garbage collector

# TensorFlow imports
import tensorflow as tf
import tensorflow.keras.backend as K

# Evaluation tools
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve
from sklearn.model_selection import KFold
from tensorflow.keras import callbacks
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.layers import Activation, BatchNormalization, Conv1D, Dense, GlobalAveragePooling1D, Input, Lambda, SpatialDropout1D, add, Dropout
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


# ------------------------
# TEMPORAL BLOCK DEFINITION
# ------------------------
def temporal_block(inputs, dilation, activation, nb_filters, kernel_size, dropout_rate=0.0):
    skip_connection = inputs
    x = Conv1D(
        filters=nb_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation,
        padding="causal",
    )(inputs)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = SpatialDropout1D(dropout_rate)(x)
    x = Conv1D(
        filters=nb_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation,
        padding="causal",
    )(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = SpatialDropout1D(dropout_rate)(x)
    if skip_connection.shape[-1] != x.shape[-1]:
        skip_connection = Conv1D(filters=nb_filters, kernel_size=1, padding="same")(skip_connection)
    gated = Lambda(sigmoid)(x)
    return skip_connection * gated


# ------------------------
# DILATED TEMPORAL NETWORK
# ------------------------
class TemporalConvNet:
    def __init__(
        self,
        nb_filters=64,
        kernel_size=2,
        nb_stacks=1,
        dilations=8,
        activation="relu",
        dropout_rate=0.1,
        name="TemporalConvNet",
    ):
        self.name = name
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.nb_stacks = nb_stacks
        self.dilations = dilations if isinstance(dilations, int) else 8
        self.activation = activation
        self.dropout_rate = dropout_rate

    def __call__(self, inputs):
        forward = inputs
        backward = Lambda(lambda x: tf.reverse(x, axis=[1]))(inputs)
        f = Conv1D(self.nb_filters, kernel_size=1, padding="causal")(forward)
        b = Conv1D(self.nb_filters, kernel_size=1, padding="causal")(backward)
        final_skips = []
        for _ in range(self.nb_stacks):
            for dilation_rate in [2**i for i in range(self.dilations)]:
                f = temporal_block(
                    f,
                    dilation_rate,
                    self.activation,
                    self.nb_filters,
                    self.kernel_size,
                    self.dropout_rate,
                )
                b = temporal_block(
                    b,
                    dilation_rate,
                    self.activation,
                    self.nb_filters,
                    self.kernel_size,
                    self.dropout_rate,
                )
                merged = add([f, b])
                pooled = GlobalAveragePooling1D()(merged)
                expanded = Lambda(lambda x: tf.expand_dims(x, axis=1))(pooled)
                final_skips.append(expanded)
        return Lambda(lambda x: tf.concat(x, axis=1))(final_skips)


class WeightLayer(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.kernel = self.add_weight(
            name="kernel",
            shape=[input_shape[1], 1],
            initializer="uniform",
            trainable=True,
        )

    def call(self, inputs):
        x = tf.transpose(inputs, [0, 2, 1])
        x = tf.matmul(x, self.kernel)
        return tf.squeeze(x, axis=-1)


def smooth_labels(labels, factor=0.1):
    # If not one-hot encoded, convert using to_categorical
    if len(labels.shape) == 1:
        labels = to_categorical(labels, num_classes=np.max(labels) + 1)

    labels = labels.astype(np.float32)
    labels *= 1 - factor
    labels += factor / labels.shape[1]
    return labels


class SpeechEmotionModel:
    def __init__(self, input_shape, class_labels, args):
        """Initialize the Speech Emotion Recognition model

        Args:
            input_shape: Shape of input data (time_steps, features)
            class_labels: List of emotion class names
            args: Arguments from argparse
        """
        self.class_labels = class_labels
        self.input_shape = input_shape
        self.args = args
        self.model = None
        self.trained = False
        self.best_fold_acc = 0
        self.best_fold_weight_path = None
        self.now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # Set up directories following project structure
        feature_type = args.feature_type.upper()
        
        # Base directories
        self.base_saved_models = os.path.join("saved_models", feature_type)
        self.base_test_models = os.path.join("test_models", args.data, feature_type)
        self.base_results = os.path.join("results", feature_type)
        
        if args.mode.startswith('test'):
            # For testing, use test_models directory
            self.model_dir = self.base_test_models
        else:
            # For training, create a unique session folder under saved_models
            session_name = f"{args.data}_{feature_type}_{self.now}"
            self.model_dir = os.path.join(self.base_saved_models, session_name)
            
        # Results directory is always under results/FEATURE_TYPE
        self.result_dir = self.base_results
        
        # Create all necessary directories
        os.makedirs(self.base_saved_models, exist_ok=True)
        os.makedirs(self.base_test_models, exist_ok=True)
        os.makedirs(self.base_results, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        
        print("\n📁 Directory Structure:")
        print(f"   Model directory: {self.model_dir}")
        print(f"   Results directory: {self.result_dir}")
        
        # Initialize tracking variables
        self.fold_histories = []
        self.training_metrics = {"acc": [], "val_acc": [], "loss": [], "val_loss": []}
        self.num_classes = len(class_labels)
        self.use_lmmd = args.mode == "train-lmmd"
        self.matrix = []  # Initialize matrix list for confusion matrices
        
        if self.use_lmmd:
            try:
                from .lmmd_loss import LMMDLoss
                self.lmmd_loss = LMMDLoss(weight=args.lmmd_weight, num_classes=self.num_classes)
            except ImportError:
                print("⚠️ LMMD loss module not found. Domain adaptation will not be performed.")
                self.use_lmmd = False

    def _create_model_architecture(self, tcn):
        """Create the model architecture without compilation"""
        # Input layer
        inputs = Input(shape=self.input_shape)

        # Handle different types of input shapes
        if len(self.input_shape) == 1:  # 1D input (like plain HUBERT features)
            x = Dense(self.args.filter_size, activation=self.args.activation)(inputs)
            x = Dense(self.args.filter_size, activation=self.args.activation)(x)
            outputs = Dense(self.num_classes, activation="softmax")(x)
        else:  # 2D input (like MFCC or LogMel spectrograms)
            conv_net = tcn
            conv_output = conv_net(inputs)
            attention_out = WeightLayer()(conv_output)
            outputs = Dense(self.num_classes, activation="softmax")(attention_out)

        return KerasModel(inputs=inputs, outputs=outputs)

    def create_model(self):
        """Create the model architecture"""
        # Input layer
        inputs = Input(shape=self.input_shape)

        # Temporal convolutional network
        tcn = TemporalConvNet(
            nb_filters=self.args.filter_size,
            kernel_size=self.args.kernel_size,
            nb_stacks=self.args.stack_size,
            dilations=self.args.dilation_size,
            activation=self.args.activation,
            dropout_rate=self.args.dropout
        )(inputs)

        # Attention mechanism
        attention_out = WeightLayer()(tcn)
        
        # Feature extraction layers with smaller dimensions
        x = Dense(128, activation="relu", name="feature_layer_1")(attention_out)
        x = Dropout(0.4)(x)
        x = Dense(64, activation="relu", name="feature_layer_2")(x)
        x = Dropout(0.4)(x)
        
        # Classification head
        outputs = Dense(self.num_classes, activation="softmax")(x)

        # Create model
        self.model = KerasModel(inputs=inputs, outputs=outputs)
        self._compile_model()
        
        # Store feature extraction model
        self.feature_model = KerasModel(
            inputs=self.model.input,
            outputs=self.model.get_layer("feature_layer_2").output,
            name="feature_extractor"
        )
        
        return self.model

    def _compile_model(self, loss=None):
        """Helper method to compile the model with consistent settings"""
        optimizer = Adam(
            learning_rate=self.args.lr,
            beta_1=self.args.beta1,
            beta_2=self.args.beta2,
            epsilon=1e-8,
        )
        
        # Use provided loss or default to categorical_crossentropy
        if loss is None:
            loss = "categorical_crossentropy"
            
        self.model.compile(
            loss=loss,
            optimizer=optimizer,
            metrics=["accuracy"]
        )

    def _setup_training(self, x, save_dir):
        """Helper method to setup training environment"""
        # Initialize k-fold cross validation
        kfold = KFold(n_splits=self.args.split_fold, shuffle=True, random_state=self.args.random_seed)
        
        # Store metrics for each fold
        fold_metrics = {
            'acc': [], 'val_acc': [], 'loss': [], 'val_loss': []
        }
        
        # Create directory for saving models
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"🔍 Using input shape for model: {self.input_shape}")
        print(f"📁 Saving models to: {save_dir}")
        
        return kfold, fold_metrics

    def _train_loop(self, x, y, is_domain_adaptation=False, target_data=None):
        """Common training loop for both regular and domain adaptation training"""
        # Setup training environment
        kfold = KFold(n_splits=self.args.split_fold, shuffle=True, random_state=self.args.random_seed)
        fold_metrics = {'acc': [], 'val_acc': [], 'loss': [], 'val_loss': []}
        
        # If doing domain adaptation, adjust target data size to match source data
        if is_domain_adaptation and target_data is not None:
            rng = np.random.RandomState(self.args.random_seed)
            if len(target_data) > len(x):
                # Subsample target data if it's larger
                target_indices = rng.choice(len(target_data), size=len(x), replace=False)
                target_data = target_data[target_indices]
            elif len(target_data) < len(x):
                # Oversample target data if it's smaller
                target_indices = rng.choice(len(target_data), size=len(x), replace=True)
                target_data = target_data[target_indices]
        
        print(f"🎯 Starting {'domain adaptation' if is_domain_adaptation else 'k-fold'} training...")
        print(f"🔍 Using input shape for model: {self.input_shape}")
        print(f"📁 Saving models to: {self.model_dir}")
        
        # K-fold cross validation
        for fold, (train_idx, val_idx) in enumerate(kfold.split(x), 1):
            fold_metrics = self._train_fold(
                fold, train_idx, val_idx, x, y, 
                self.model_dir, fold_metrics,
                is_domain_adaptation=is_domain_adaptation,
                target_data=target_data
            )
        
        # Plot combined metrics across all folds
        self._plot_combined_training_metrics(fold_metrics, self.model_dir)
        
        return self.best_fold_weight_path

    def _train_fold(self, fold, train_idx, val_idx, x, y, save_dir, fold_metrics, is_domain_adaptation=False, target_data=None):
        """Train a single fold of the model"""
        print(f"🔁 Fold {fold}/{self.args.split_fold}")
        
        # Split source data
        x_train, x_val = x[train_idx], x[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Split target data if doing domain adaptation
        if is_domain_adaptation and target_data is not None:
            target_train = target_data[train_idx]
            target_val = target_data[val_idx]
            print(f"Source train shape: {x_train.shape}")
            print(f"Target train shape: {target_train.shape}")
        
        # Create model for this fold
        self.create_model()
        
        # Setup callbacks with proper checkpoint path
        checkpoint_base = os.path.join(save_dir, f"{self.args.split_fold}-fold_weights_best_{fold}")
        callbacks_list = [
            tf.keras.callbacks.ModelCheckpoint(
                checkpoint_base,  # TensorFlow will automatically add .index and .data-* extensions
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=True,
                mode='max',
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,  # Increased patience to allow more epochs
                restore_best_weights=True,
                verbose=1  # Added verbose to see when early stopping triggers
            )
        ]
        
        # Train the model
        if is_domain_adaptation and target_data is not None:
            # Custom training loop for domain adaptation
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.args.lr,
                beta_1=self.args.beta1,
                beta_2=self.args.beta2
            )
            
            # For LMMD training, we'll store history manually
            history_dict = {
                'accuracy': [],
                'val_accuracy': [],
                'loss': [],
                'val_loss': [],
                'class_loss': [],
                'adapt_loss': []
            }
            
            # Training loop
            best_val_acc = 0
            patience_counter = 0
            
            for epoch in range(self.args.epoch):
                print(f"Epoch {epoch + 1}/{self.args.epoch}")
                
                # Shuffle training data
                indices = np.random.permutation(len(x_train))
                x_train_shuffled = x_train[indices]
                y_train_shuffled = y_train[indices]
                target_train_shuffled = target_train[indices]  # Use same indices for target
                
                # Mini-batch training
                progbar = tf.keras.utils.Progbar(len(x_train))
                total_loss = 0
                total_class_loss = 0
                total_adapt_loss = 0
                
                # Process in smaller internal batches to save memory
                actual_batch_size = min(self.args.batch_size, 16)  # Cap maximum batch size for memory efficiency
                print(f"Using batch size of {actual_batch_size} for memory efficiency")
                
                for i in range(0, len(x_train), actual_batch_size):
                    batch_x = x_train_shuffled[i:i + actual_batch_size]
                    batch_y = y_train_shuffled[i:i + actual_batch_size]
                    batch_target = target_train_shuffled[i:i + actual_batch_size]
                    
                    with tf.GradientTape() as tape:
                        # Forward pass
                        source_pred = self.model(batch_x, training=True)
                        target_pred = self.model(batch_target, training=True)
                        
                        # Extract features using feature model
                        source_features = self.feature_model(batch_x, training=True)
                        target_features = self.feature_model(batch_target, training=True)
                        
                        # Calculate losses
                        class_loss = tf.keras.losses.categorical_crossentropy(batch_y, source_pred)
                        class_loss = tf.reduce_mean(class_loss)
                        
                        # Get pseudo labels for target domain
                        target_pseudo_labels = tf.nn.softmax(target_pred)
                        
                        # Calculate LMMD loss
                        adapt_loss = self.lmmd_loss.call(
                            source_features=source_features,
                            target_features=target_features,
                            source_labels=batch_y,
                            target_pseudo_labels=target_pseudo_labels,
                            num_classes=self.num_classes
                        )
                        
                        # Total loss
                        batch_loss = class_loss + (self.args.lmmd_weight * adapt_loss)
                    
                    # Compute gradients and update weights
                    gradients = tape.gradient(batch_loss, self.model.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                    
                    # Update metrics
                    total_loss += batch_loss
                    total_class_loss += class_loss
                    total_adapt_loss += adapt_loss
                    
                    # Update progress bar
                    progbar.add(len(batch_x), values=[
                        ('total_loss', batch_loss),
                        ('class_loss', class_loss),
                        ('adapt_loss', adapt_loss)
                    ])
                
                # Calculate average losses for the epoch
                avg_loss = total_loss / (len(x_train) / self.args.batch_size)
                avg_class_loss = total_class_loss / (len(x_train) / self.args.batch_size)
                avg_adapt_loss = total_adapt_loss / (len(x_train) / self.args.batch_size)
                
                # Evaluate on validation set
                val_loss, val_acc = self.model.evaluate(x_val, y_val, verbose=0)
                print(f"\nEpoch {epoch + 1} Summary:")
                print(f"Average Loss: {avg_loss:.4f}")
                print(f"Classification Loss: {avg_class_loss:.4f}")
                print(f"Adaptation Loss: {avg_adapt_loss:.4f}")
                print(f"Validation Accuracy: {val_acc:.4f}")
                
                # Store metrics in history dictionary
                train_loss = float(avg_loss)
                train_acc = float(self.model.evaluate(x_train, y_train, verbose=0)[1])
                
                history_dict['accuracy'].append(train_acc)
                history_dict['val_accuracy'].append(val_acc)
                history_dict['loss'].append(train_loss)
                history_dict['val_loss'].append(val_loss)
                history_dict['class_loss'].append(float(avg_class_loss))
                history_dict['adapt_loss'].append(float(avg_adapt_loss))
                
                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self.model.save_weights(checkpoint_base)
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= 10:  # Early stopping
                        print("Early stopping triggered")
                        break
            
            # Use manually collected history for LMMD
            history = history_dict
        else:
            # Regular training
            # Cap batch size for memory efficiency
            actual_batch_size = min(self.args.batch_size, 16)
            print(f"Using batch size of {actual_batch_size} for memory efficiency")
            
            history = self.model.fit(
                x_train,
                y_train,
                batch_size=actual_batch_size,
                epochs=self.args.epoch,
                validation_data=(x_val, y_val),
                callbacks=callbacks_list,
                verbose=1
            )
        
        # Store metrics
        if not is_domain_adaptation:
            # For normal training, history is a keras History object with a history dict
            history_dict = history.history
            # Ensure we have all epochs
            num_epochs = len(history_dict.get('accuracy', []))
            print(f"\n📊 Fold {fold} History Information:")
            print(f"   Total epochs recorded: {num_epochs}")
            print(f"   Expected epochs: {self.args.epoch}")
            print(f"   History keys: {list(history_dict.keys())}")
            
            if num_epochs < self.args.epoch:
                print(f"⚠️ Warning: Training stopped at epoch {num_epochs}/{self.args.epoch}")
                print("   This might be due to early stopping or an error during training")
            
            fold_metrics['acc'].extend(history_dict.get('accuracy', []))
            fold_metrics['val_acc'].extend(history_dict.get('val_accuracy', []))
            fold_metrics['loss'].extend(history_dict.get('loss', []))
            fold_metrics['val_loss'].extend(history_dict.get('val_loss', []))
        else:
            # For LMMD, history is already a dict
            num_epochs = len(history.get('accuracy', []))
            print(f"\n📊 Fold {fold} History Information:")
            print(f"   Total epochs recorded: {num_epochs}")
            print(f"   Expected epochs: {self.args.epoch}")
            print(f"   History keys: {list(history.keys())}")
            
            if num_epochs < self.args.epoch:
                print(f"⚠️ Warning: Training stopped at epoch {num_epochs}/{self.args.epoch}")
                print("   This might be due to early stopping or an error during training")
            
            fold_metrics['acc'].extend(history.get('accuracy', []))
            fold_metrics['val_acc'].extend(history.get('val_accuracy', []))
            fold_metrics['loss'].extend(history.get('loss', []))
            fold_metrics['val_loss'].extend(history.get('val_loss', []))
        
        # Plot metrics for this fold
        self._plot_fold_training_metrics(history, fold, save_dir)
        
        # Load best weights for this fold
        self._load_weights(checkpoint_base)  # TensorFlow will automatically find the .index and .data files
        
        # Evaluate on validation set
        val_loss, val_acc = self.model.evaluate(x_val, y_val, verbose=0)
        y_pred = self.model.predict(x_val)
            
        print(f"Fold {fold} - Validation Accuracy: {val_acc:.4f}")
        
        # Store confusion matrix
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_val, axis=1)
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        self.matrix.append(cm)
        
        # Update best fold if needed
        if val_acc > self.best_fold_acc:
            self.best_fold_acc = val_acc
            self.best_fold_weight_path = checkpoint_base  # Store base path without extensions
        
        # Clean up memory after each fold to avoid memory leak
        if not is_domain_adaptation:
            print(f"\n🧹 Cleaning fold {fold} memory...")
            # Only clear the session (not full reset to keep weights)
            K.clear_session()
            gc.collect()
            # Recreate the session for the next fold
            self.model = None
        
        return fold_metrics

    def train(self, x, y):
        """Train the model using k-fold cross-validation"""
        result = self._train_loop(x, y)
        
        # Clear TensorFlow session and force garbage collection to free memory
        print("\n🧹 Cleaning up memory...")
        tf.keras.backend.clear_session()
        gc.collect()
        
        return result

    def train_with_domain_adaptation(self, source_data, source_labels, target_data, target_labels=None):
        """Train the model with domain adaptation using LMMD loss"""
        if not self.use_lmmd:
            print("⚠️ LMMD loss not enabled. Use --mode train-lmmd for domain adaptation.")
            return self.train(source_data, source_labels)
            
        # Determine max sequence length
        max_seq_len = max(source_data.shape[1], target_data.shape[1])
        feature_dim = source_data.shape[2]
        
        # Pad or truncate sequences
        if source_data.shape[1] < max_seq_len:
            # Pad source data
            pad_length = max_seq_len - source_data.shape[1]
            source_data = np.pad(
                source_data,
                ((0, 0), (0, pad_length), (0, 0)),
                mode='constant'
            )
        elif source_data.shape[1] > max_seq_len:
            # Truncate source data
            source_data = source_data[:, :max_seq_len, :]
            
        if target_data.shape[1] < max_seq_len:
            # Pad target data
            pad_length = max_seq_len - target_data.shape[1]
            target_data = np.pad(
                target_data,
                ((0, 0), (0, pad_length), (0, 0)),
                mode='constant'
            )
        elif target_data.shape[1] > max_seq_len:
            # Truncate target data
            target_data = target_data[:, :max_seq_len, :]
            
        # Update input shape for the model
        self.input_shape = (max_seq_len, feature_dim)
        print(f"Adjusted sequence length to {max_seq_len}")
        print(f"Source data shape: {source_data.shape}")
        print(f"Target data shape: {target_data.shape}")
        
        result = self._train_loop(source_data, source_labels, is_domain_adaptation=True, target_data=target_data)
        
        # Clear TensorFlow session and force garbage collection to free memory
        print("\n🧹 Cleaning up memory...")
        tf.keras.backend.clear_session()
        gc.collect()
        
        return result

    def _plot_fold_training_metrics(self, history, fold_idx, save_dir):
        """Plot and save training metrics for a specific fold"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Check if history is already a dict or has a history attribute
        hist_dict = history if isinstance(history, dict) else history.history
        
        # Ensure we have all metrics
        metrics = {
            'accuracy': hist_dict.get('accuracy', []),
            'val_accuracy': hist_dict.get('val_accuracy', []),
            'loss': hist_dict.get('loss', []),
            'val_loss': hist_dict.get('val_loss', [])
        }
        
        # Create x-axis values based on the length of the metrics
        epochs = range(1, len(metrics['accuracy']) + 1)
        
        # Print debug information
        print(f"\n📊 Fold {fold_idx} Training Metrics:")
        print(f"   Total epochs recorded: {len(epochs)}")
        print(f"   Expected epochs: {self.args.epoch}")
        
        # Plot accuracy
        ax1.plot(epochs, metrics['accuracy'], label='Training')
        ax1.plot(epochs, metrics['val_accuracy'], label='Validation')
        ax1.set_title(f'Fold {fold_idx} - Model Accuracy')
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylim([0, 1])
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(epochs, metrics['loss'], label='Training')
        ax2.plot(epochs, metrics['val_loss'], label='Validation')
        ax2.set_title(f'Fold {fold_idx} - Model Loss')
        ax2.set_ylabel('Loss')
        ax2.set_xlabel('Epoch')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        metrics_file = os.path.join(save_dir, f'fold_{fold_idx}_metrics.png')
        plt.savefig(metrics_file)
        plt.close()
        
        print(f"📈 Fold {fold_idx} training metrics saved to: {metrics_file}")

    def _plot_combined_training_metrics(self, fold_metrics, save_dir):
        """Plot metrics combined from all folds"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Print debug information
        print("\n📊 Combined Training Metrics:")
        print(f"   Total epochs recorded: {len(fold_metrics['acc'])}")
        print(f"   Expected epochs per fold: {self.args.epoch}")
        print(f"   Number of folds: {self.args.split_fold}")
        print(f"   Expected total epochs: {self.args.epoch * self.args.split_fold}")
        
        # Plot accuracy
        ax1.plot(fold_metrics['val_acc'], label='Validation')
        ax1.plot(fold_metrics['acc'], label='Training')
        ax1.set_title('Model Accuracy Across Folds')
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylim([0, 1])
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(fold_metrics['val_loss'], label='Validation')
        ax2.plot(fold_metrics['loss'], label='Training')
        ax2.set_title('Model Loss Across Folds')
        ax2.set_ylabel('Loss')
        ax2.set_xlabel('Epoch')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        metrics_file = os.path.join(save_dir, 'training_metrics.png')
        plt.savefig(metrics_file)
        plt.close()
        
        print(f"📊 Combined training metrics saved to: {metrics_file}")

    def _evaluate_model(self, x, y, is_domain_adaptation=False, target_data=None):
        """Helper method to evaluate model on given data"""
        if is_domain_adaptation and target_data is not None:
            loss, accuracy = self.model.evaluate([x, target_data], y, verbose=0)
            y_pred = self.model.predict([x, target_data])
        else:
            loss, accuracy = self.model.evaluate(x, y, verbose=0)
            y_pred = self.model.predict(x)
            
        return loss, accuracy, y_pred

    def _generate_evaluation_metrics(self, y_true, y_pred, class_labels):
        """Generate evaluation metrics including confusion matrix and classification report"""
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_true, axis=1)
        
        # Generate confusion matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        
        # Generate classification report
        report = classification_report(
            y_true_classes, 
            y_pred_classes,
            target_names=class_labels,
            digits=4
        )
        
        return cm, report

    def _save_evaluation_results(self, cm, report, result_dir, result_filename):
        """Save evaluation results to files"""
        # Create result directory if it doesn't exist
        os.makedirs(result_dir, exist_ok=True)
        
        # Save confusion matrix plot
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.colorbar()
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(result_dir, f"{result_filename}_confusion_matrix.png"))
        plt.close()
        
        # Save classification report
        with open(os.path.join(result_dir, f"{result_filename}_classification_report.txt"), 'w') as f:
            f.write(report)
            
        print(f"📊 Evaluation results saved to: {result_dir}")

    def evaluate_test(self, x_test, y_test, path=None, result_filename=None, result_dir=None):
        """Evaluate model on test data and generate metrics"""
        # Ensure model exists
        if self.model is None:
            print("\n🔄 Creating model architecture...")
            self.create_model()
            
        # Load weights if path is provided
        if path:
            self._load_weights(path)
            
        print("\n🔍 Validating test data:")
        print(f"Input shape: {x_test.shape}")
        print(f"Label shape: {y_test.shape}")
        
        # Validate input data
        if len(x_test.shape) != 3:
            raise ValueError(f"Expected 3D input (batch_size, time_steps, features), got shape {x_test.shape}")
            
        # Validate label shape
        if len(y_test.shape) != 2:
            raise ValueError(f"Expected 2D labels (batch_size, num_classes), got shape {y_test.shape}")
            
        # Validate number of classes
        if y_test.shape[1] != len(self.class_labels):
            raise ValueError(f"Expected {len(self.class_labels)} classes in labels, got {y_test.shape[1]}")
            
        # Validate feature dimension
        if x_test.shape[2] != self.input_shape[1]:
            raise ValueError(f"Feature dimension mismatch: expected {self.input_shape[1]}, got {x_test.shape[2]}")
            
        # Validate matching number of samples
        if x_test.shape[0] != y_test.shape[0]:
            raise ValueError(f"Number of samples mismatch: {x_test.shape[0]} inputs vs {y_test.shape[0]} labels")
            
        # Handle sequence length mismatch
        if x_test.shape[1] != self.input_shape[0]:
            print(f"\n⚠️ Sequence length mismatch:")
            print(f"   Expected: {self.input_shape[0]}")
            print(f"   Got: {x_test.shape[1]}")
            print("   Adjusting sequence length...")
            
            if x_test.shape[1] < self.input_shape[0]:
                # Pad
                pad_length = self.input_shape[0] - x_test.shape[1]
                x_test = np.pad(
                    x_test,
                    ((0, 0), (0, pad_length), (0, 0)),
                    mode='constant'
                )
                print(f"   Padded to length: {x_test.shape[1]}")
            else:
                # Truncate
                x_test = x_test[:, :self.input_shape[0], :]
                print(f"   Truncated to length: {x_test.shape[1]}")
            
        print(f"\n✅ Validation complete:")
        print(f"   Final input shape: {x_test.shape}")
        print(f"   Label shape: {y_test.shape}")
        print(f"   Number of classes: {len(self.class_labels)}")
        
        # Set default result filename using consistent naming
        if result_filename is None:
            result_filename = f"{self.args.data}_{self.args.feature_type.upper()}_{self.now}_test"
            
        # Use base results directory unless custom one provided (e.g., for cross-corpus)
        eval_dir = result_dir if result_dir else self.base_results
        
        # Ensure evaluation directory exists
        os.makedirs(eval_dir, exist_ok=True)
        
        # Evaluate model
        loss, accuracy, y_pred = self._evaluate_model(x_test, y_test)
        print(f"\n📊 Test Results:")
        print(f"   Loss: {loss:.4f}")
        print(f"   Accuracy: {accuracy:.4f}")
        
        # Generate and save metrics
        cm, report = self._generate_evaluation_metrics(y_test, y_pred, self.class_labels)
        
        # Save results
        results_base = os.path.join(eval_dir, result_filename)
        
        # Save confusion matrix visualization
        plt.figure(figsize=(12, 9))
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.colorbar()
        plt.title(f'Confusion Matrix\nAccuracy: {accuracy:.4f}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f"{results_base}_confusion_matrix.png")
        plt.close()
        
        # Save classification report
        with open(f"{results_base}_classification_report.txt", 'w') as f:
            f.write(report)
        
        # Save detailed results to Excel
        self._save_results_to_excel(
            f"{results_base}_results.xlsx",
            cm,
            report,
            accuracy,
            self.class_labels
        )
        
        print(f"\n💾 Results saved to:")
        print(f"   {results_base}_confusion_matrix.png")
        print(f"   {results_base}_classification_report.txt")
        print(f"   {results_base}_results.xlsx")
        
        # Clear memory before returning
        print("\n🧹 Cleaning up memory...")
        tf.keras.backend.clear_session()
        gc.collect()
        
        return y_pred, accuracy

    def _load_weights(self, path):
        """Load model weights from the specified path"""
        # Ensure model exists
        if self.model is None:
            print("\n🔄 Creating model architecture...")
            self.create_model()
            
        try:
            print(f"📋 Loading model weights from: {path}")
            # Check if path is a directory
            if os.path.isdir(path):
                # Find the most recent weight file
                weight_files = [f for f in os.listdir(path) if f.endswith('.index')]
                if not weight_files:
                    raise ValueError(f"No weight files found in {path}")
                
                # Sort by creation time (most recent first)
                weight_files.sort(key=lambda x: os.path.getmtime(os.path.join(path, x)), reverse=True)
                
                # Get base name without .index extension
                latest_weights = os.path.join(path, weight_files[0][:-6])  # Remove .index
                print(f"✅ Found latest weights: {latest_weights}")
                self.model.load_weights(latest_weights)
            else:
                # Direct path to weights (TensorFlow will automatically handle extensions)
                self.model.load_weights(path)
            print("✅ Model weights loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model weights: {e}")
            raise

    def load_weights(self, path):
        """Public method to load model weights"""
        self._load_weights(path)

    def save_weights(self, path):
        """Save model weights to the specified path"""
        try:
            print(f"💾 Saving model weights to: {path}")
            # TensorFlow will automatically add .index and .data-* extensions
            self.model.save_weights(path, save_format="tf")
            print("✅ Model weights saved successfully")
        except Exception as e:
            print(f"❌ Error saving model weights: {e}")
            raise

    def _save_results_to_excel(self, excel_path, confusion_matrix, classification_report, accuracy, used_labels):
        """Save evaluation results to Excel file"""
        print(f"📝 Saving evaluation results to: {excel_path}")

        try:
            # Create Excel writer
            writer = pd.ExcelWriter(excel_path, engine="openpyxl")

            # Save confusion matrix
            df_cm = pd.DataFrame(confusion_matrix, index=used_labels, columns=used_labels)
            df_cm.to_excel(writer, sheet_name="Confusion_Matrix")

            # Save classification report
            if isinstance(classification_report, str):
                # Parse string report into dictionary
                lines = classification_report.strip().split("\n")
                headers = [x for x in lines[0].split("  ") if x]

                # Skip the first two lines (headers) and the last line (accuracy)
                data = []
                for line in lines[2:-1]:
                    if line.strip():
                        row = {}
                        values = [x for x in line.split("  ") if x]
                        if len(values) >= len(headers):
                            for i, header in enumerate(headers):
                                row[header] = values[i]
                            data.append(row)

                df_cr = pd.DataFrame(data)
            else:
                # Classification report is already a dictionary
                df_cr = pd.DataFrame(classification_report).T

            df_cr.to_excel(writer, sheet_name="Classification_Report")

            # Save accuracy as a separate sheet
            df_acc = pd.DataFrame({"Accuracy": [accuracy]})
            df_acc.to_excel(writer, sheet_name="Accuracy")

            # Close writer
            writer.close()
            print("✅ Evaluation results saved successfully")

        except Exception as e:
            print(f"❌ Failed to save results: {e}")

        # Clean up
        K.clear_session()
        self.matrix = []
        self.eva_matrix = []
        self.trained = True
