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

# Standard libraries
import numpy as np
import datetime
import copy
import pandas as pd
import shutil

# TensorFlow imports
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import (
    Conv1D, SpatialDropout1D, BatchNormalization, Activation, add,
    GlobalAveragePooling1D, Lambda, Dense, Input
)
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import callbacks
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Evaluation tools
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import KFold

# ------------------------
# TEMPORAL BLOCK DEFINITION
# ------------------------
def temporal_block(inputs, dilation, activation, nb_filters, kernel_size, dropout_rate=0.0):
    skip_connection = inputs
    x = Conv1D(filters=nb_filters, kernel_size=kernel_size, dilation_rate=dilation, padding='causal')(inputs)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = SpatialDropout1D(dropout_rate)(x)
    x = Conv1D(filters=nb_filters, kernel_size=kernel_size, dilation_rate=dilation, padding='causal')(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = SpatialDropout1D(dropout_rate)(x)
    if skip_connection.shape[-1] != x.shape[-1]:
        skip_connection = Conv1D(filters=nb_filters, kernel_size=1, padding='same')(skip_connection)
    gated = Lambda(sigmoid)(x)
    return skip_connection * gated

# ------------------------
# DILATED TEMPORAL NETWORK
# ------------------------
class TemporalConvNet:
    def __init__(self, nb_filters=64, kernel_size=2, nb_stacks=1, dilations=8,
                 activation="relu", dropout_rate=0.1, name='TemporalConvNet'):
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
        f = Conv1D(self.nb_filters, kernel_size=1, padding='causal')(forward)
        b = Conv1D(self.nb_filters, kernel_size=1, padding='causal')(backward)
        final_skips = []
        for _ in range(self.nb_stacks):
            for dilation_rate in [2 ** i for i in range(self.dilations)]:
                f = temporal_block(f, dilation_rate, self.activation, self.nb_filters, self.kernel_size, self.dropout_rate)
                b = temporal_block(b, dilation_rate, self.activation, self.nb_filters, self.kernel_size, self.dropout_rate)
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
            initializer='uniform',
            trainable=True
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
    labels *= (1 - factor)
    labels += (factor / labels.shape[1])
    return labels



class SpeechEmotionModel:
    def __init__(self, input_shape, class_labels, args):
        self.args = args
        self.input_shape = input_shape
        
        # For RAVDESS, we need to ensure num_classes is consistent with the one-hot encoded labels
        if args.data == "RAVDESS":
            # Use the shape of the one-hot encoded labels passed to fit() later
            self.num_classes = max(10, len(class_labels))  # RAVDESS has indices up to 9, requiring 10 classes
        else:
            self.num_classes = len(class_labels)
            
        self.class_labels = class_labels
        self.model = None
        self.matrix = []
        self.eva_matrix = []
        self.acc = 0
        self.trained = False
        self.result_dir = os.path.join(self.args.result_path, self.args.data)
        self.now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.best_fold_acc = 0
        self.best_fold_weight_path = ""
        self.use_lmmd = hasattr(args, 'use_lmmd') and args.use_lmmd

        # Configure GPU if available (Metal or CUDA)
        try:
            with tf.device('/device:GPU:0'):
                # Test if GPU is working
                tf.zeros((1, 1))
                if tf.test.is_built_with_cuda():
                    print("✅ Model will use CUDA GPU for computation")
                else:
                    print("✅ Model will use Metal GPU for computation")
        except Exception as e:
            print("⚠️ Model will use CPU")

        print(f"🧠 Initialized SER model with input shape: {input_shape}")
        
        # Initialize LMMD loss if needed
        if self.use_lmmd:
            try:
                from .lmmd_loss import LMMDLoss
                self.lmmd_loss = LMMDLoss(
                    num_classes=self.num_classes,
                    weight=0.5,  # Weight for LMMD loss (can be tuned)
                    kernel_mul=2.0,
                    kernel_num=5
                )
                print(f"✅ LMMD Loss initialized with {self.num_classes} classes")
            except ImportError:
                print("⚠️ Could not import LMMD loss, falling back to regular training")
                self.use_lmmd = False

    def create_model(self):
        print("\n🛠️ Building model architecture...")
        inputs = Input(shape=(self.input_shape[0], self.input_shape[1]))
        conv_net = TemporalConvNet(
            nb_filters=self.args.filter_size,
            kernel_size=self.args.kernel_size,
            nb_stacks=self.args.stack_size,
            dilations=self.args.dilation_size,
            dropout_rate=self.args.dropout,
            activation=self.args.activation
        )
        conv_output = conv_net(inputs)
        attention_out = WeightLayer()(conv_output)
        outputs = Dense(self.num_classes, activation='softmax')(attention_out)
        self.model = KerasModel(inputs=inputs, outputs=outputs)
        optimizer = Adam(learning_rate=self.args.lr, beta_1=self.args.beta1, beta_2=self.args.beta2, epsilon=1e-8)
        self.model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
        print("✅ Model compiled successfully!\n")

    def train(self, x, y):
        print("🎯 Starting k-fold training...")
         # Pad sequences if needed
        if isinstance(x, list) or len(x.shape) != 3:
            x = pad_sequences(x, padding='post', dtype='float32')

        # Update input shape AFTER padding
        self.input_shape = x.shape[1:]

        save_dir = self.args.model_path
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        test_model_dir = os.path.join("test_models", self.args.data)
        os.makedirs(test_model_dir, exist_ok=True)
        kfold = KFold(n_splits=self.args.split_fold, shuffle=True, random_state=self.args.random_seed)
        avg_acc, avg_loss = 0, 0

        for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(x, y), 1):
            print(f"🔁 Fold {fold_idx}/{self.args.split_fold}")

            self.create_model()
            y_train_smoothed = smooth_labels(copy.deepcopy(y[train_idx]), 0.1)
            fold_folder = os.path.join(save_dir, f"{self.args.data}_{self.args.random_seed}_{self.now}")
            os.makedirs(fold_folder, exist_ok=True)
            weight_name = f"{self.args.split_fold}-fold_weights_best_{fold_idx}.weights.h5"
            weight_path = os.path.join(fold_folder, weight_name)
            self.model.fit(
                x[train_idx], y_train_smoothed,
                validation_data=(x[test_idx], y[test_idx]),
                batch_size=self.args.batch_size,
                epochs=self.args.epoch,
                verbose=1,
                callbacks=[callbacks.ModelCheckpoint(weight_path, verbose=1, save_weights_only=True)]
            )
            self.model.load_weights(weight_path)
            loss, acc = self.model.evaluate(x[test_idx], y[test_idx], verbose=0)
            avg_loss += loss
            avg_acc += acc
            print(f"✅ Fold {fold_idx} Accuracy: {round(acc * 100, 2)}%")

            if acc > self.best_fold_acc:
                self.best_fold_acc = acc
                self.best_fold_weight_path = weight_path

            y_pred = self.model.predict(x[test_idx])
            self.matrix.append(confusion_matrix(np.argmax(y[test_idx], axis=1), np.argmax(y_pred, axis=1)))
            eval_dict = classification_report(
                np.argmax(y[test_idx], axis=1),
                np.argmax(y_pred, axis=1),
                target_names=self.class_labels,
                output_dict=True, zero_division=0
            )
            self.eva_matrix.append(eval_dict)
            print(classification_report(
                np.argmax(y[test_idx], axis=1),
                np.argmax(y_pred, axis=1),
                target_names=self.class_labels, zero_division=0))

        self.acc = avg_acc / self.args.split_fold
        print(f"\n📊 Average Accuracy over {self.args.split_fold} folds: {round(self.acc * 100, 2)}%")

        # Save best weight to test_models
        if self.best_fold_weight_path:
            best_name = os.path.basename(self.best_fold_weight_path)

            feature_subfolder = os.path.join("test_models", self.args.data, self.args.feature_type.upper())

            # FIX: Remove any existing file with the same path before creating folder
            if os.path.exists(feature_subfolder) and not os.path.isdir(feature_subfolder):
                os.remove(feature_subfolder)

            os.makedirs(feature_subfolder, exist_ok=True)
            # Add feature_type-specific subfolder (e.g., MFCC or LOGMEL)
            #feature_subfolder = os.path.join("test_models", self.args.data, self.args.feature_type.upper())
            #os.makedirs(feature_subfolder, exist_ok=True)

            # Save the weight
            shutil.copy(self.best_fold_weight_path, os.path.join(feature_subfolder, best_name))
            print(f"🏆 Best fold model ({round(self.best_fold_acc * 100, 2)}%) saved to {feature_subfolder}/{best_name}")

    def train_with_domain_adaptation(self, source_data, source_labels, target_data, target_labels=None):
        """
        Train the model with domain adaptation using LMMD loss
        
        Args:
            source_data: Source domain data (e.g., EMODB)
            source_labels: Source domain labels
            target_data: Target domain data (e.g., RAVDESS)
            target_labels: Optional ground truth labels for the target domain
        """
        if not self.use_lmmd:
            print("⚠️ LMMD loss not enabled. Use --use_lmmd flag for domain adaptation.")
            return self.train(source_data, source_labels)
            
        print(f"🚀 Starting domain adaptation training with LMMD loss...")
        print(f"📊 Source domain: shape {source_data.shape}")
        print(f"📊 Target domain: shape {target_data.shape}")
            
        # Import LMMD loss
        from .lmmd_loss import LMMDLoss, get_lmmd_loss
            
        # Create the model
        if self.model is None:
            self.create_model()
            
        # Make sure source labels are one-hot encoded
        if len(source_labels.shape) == 1:
            source_labels_onehot = to_categorical(source_labels, num_classes=self.num_classes)
        else:
            source_labels_onehot = source_labels
            
        # Get initial pseudo-labels for target domain if no labels are provided
        if target_labels is None:
            # Use model to predict initial pseudo-labels
            self.model.compile(
                optimizer=Adam(learning_rate=self.args.lr, beta_1=self.args.beta1, beta_2=self.args.beta2),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Fit on source data first to get basic model
            print("🔍 Pretraining on source domain for initial pseudo-labels...")
            self.model.fit(
                source_data, source_labels_onehot,
                batch_size=self.args.batch_size,
                epochs=5,  # Just a few epochs for initialization
                verbose=0
            )
            
            # Generate pseudo-labels
            target_preds = self.model.predict(target_data)
            target_pseudo_labels = target_preds
        else:
            # Use provided target labels
            if len(target_labels.shape) == 1:
                target_pseudo_labels = to_categorical(target_labels, num_classes=self.num_classes)
            else:
                target_pseudo_labels = target_labels
                
        # Prepare data for cross-validation
        kf = KFold(n_splits=self.args.split_fold, shuffle=True, random_state=self.args.random_seed)
        fold = 1
        
        # Create model result directory
        os.makedirs(self.result_dir, exist_ok=True)
        
        # Cross-validation loop
        acc_sum = 0
        
        test_models_path = f"test_models/{self.args.data}/{self.args.feature_type}/"
        os.makedirs(test_models_path, exist_ok=True)
            
        for train_idx, val_idx in kf.split(source_data):
            print(f"\n📂 Fold {fold}/{self.args.split_fold}")
            
            # Split source data
            x_train, x_val = source_data[train_idx], source_data[val_idx]
            y_train, y_val = source_labels_onehot[train_idx], source_labels_onehot[val_idx]
            
            # Get validation indices for target domain (same proportion as source)
            target_val_size = int(len(val_idx) / len(train_idx) * target_data.shape[0])
            target_val_indices = np.random.choice(target_data.shape[0], target_val_size, replace=False)
            target_train_indices = np.array([i for i in range(target_data.shape[0]) if i not in target_val_indices])
            
            # Split target data
            target_train = target_data[target_train_indices]
            target_val = target_data[target_val_indices]
            target_train_labels = target_pseudo_labels[target_train_indices]
            target_val_labels = target_pseudo_labels[target_val_indices]
            
            # Recreate model for this fold
            tf.keras.backend.clear_session()
            self.create_model()
            
            # Create a custom training function using LMMD loss
            print("🛠️ Building domain adaptation model...")
            
            # Define a custom training step
            @tf.function
            def train_step(source_batch, source_labels_batch, target_batch, target_labels_batch):
                with tf.GradientTape() as tape:
                    # Get source predictions
                    source_preds = self.model(source_batch, training=True)
                    
                    # Get target predictions
                    target_preds = self.model(target_batch, training=True)
                    
                    # Extract features from the layer before final dense layer
                    source_features_model = tf.keras.Model(
                        inputs=self.model.input,
                        outputs=self.model.layers[-2].output
                    )
                    target_features_model = tf.keras.Model(
                        inputs=self.model.input,
                        outputs=self.model.layers[-2].output
                    )
                    
                    source_features = source_features_model(source_batch, training=True)
                    target_features = target_features_model(target_batch, training=True)
                    
                    # Calculate classification loss
                    class_loss = tf.keras.losses.categorical_crossentropy(
                        source_labels_batch, source_preds
                    )
                    
                    # Calculate LMMD loss
                    lmmd = self.lmmd_loss.get_lmmd_loss_fn()(
                        source_labels_batch, source_preds,
                        source_features, target_features,
                        source_labels_batch, target_labels_batch
                    )
                    
                    # Calculate total loss
                    total_loss = tf.reduce_mean(class_loss) + lmmd
                    
                # Compute gradients
                gradients = tape.gradient(total_loss, self.model.trainable_variables)
                
                # Update weights
                optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                
                return total_loss, tf.reduce_mean(class_loss), lmmd
            
            # Create optimizer
            optimizer = Adam(learning_rate=self.args.lr, beta_1=self.args.beta1, beta_2=self.args.beta2)
            
            # Compile the model for evaluation metrics
            self.model.compile(
                optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Setup callbacks
            weight_path = os.path.join(test_models_path, f"{self.args.split_fold}-fold_weights_{fold}.weights.h5")
            best_weight_path = os.path.join(test_models_path, f"{self.args.split_fold}-fold_weights_best_{fold}.weights.h5")
            
            checkpoint = callbacks.ModelCheckpoint(
                weight_path,
                monitor='val_accuracy',
                verbose=0,
                save_best_only=False,
                save_weights_only=True,
                mode='max'
            )
            
            best_checkpoint = callbacks.ModelCheckpoint(
                best_weight_path,
                monitor='val_accuracy',
                verbose=0,
                save_best_only=True,
                save_weights_only=True,
                mode='max'
            )
            
            early_stopping = callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=30,
                restore_best_weights=True
            )
            
            # Train the model with domain adaptation
            print(f"🚂 Training with domain adaptation...")
            
            # Number of batches
            batch_size = self.args.batch_size
            steps_per_epoch = len(x_train) // batch_size
            
            # Training loop
            best_val_acc = 0
            patience_counter = 0
            max_patience = 30
            
            for epoch in range(self.args.epoch):
                # Shuffle indices
                source_indices = np.random.permutation(len(x_train))
                target_indices = np.random.permutation(len(target_train))
                
                epoch_loss = 0
                epoch_class_loss = 0
                epoch_lmmd_loss = 0
                
                # Process batches
                for batch in range(steps_per_epoch):
                    start_idx = batch * batch_size
                    end_idx = min(start_idx + batch_size, len(x_train))
                    
                    # Get source batch
                    source_batch_indices = source_indices[start_idx:end_idx]
                    source_batch = x_train[source_batch_indices]
                    source_batch_labels = y_train[source_batch_indices]
                    
                    # Get target batch (cyclically if needed)
                    target_batch_indices = target_indices[(start_idx % len(target_train)):((start_idx % len(target_train)) + batch_size)]
                    if len(target_batch_indices) < batch_size:
                        target_batch_indices = np.concatenate([
                            target_batch_indices, 
                            target_indices[:(batch_size - len(target_batch_indices))]
                        ])
                    
                    target_batch = target_train[target_batch_indices]
                    target_batch_labels = target_train_labels[target_batch_indices]
                    
                    # Train on this batch
                    batch_loss, batch_class_loss, batch_lmmd_loss = train_step(
                        source_batch, source_batch_labels, 
                        target_batch, target_batch_labels
                    )
                    
                    epoch_loss += batch_loss / steps_per_epoch
                    epoch_class_loss += batch_class_loss / steps_per_epoch
                    epoch_lmmd_loss += batch_lmmd_loss / steps_per_epoch
                
                # Evaluate on validation set
                val_results = self.model.evaluate(x_val, y_val, verbose=0)
                val_loss, val_acc = val_results
                
                # Update best accuracy
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    # Save best weights
                    self.model.save_weights(best_weight_path)
                else:
                    patience_counter += 1
                
                # Print progress
                print(f"Epoch {epoch+1}/{self.args.epoch} - Loss: {epoch_loss:.4f} (Class: {epoch_class_loss:.4f}, LMMD: {epoch_lmmd_loss:.4f}) - Val_acc: {val_acc:.4f}")
                
                # Update pseudo-labels for target domain every 10 epochs
                if (epoch + 1) % 10 == 0:
                    target_preds = self.model.predict(target_train)
                    target_train_labels = target_preds
                
                # Early stopping
                if patience_counter >= max_patience:
                    print(f"⏱️ Early stopping at epoch {epoch+1}")
                    break
            
            # Load best weights
            self.model.load_weights(best_weight_path)
            
            # Evaluate on validation set
            val_results = self.model.evaluate(x_val, y_val, verbose=0)
            val_loss, val_acc = val_results
            
            print(f"📊 Fold {fold} validation accuracy: {val_acc:.4f}")
            acc_sum += val_acc
            
            # Save confusion matrix
            y_pred = np.argmax(self.model.predict(x_val), axis=1)
            y_true = np.argmax(y_val, axis=1)
            cm = confusion_matrix(y_true, y_pred)
            self.matrix.append(cm)
            
            if val_acc > self.best_fold_acc:
                self.best_fold_acc = val_acc
                self.best_fold_weight_path = best_weight_path
            
            fold += 1
        
        # Calculate average accuracy
        acc_avg = acc_sum / self.args.split_fold
        print(f"\n📊 Average validation accuracy: {acc_avg:.4f}")
        print(f"🥇 Best fold accuracy: {self.best_fold_acc:.4f}")
        print(f"💾 Best weights saved to: {self.best_fold_weight_path}")
        
        # Load best weights for inference
        self.model.load_weights(self.best_fold_weight_path)
        self.trained = True
        self.acc = acc_avg
        
        return self.acc

    def evaluate_test(self, x_test, y_test, path=None, result_filename=None, result_dir=None):
        """
        Evaluate model on test data.
        
        Args:
            x_test: Test features
            y_test: Test labels (one-hot encoded)
            path: Path to model weights
            result_filename: Custom filename for results (used in cross-corpus validation)
            result_dir: Custom directory for saving results (used in cross-corpus validation)
        """
        if not path:
            print("❌ No model weights path provided!")
            return
            
        print(f"📋 Loading model weights from: {path}")
        self._load_weights(path)
        
        y_pred = self.model.predict(x_test)
        
        # Convert one-hot encoded to class indices
        y_true = np.argmax(y_test, axis=1)
        y_pred = np.argmax(y_pred, axis=1)

        # Generate evaluation matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Check for class mismatch and handle it
        unique_classes = np.unique(np.concatenate([y_true, y_pred]))
        num_unique_classes = len(unique_classes)
        
        if num_unique_classes != len(self.class_labels):
            print(f"⚠️ Class mismatch: {num_unique_classes} classes detected, but {len(self.class_labels)} class labels provided")
            # Use a subset of labels or generate generic labels
            if num_unique_classes < len(self.class_labels):
                # Find which classes are actually present in the data
                present_classes = sorted(unique_classes)
                present_labels = [self.class_labels[i] for i in present_classes if i < len(self.class_labels)]
                
                # Fill in any missing labels with generic names
                if len(present_labels) < num_unique_classes:
                    present_labels = [f"Class {i}" for i in range(num_unique_classes)]
                
                print(f"🔄 Using subset of labels: {present_labels}")
                cr = classification_report(y_true, y_pred, labels=range(num_unique_classes), target_names=present_labels)
                used_labels = present_labels
            else:
                # Generate generic labels if we have more classes than labels
                generic_labels = [f"Class {i}" for i in range(num_unique_classes)]
                print(f"🔄 Using generic labels: {generic_labels}")
                cr = classification_report(y_true, y_pred, labels=range(num_unique_classes), target_names=generic_labels)
                used_labels = generic_labels
        else:
            cr = classification_report(y_true, y_pred, target_names=self.class_labels)
            used_labels = self.class_labels
            
        accuracy = np.sum(np.diag(cm)) / np.sum(cm)
        
        # Print results
        print(f"\n📊 Test Accuracy: {accuracy:.4f}\n")
        print("📝 Classification Report:")
        print(cr)
        print("\n🧮 Confusion Matrix:")
        print(cm)
        
        # Override result_dir if provided (for cross-corpus validation)
        if result_dir:
            self.result_dir = result_dir
            
        # Create result directory if it doesn't exist
        if not self.result_dir:
            self.result_dir = os.path.join(self.args.result_path, self.args.data)
        os.makedirs(self.result_dir, exist_ok=True)
            
        # Use custom filename if provided, otherwise use timestamp
        if result_filename:
            filename = f"{result_filename}.xlsx"
        else:
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            filename = f"evaluation_{timestamp}.xlsx"
            
        excel_path = os.path.join(self.result_dir, filename)
        
        self._save_results_to_excel(excel_path, cm, cr, accuracy, used_labels)
        print(f"✅ Results saved to: {excel_path}")
        return y_pred, accuracy

    def _load_weights(self, path):
        """Helper method to load weights from path"""
        # Check if path is a directory or a file
        if os.path.isdir(path):
            print(f"📂 Searching for model weights in directory: {path}")
            weight_files = [f for f in os.listdir(path) if f.endswith('.h5')]
            
            if not weight_files:
                raise ValueError(f"No weight files found in {path}")
                
            # Sort by creation time (most recent first)
            weight_files.sort(key=lambda x: os.path.getmtime(os.path.join(path, x)), reverse=True)
            
            # Prefer files with "best" in the name
            best_files = [f for f in weight_files if "best" in f.lower()]
            if best_files:
                weight_file = os.path.join(path, best_files[0])
                print(f"✅ Found best model: {best_files[0]}")
            else:
                weight_file = os.path.join(path, weight_files[0])
                print(f"✅ Using most recent model: {weight_files[0]}")
        else:
            weight_file = path
        
        print(f"📂 Loading weights from: {weight_file}")
        
        # Just create the model architecture with the current number of classes
        self.create_model()
        
        # If this is a weights-only file (most likely case)
        if "weights" in os.path.basename(weight_file).lower():
            try:
                # First try to load directly with skip_mismatch=True
                print("🔍 Attempting to load weights with skip_mismatch=True")
                self.model.load_weights(weight_file, skip_mismatch=True)
                print("✅ Successfully loaded compatible weights")
                return
            except Exception as e:
                print(f"⚠️ Error with skip_mismatch approach: {e}")
                print("🔄 Trying manual layer-by-layer loading...")
                
                try:
                    # Try to load weights one layer at a time, skipping problematic layers
                    import h5py
                    f = h5py.File(weight_file, 'r')
                    
                    # Get layer names from the file
                    if 'layer_names' in f.attrs:
                        layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]
                        print(f"📄 Found {len(layer_names)} layers in weights file")
                        
                        # For each layer in the model, try to load weights if name matches
                        for layer in self.model.layers:
                            if layer.name in layer_names and 'dense' not in layer.name:
                                # Get weight names for this layer
                                g = f[layer.name]
                                weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
                                
                                # Load weights for this layer
                                weight_values = [np.array(g[weight_name]) for weight_name in weight_names]
                                try:
                                    layer.set_weights(weight_values)
                                    print(f"✓ Loaded weights for layer: {layer.name}")
                                except Exception as layer_error:
                                    print(f"⚠️ Could not load weights for layer {layer.name}: {layer_error}")
                        
                        print("✅ Successfully loaded compatible layer weights")
                        return
                    else:
                        print("⚠️ No layer_names attribute found in weights file")
                except Exception as h5_error:
                    print(f"⚠️ Failed to load with h5py: {h5_error}")
                    
                # Try numpy approach if h5py failed
                try:
                    print("🔄 Trying NumPy approach...")
                    weights_data = np.load(weight_file, allow_pickle=True)
                    if isinstance(weights_data, np.ndarray):
                        weights_data = weights_data.item()
                        
                        # Find layers that don't have 'dense' in their name
                        layer_names = [name for name in weights_data.keys() if 'dense' not in name]
                        print(f"📄 Found {len(layer_names)} non-dense layers in weights file")
                        
                        # Load weights for non-dense layers
                        for layer in self.model.layers:
                            if layer.name in layer_names:
                                try:
                                    # Get weight values for this layer
                                    if layer.name in weights_data:
                                        if isinstance(weights_data[layer.name], dict):
                                            # Extract weight values from dictionary
                                            weight_values = [weights_data[layer.name][key] for key in weights_data[layer.name].keys()]
                                        else:
                                            # Weights are directly stored
                                            weight_values = weights_data[layer.name]
                                            
                                        layer.set_weights(weight_values)
                                        print(f"✓ Loaded weights for layer: {layer.name}")
                                except Exception as layer_error:
                                    print(f"⚠️ Could not load weights for layer {layer.name}: {layer_error}")
                                    
                        print("✅ Successfully loaded compatible layer weights (skipping dense layers)")
                        return
                except Exception as np_error:
                    print(f"⚠️ Failed to load with NumPy: {np_error}")
                    
        # As a last resort, try to load the model directly
        try:
            print("🔍 Attempting to load or recreate model")
            
            try:
                # Try to load the model directly 
                loaded_model = tf.keras.models.load_model(weight_file)
                
                # If successful, transfer weights from all but the last layer
                for i, layer in enumerate(self.model.layers[:-1]):
                    if i < len(loaded_model.layers) - 1:
                        layer.set_weights(loaded_model.layers[i].get_weights())
                
                print("✅ Successfully transferred weights from compatible layers")
                return
            except Exception as e:
                print(f"⚠️ Error loading complete model: {e}")
                # If all approaches fail, create a simple model
                print("⚠️ Could not load weights. Using initialized model.")
                
        except Exception as e:
            print(f"❌ All loading approaches failed!")
            print(f"❌ Last error: {e}")
            raise ValueError(f"Could not load model weights after multiple attempts. Try training the model first.")

    def _save_results_to_excel(self, excel_path, confusion_matrix, classification_report, accuracy, used_labels):
        """Save evaluation results to Excel file"""
        print(f"📝 Saving evaluation results to: {excel_path}")
        
        try:
            # Create Excel writer
            writer = pd.ExcelWriter(excel_path, engine='openpyxl')
            
            # Save confusion matrix
            df_cm = pd.DataFrame(confusion_matrix, 
                                index=used_labels, 
                                columns=used_labels)
            df_cm.to_excel(writer, sheet_name="Confusion_Matrix")
            
            # Save classification report
            if isinstance(classification_report, str):
                # Parse string report into dictionary
                lines = classification_report.strip().split('\n')
                headers = [x for x in lines[0].split('  ') if x]
                
                # Skip the first two lines (headers) and the last line (accuracy)
                data = []
                for line in lines[2:-1]:
                    if line.strip():
                        row = {}
                        values = [x for x in line.split('  ') if x]
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
            df_acc = pd.DataFrame({'Accuracy': [accuracy]})
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
