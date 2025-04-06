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

        print(f"🧠 Initialized SER model with input shape: {input_shape}")

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

    def evaluate_test(self, x_test, y_test, path):
         # Pad test data if needed
        if isinstance(x_test, list) or len(x_test.shape) != 3:
            x_test = pad_sequences(x_test, padding='post', dtype='float32')

        # Update input shape BEFORE creating model
        self.input_shape = x_test.shape[1:]

        self.create_model()
        self.model.load_weights(path)
        loss, acc = self.model.evaluate(x_test, y_test, verbose=1)
        print(f"\n🎯 Test Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")
        y_pred = self.model.predict(x_test)
        print("\n🧾 Classification Report:")
        print(classification_report(
            np.argmax(y_test, axis=1),
            np.argmax(y_pred, axis=1),
            target_names=self.class_labels,
            zero_division=0
        ))
        # Create a result subfolder for the feature type (e.g., results/EMODB/MFCC/)
        result_subfolder = os.path.join("test_models", self.args.data, self.args.feature_type.upper())

        # Fix: if a file (not a folder) exists at this path, remove it first
        if os.path.exists(result_subfolder) and not os.path.isdir(result_subfolder):
            os.remove(result_subfolder)

        os.makedirs(result_subfolder, exist_ok=True)

        #result_subfolder = os.path.join(self.result_dir, self.args.feature_type.upper())
        #os.makedirs(result_subfolder, exist_ok=True)

        # Create a descriptive filename like EMODB_MFCC_test_85.5_46_2025-03-28_17-24-59.xlsx
        filename = f"{self.args.data}_{self.args.feature_type.upper()}_test_{round(acc * 100, 2)}_{self.args.random_seed}_{self.now}.xlsx"
        result_file = os.path.join(result_subfolder, filename)

        print(f"📁 Saving evaluation results to: {result_file}")
        writer = pd.ExcelWriter(result_file)
        df_cm = pd.DataFrame(confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)), columns=self.class_labels, index=self.class_labels)
        df_cm.to_excel(writer, sheet_name="Confusion_Matrix")
        df_eval = pd.DataFrame(classification_report(
            np.argmax(y_test, axis=1),
            np.argmax(y_pred, axis=1),
            target_names=self.class_labels,
            output_dict=True, zero_division=0
        )).T
        df_eval.to_excel(writer, sheet_name="Classification_Report")
        writer.close()
        K.clear_session()
        self.matrix = []
        self.eva_matrix = []
        self.trained = True
        print("📝 Evaluation saved. Testing complete.\n")
