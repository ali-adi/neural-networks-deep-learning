# model.py
"""
model.py

DESCRIPTION:
Implements a DTPM (Discriminant Temporal Pyramid Matching) + Attention-based
architecture for Speech Emotion Recognition (SER).

The model uses:
- Temporal convolution blocks with dilation for wide receptive fields (TCN style)
- Bidirectional modeling (forward and backward streams)
- Gated residual connections to preserve feature identity
- Weighted feature aggregation via learnable attention
- Cross-validation training and evaluation support

Key research-oriented features:
- Temporal modeling similar to TM-Net
- Attention-based aggregation (akin to self-attention)
- Label smoothing for generalization
- Excel output for evaluation metrics (saved in result_path/ under dataset-specific folders)
"""

# ---------------------------
# WARNING SUPPRESSION
# ---------------------------
import os
import warnings
warnings.filterwarnings("ignore")        # Suppress Python-level warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # Hide most TensorFlow logs (only errors remain)
# ---------------------------

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

# Evaluation tools
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import KFold

# ------------------------
# TEMPORAL BLOCK DEFINITION (DTPM Building Block)
# ------------------------
def temporal_block(inputs, dilation, activation, nb_filters, kernel_size, dropout_rate=0.0):
    """
    A single temporal block that uses dilated convolutions, gating, and residual connections.
    
    - 'dilation' indicates the temporal spacing for the convolutions, capturing multi-scale patterns.
    - 'skip_connection' references the input so we can add a residual path.
    - 'SpatialDropout1D' applies dropout across feature maps for better regularization.
    - The final gating (sigmoid) filters features, enabling DTPM to selectively highlight discriminative patterns.
    """
    skip_connection = inputs
    # 1st dilated convolution + batch norm + activation + dropout
    x = Conv1D(filters=nb_filters, kernel_size=kernel_size, dilation_rate=dilation, padding='causal')(inputs)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = SpatialDropout1D(dropout_rate)(x)

    # 2nd dilated convolution + batch norm + activation + dropout
    x = Conv1D(filters=nb_filters, kernel_size=kernel_size, dilation_rate=dilation, padding='causal')(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = SpatialDropout1D(dropout_rate)(x)

    # If the input channel dimension (skip_connection) doesn't match 'x', fix it with a 1x1 conv
    if skip_connection.shape[-1] != x.shape[-1]:
        skip_connection = Conv1D(filters=nb_filters, kernel_size=1, padding='same')(skip_connection)

    # Apply a gating mechanism with sigmoid, then multiply elementwise with skip connection
    gated = Lambda(sigmoid)(x)
    return skip_connection * gated

# ------------------------
# DILATED TEMPORAL NETWORK (DTPM + Bidirectional Streams)
# ------------------------
class TemporalConvNet:
    """
    Constructs a temporal pyramid using multiple dilation rates, in both forward and backward directions.
    This design is akin to DTPM for capturing multi-scale features, and TM-Net style for bidirectional streams.
    """
    def __init__(self, nb_filters=64, kernel_size=2, nb_stacks=1, dilations=8,
                 activation="relu", dropout_rate=0.1, name='TemporalConvNet'):
        # Initialize hyperparameters for the TCN
        self.name = name
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.nb_stacks = nb_stacks
        # 'dilations' can be int -> we do 2^i for i in range(dilations)
        self.dilations = dilations if isinstance(dilations, int) else 8
        self.activation = activation
        self.dropout_rate = dropout_rate

    def __call__(self, inputs):
        # 'forward' is the original input, 'backward' is reversed in time
        forward = inputs
        backward = Lambda(lambda x: tf.reverse(x, axis=[1]))(inputs)

        # 1x1 conv transformations for forward/backward to standardize dimensions
        f = Conv1D(self.nb_filters, kernel_size=1, padding='causal')(forward)
        b = Conv1D(self.nb_filters, kernel_size=1, padding='causal')(backward)

        # We'll collect pooled outputs (final_skips) after each dilation step
        final_skips = []

        # 'nb_stacks' allows repetition of the entire dilation cycle
        for _ in range(self.nb_stacks):
            # Dilation rates go as powers of 2 up to self.dilations
            for dilation_rate in [2 ** i for i in range(self.dilations)]:
                # Apply 'temporal_block' to the forward and backward streams
                f = temporal_block(f, dilation_rate, self.activation, self.nb_filters, self.kernel_size, self.dropout_rate)
                b = temporal_block(b, dilation_rate, self.activation, self.nb_filters, self.kernel_size, self.dropout_rate)

                # Merge forward + backward with an elementwise add
                merged = add([f, b])

                # GlobalAveragePooling to condense temporal dimension
                pooled = GlobalAveragePooling1D()(merged)

                # Expand dims so we can concatenate these across all dilation rates
                expanded = Lambda(lambda x: tf.expand_dims(x, axis=1))(pooled)

                # Accumulate each step into final_skips
                final_skips.append(expanded)

        # Concatenate all pyramid outputs along axis=1 (the newly expanded dimension)
        return Lambda(lambda x: tf.concat(x, axis=1))(final_skips)

# ------------------------
# WEIGHT LAYER (Attention-like Aggregation)
# ------------------------
class WeightLayer(tf.keras.layers.Layer):
    """
    A learnable layer that assigns weights to each timestep (pyramid level),
    effectively an attention-like mechanism that sums across the time dimension.
    """
    def build(self, input_shape):
        # Create a trainable kernel for weighting each time-step
        self.kernel = self.add_weight("kernel", shape=(input_shape[1], 1), initializer='uniform', trainable=True)

    def call(self, inputs):
        # Inputs shape: (batch, time_steps, features)
        # Transpose to (batch, features, time_steps) to multiply by 'kernel'
        x = tf.transpose(inputs, [0, 2, 1])
        # Weighted sum across the 'time_steps' dimension
        x = tf.matmul(x, self.kernel)
        # Squeeze out the last dim -> final shape: (batch, features)
        return tf.squeeze(x, axis=-1)

def smooth_labels(labels, factor=0.1):
    """
    Slightly reduces confidence in the true label, distributing it across other classes.
    Helps with generalization and reduces overfitting (label smoothing).
    """
    labels *= (1 - factor)
    labels += (factor / labels.shape[1])
    return labels

# ------------------------
# SPEECH EMOTION MODEL (DTPM + TM-Net–style + Cross-Validation)
# ------------------------
class SpeechEmotionModel:
    def __init__(self, input_shape, class_labels, args):
        """
        Initialize the SER model with hyperparameters, class labels, etc.
        Args:
          input_shape (tuple): shape of input features -> (time_steps, feature_dims)
          class_labels (tuple): all emotion classes
          args: command-line arguments holding model config
        """
        self.args = args
        self.input_shape = input_shape
        self.num_classes = len(class_labels)
        self.class_labels = class_labels
        self.model = None           # Will be created in create_model()
        self.matrix = []            # Store confusion matrices after each fold
        self.eva_matrix = []        # Store classification reports after each fold
        self.acc = 0                # Average accuracy across folds
        self.trained = False

        # Where to save results (organized by dataset)
        self.result_dir = os.path.join(self.args.result_path, self.args.data)
        # Timestamp for unique naming
        self.now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        # Track best fold accuracy and path to that fold's weights
        self.best_fold_acc = 0
        self.best_fold_weight_path = ""

        print(f"🧠 Initialized SER model with input shape: {input_shape}")

    def create_model(self):
        """
        Build the full model architecture:
          - Input layer
          - TemporalConvNet (multi-scale dilated TCN, bidirectional)
          - WeightLayer for attention-based aggregation
          - Dense softmax for final emotion classification
        """
        print("\n🛠️ Building model architecture...")

        # Input: shape = (time_steps, feature_dims)
        inputs = Input(shape=(self.input_shape[0], self.input_shape[1]))

        # Create the DTPM + TM-Net–style network (TemporalConvNet)
        conv_net = TemporalConvNet(
            nb_filters=self.args.filter_size,
            kernel_size=self.args.kernel_size,
            nb_stacks=self.args.stack_size,
            dilations=self.args.dilation_size,
            dropout_rate=self.args.dropout,
            activation=self.args.activation
        )

        # conv_output -> multi-scale pyramid of features
        conv_output = conv_net(inputs)

        # WeightLayer aggregates the pyramid using learnable weights (attention)
        attention_out = WeightLayer()(conv_output)

        # Final classification layer: softmax over emotion classes
        outputs = Dense(self.num_classes, activation='softmax')(attention_out)

        # Build Keras model
        self.model = KerasModel(inputs=inputs, outputs=outputs)
        optimizer = Adam(
            learning_rate=self.args.lr,
            beta_1=self.args.beta1,
            beta_2=self.args.beta2,
            epsilon=1e-8
        )
        self.model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
        print("✅ Model compiled successfully!\n")

    def train(self, x, y):
        """
        Train the model with k-fold cross-validation, saving the best weights per fold
        and computing metrics (confusion matrix, classification report).
        """
        print("🎯 Starting k-fold training...")
        save_dir = self.args.model_path
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)

        # test_model_dir: location for best fold's weights
        test_model_dir = os.path.join("test_models", self.args.data)
        os.makedirs(test_model_dir, exist_ok=True)

        # Prepare K-Fold
        kfold = KFold(n_splits=self.args.split_fold, shuffle=True, random_state=self.args.random_seed)
        avg_acc, avg_loss = 0, 0

        # For each fold in cross-validation
        for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(x, y), 1):
            print(f"🔁 Fold {fold_idx}/{self.args.split_fold}")
            self.create_model()

            # Apply label smoothing on the training set
            y_train_smoothed = smooth_labels(copy.deepcopy(y[train_idx]), 0.1)

            # Folder for storing weights of this fold
            fold_folder = os.path.join(save_dir, f"{self.args.data}_{self.args.random_seed}_{self.now}")
            os.makedirs(fold_folder, exist_ok=True)
            weight_name = f"{self.args.split_fold}-fold_weights_best_{fold_idx}.weights.h5"
            weight_path = os.path.join(fold_folder, weight_name)

            # Train the model on the current fold
            self.model.fit(
                x[train_idx], y_train_smoothed,
                validation_data=(x[test_idx], y[test_idx]),
                batch_size=self.args.batch_size,
                epochs=self.args.epoch,
                verbose=1,
                callbacks=[callbacks.ModelCheckpoint(weight_path, verbose=1, save_weights_only=True)]
            )

            # Load the saved weights for evaluation
            self.model.load_weights(weight_path)
            loss, acc = self.model.evaluate(x[test_idx], y[test_idx], verbose=0)
            avg_loss += loss
            avg_acc += acc

            print(f"✅ Fold {fold_idx} Accuracy: {round(acc * 100, 2)}%")

            # Track best fold
            if acc > self.best_fold_acc:
                self.best_fold_acc = acc
                self.best_fold_weight_path = weight_path

            # Prediction and metrics
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

        # Average metrics across folds
        self.acc = avg_acc / self.args.split_fold
        print(f"\n📊 Average Accuracy over {self.args.split_fold} folds: {round(self.acc * 100, 2)}%")

        # Save the best fold model to 'test_models'
        if self.best_fold_weight_path:
            best_name = os.path.basename(self.best_fold_weight_path)
            shutil.copy(self.best_fold_weight_path, os.path.join(test_model_dir, best_name))
            print(f"🏆 Best fold model ({round(self.best_fold_acc * 100, 2)}%) saved to test_models/{self.args.data}/{best_name}")

    def evaluate_test(self, x_test, y_test, path):
        """
        Evaluate the model on a separate test set:
          - Create model, load weights, evaluate on x_test, y_test.
          - Print test loss, test accuracy, classification report.
          - Save confusion matrix + classification report to Excel file.
        """
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

        result_file = os.path.join(self.result_dir, f"{self.args.data}_test_{round(acc * 100, 2)}_{self.args.random_seed}_{self.now}.xlsx")
        print(f"📁 Saving evaluation results to: {result_file}")
        writer = pd.ExcelWriter(result_file)

        # Confusion Matrix
        df_cm = pd.DataFrame(
            confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)),
            columns=self.class_labels,
            index=self.class_labels
        )
        df_cm.to_excel(writer, sheet_name="Confusion_Matrix")

        # Classification Report
        df_eval = pd.DataFrame(classification_report(
            np.argmax(y_test, axis=1),
            np.argmax(y_pred, axis=1),
            target_names=self.class_labels,
            output_dict=True,
            zero_division=0
        )).T
        df_eval.to_excel(writer, sheet_name="Classification_Report")

        writer.close()
        K.clear_session()
        self.matrix = []
        self.eva_matrix = []
        self.trained = True
        print("📝 Evaluation saved. Testing complete.\n")
