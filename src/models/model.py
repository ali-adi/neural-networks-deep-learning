# model.py

"""
model.py

DESCRIPTION:
Implements a temporal convolution-based approach for speech emotion recognition,
including cross-validation, checkpointing, confusion matrices, and metrics.
"""

import os
# --- WARNING SUPPRESSION ---
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Hide all TensorFlow warnings
import warnings
warnings.filterwarnings("ignore")         # Hide Python-level warnings
# ---------------------------

import numpy as np
import datetime
import copy
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.layers import (
    Conv1D,
    SpatialDropout1D,
    BatchNormalization,
    Activation,
    add,
    GlobalAveragePooling1D,
    Lambda,
    Dense,
    Input,
)
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import callbacks
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.utils import to_categorical

def temporal_block(
    x,
    dilation,
    activation,
    nb_filters,
    kernel_size,
    dropout_rate=0.0
):
    original_x = x

    # 1st conv
    conv_1 = Conv1D(filters=nb_filters,
                    kernel_size=kernel_size,
                    dilation_rate=dilation,
                    padding='causal')(x)
    conv_1 = BatchNormalization(trainable=True, axis=-1)(conv_1)
    conv_1 = Activation(activation)(conv_1)
    conv_1 = SpatialDropout1D(dropout_rate)(conv_1)

    # 2nd conv
    conv_2 = Conv1D(filters=nb_filters,
                    kernel_size=kernel_size,
                    dilation_rate=dilation,
                    padding='causal')(conv_1)
    conv_2 = BatchNormalization(trainable=True, axis=-1)(conv_2)
    conv_2 = Activation(activation)(conv_2)
    conv_2 = SpatialDropout1D(dropout_rate)(conv_2)

    # Dim match
    if original_x.shape[-1] != conv_2.shape[-1]:
        original_x = Conv1D(filters=nb_filters,
                            kernel_size=1,
                            padding='same')(original_x)

    # Gating
    conv_2 = Lambda(sigmoid)(conv_2)
    gated_output = original_x * conv_2

    return gated_output

class TemporalConvNet:
    def __init__(
        self,
        nb_filters=64,
        kernel_size=2,
        nb_stacks=1,
        dilations=None,
        activation="relu",
        dropout_rate=0.1,
        name='TemporalConvNet'
    ):
        self.name = name
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.dilations = dilations if dilations is not None else 8
        self.nb_stacks = nb_stacks
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters

    def __call__(self, inputs):
        forward = inputs
        backward = Lambda(lambda x: tf.reverse(x, axis=[1]))(inputs)

        # Initial 1x1 conv
        forward_conv = Conv1D(filters=self.nb_filters, kernel_size=1,
                              dilation_rate=1, padding='causal')(forward)
        backward_conv = Conv1D(filters=self.nb_filters, kernel_size=1,
                               dilation_rate=1, padding='causal')(backward)

        final_skips = []
        skip_forward = forward_conv
        skip_backward = backward_conv

        for _ in range(self.nb_stacks):
            for d_pow in [2 ** i for i in range(self.dilations)]:
                skip_forward = temporal_block(
                    skip_forward,
                    dilation=d_pow,
                    activation=self.activation,
                    nb_filters=self.nb_filters,
                    kernel_size=self.kernel_size,
                    dropout_rate=self.dropout_rate
                )
                skip_backward = temporal_block(
                    skip_backward,
                    dilation=d_pow,
                    activation=self.activation,
                    nb_filters=self.nb_filters,
                    kernel_size=self.kernel_size,
                    dropout_rate=self.dropout_rate
                )

                merged_skip = add([skip_forward, skip_backward])
                merged_skip = GlobalAveragePooling1D()(merged_skip)
                merged_skip = Lambda(lambda x: tf.expand_dims(x, axis=1))(merged_skip)
                final_skips.append(merged_skip)

        # Combine all skip outputs
        output = Lambda(lambda tensors: tf.concat(tensors, axis=-2))(final_skips)
        return output

class WeightLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(WeightLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_shape[1], 1),
            initializer='uniform',
            trainable=True
        )
        super(WeightLayer, self).build(input_shape)

    def call(self, inputs):
        transposed = tf.transpose(inputs, [0, 2, 1])
        x = tf.matmul(transposed, self.kernel)
        return tf.squeeze(x, axis=-1)

def smooth_labels(labels, factor=0.1):
    labels *= (1 - factor)
    labels += (factor / labels.shape[1])
    return labels

class SpeechEmotionModel:
    def __init__(self, input_shape, class_labels, args):
        self.args = args
        self.data_shape = input_shape
        self.num_classes = len(class_labels)
        self.class_labels = class_labels
        self.model = None
        self.matrix = []
        self.eva_matrix = []
        self.acc = 0
        self.trained = False

        print(f"🧠 Initialized SpeechEmotionModel with input shape: {input_shape}")

    def create_model(self):
        print("\n🛠️ Building model architecture...")
        inputs = Input(shape=(self.data_shape[0], self.data_shape[1]))

        conv_net = TemporalConvNet(
            nb_filters=self.args.filter_size,
            kernel_size=self.args.kernel_size,
            nb_stacks=self.args.stack_size,
            dilations=self.args.dilation_size,
            dropout_rate=self.args.dropout,
            activation=self.args.activation
        )
        conv_out = conv_net(inputs)
        weighted = WeightLayer()(conv_out)
        outputs = Dense(self.num_classes, activation='softmax')(weighted)

        self.model = KerasModel(inputs=inputs, outputs=outputs)

        optimizer = Adam(
            learning_rate=self.args.lr,
            beta_1=self.args.beta1,
            beta_2=self.args.beta2,
            epsilon=1e-8
        )
        self.model.compile(
            loss="categorical_crossentropy",
            optimizer=optimizer,
            metrics=['accuracy']
        )
        print("✅ Model compiled successfully!\n")

    def train(self, x, y):
        print("🎯 Starting training loop with cross-validation...\n")
        filepath = self.args.model_path
        resultpath = self.args.result_path
        os.makedirs(filepath, exist_ok=True)
        os.makedirs(resultpath, exist_ok=True)

        from sklearn.model_selection import KFold
        kfold = KFold(n_splits=self.args.split_fold, shuffle=True, random_state=self.args.random_seed)
        avg_accuracy, avg_loss = 0, 0
        i = 1
        now_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        for train_idx, test_idx in kfold.split(x, y):
            print(f"🔁 Fold {i} of {self.args.split_fold}")
            self.create_model()
            y_train_smooth = smooth_labels(copy.deepcopy(y[train_idx]), 0.1)

            fold_folder = os.path.join(filepath, f"{self.args.data}_{self.args.random_seed}_{now_time}")
            os.makedirs(fold_folder, exist_ok=True)

            weight_path = os.path.join(fold_folder, f"{self.args.split_fold}-fold_weights_best_{i}.weights.h5")

            checkpoint = callbacks.ModelCheckpoint(
                weight_path,
                verbose=1,
                save_weights_only=True,
                save_best_only=False
            )
            self.model.fit(
                x[train_idx], y_train_smooth,
                validation_data=(x[test_idx], y[test_idx]),
                batch_size=self.args.batch_size,
                epochs=self.args.epoch,
                verbose=1,
                callbacks=[checkpoint]
            )

            print("📊 Evaluating model...")
            self.model.load_weights(weight_path)
            best_eva_list = self.model.evaluate(x[test_idx], y[test_idx], verbose=0)
            avg_loss += best_eva_list[0]
            avg_accuracy += best_eva_list[1]

            print(f"✅ Fold {i} Accuracy: {round(best_eva_list[1] * 100, 2)}%")

            y_pred_best = self.model.predict(x[test_idx])
            from sklearn.metrics import confusion_matrix, classification_report
            self.matrix.append(confusion_matrix(np.argmax(y[test_idx], axis=1),
                                                np.argmax(y_pred_best, axis=1)))

            em = classification_report(
                np.argmax(y[test_idx], axis=1),
                np.argmax(y_pred_best, axis=1),
                target_names=self.class_labels,
                output_dict=True,
                zero_division=0
            )
            self.eva_matrix.append(em)

            print(classification_report(
                np.argmax(y[test_idx], axis=1),
                np.argmax(y_pred_best, axis=1),
                target_names=self.class_labels,
                zero_division=0
            ))
            i += 1

        print(f"\n📈 Average Accuracy: {round(avg_accuracy / self.args.split_fold * 100, 2)}%")
        self.acc = avg_accuracy / self.args.split_fold

        result_filename = f"{self.args.data}_{self.args.split_fold}fold_{round(self.acc*100, 2)}_{self.args.random_seed}_{now_time}.xlsx"
        result_filepath = os.path.join(resultpath, result_filename)
        writer = pd.ExcelWriter(result_filepath)

        for j, matrix_data in enumerate(self.matrix):
            temp_dict = {" ": self.class_labels}
            for c_idx, row in enumerate(matrix_data):
                temp_dict[self.class_labels[c_idx]] = row
            df_cm = pd.DataFrame(temp_dict)
            df_cm.to_excel(writer, sheet_name=str(j))

            df_eval = pd.DataFrame(self.eva_matrix[j]).transpose()
            df_eval.to_excel(writer, sheet_name=str(j) + "_evaluate")

        writer.close()

        K.clear_session()
        self.matrix = []
        self.eva_matrix = []
        self.acc = 0
        self.trained = True
        print("📝 Evaluation results saved!\n")

    def test(self, x, y, path):
        print(f"\n🔎 Beginning model evaluation on test set...")
        from sklearn.model_selection import KFold
        kfold = KFold(n_splits=self.args.split_fold, shuffle=True, random_state=self.args.random_seed)
        avg_accuracy, avg_loss = 0, 0
        i = 1
        x_feats, y_labels = [], []

        for train_idx, test_idx in kfold.split(x, y):
            print(f"📁 Testing Fold {i}")
            self.create_model()

            weight_path = os.path.join(path, f"{self.args.split_fold}-fold_weights_best_{i}.weights.h5")

            self.model.fit(
                x[train_idx], y[train_idx],
                validation_data=(x[test_idx], y[test_idx]),
                batch_size=self.args.batch_size,
                epochs=0,
                verbose=0
            )

            self.model.load_weights(weight_path)
            best_eva_list = self.model.evaluate(x[test_idx], y[test_idx], verbose=0)
            avg_loss += best_eva_list[0]
            avg_accuracy += best_eva_list[1]

            print(f"✅ Fold {i} Accuracy: {round(best_eva_list[1] * 100, 2)}%")

            y_pred_best = self.model.predict(x[test_idx])
            from sklearn.metrics import confusion_matrix, classification_report
            self.matrix.append(confusion_matrix(np.argmax(y[test_idx], axis=1),
                                                np.argmax(y_pred_best, axis=1)))

            em = classification_report(
                np.argmax(y[test_idx], axis=1),
                np.argmax(y_pred_best, axis=1),
                target_names=self.class_labels,
                output_dict=True,
                zero_division=0
            )
            self.eva_matrix.append(em)

            print(classification_report(
                np.argmax(y[test_idx], axis=1),
                np.argmax(y_pred_best, axis=1),
                target_names=self.class_labels,
                zero_division=0
            ))

            feature_extractor = KerasModel(
                inputs=self.model.input,
                outputs=self.model.layers[-2].output
            )
            feats = feature_extractor.predict(x[test_idx])
            x_feats.append(feats)
            y_labels.append(y[test_idx])

            i += 1

        print(f"\n📊 Average Accuracy across all folds: {round(avg_accuracy / self.args.split_fold * 100, 2)}%")
        self.acc = avg_accuracy / self.args.split_fold
        return x_feats, y_labels
