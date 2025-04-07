"""
lmmd_loss.py

DESCRIPTION:
------------
Implementation of Local Maximum Mean Discrepancy (LMMD) loss function for domain adaptation
in Speech Emotion Recognition. This loss helps align feature distributions between source and
target domains (e.g., between EMODB and RAVDESS datasets) to improve cross-corpus performance.

LMMD extends standard MMD by considering class-conditional distributions, enabling class-aware
feature alignment between domains.

KEY COMPONENTS:
--------------
- `gaussian_kernel`: RBF kernel for measuring point-wise similarity in feature space
- `mmd_loss`: Standard Maximum Mean Discrepancy loss
- `lmmd_loss`: Local MMD loss that performs class-conditional alignment
- `get_lmmd_loss`: Function builder that returns a TensorFlow-compatible loss function

MATH FOUNDATION:
---------------
LMMD measures and minimizes the difference between the probability distributions
of source and target domains in a reproducing kernel Hilbert space (RKHS).

LMMD = sum_c weight_c * MMD^2(Ds_c, Dt_c)
- Ds_c: source domain examples of class c
- Dt_c: target domain examples of class c
- weight_c: weight for class c (based on sample count)
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    Calculate the RBF (Gaussian) kernel between source and target

    Args:
        source: source domain features, shape [batch_size, feature_dim]
        target: target domain features, shape [batch_size, feature_dim]
        kernel_mul: kernel multiplier for bandwidth
        kernel_num: number of kernels with different bandwidths
        fix_sigma: predefined bandwidth (if None, calculated from data)

    Returns:
        sum of kernels with different bandwidths
    """
    n_samples = tf.shape(source)[0] + tf.shape(target)[0]
    total = tf.concat([source, target], axis=0)

    # Calculate pairwise distances
    total0 = tf.expand_dims(total, axis=0)
    total1 = tf.expand_dims(total, axis=1)
    L2_distance = tf.reduce_sum(tf.math.squared_difference(total0, total1), axis=2)

    # Calculate kernel bandwidth
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = tf.reduce_mean(L2_distance)

    # Calculate kernels with different bandwidths
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = tf.zeros_like(L2_distance)

    for bandwidth in bandwidth_list:
        kernel_val += tf.exp(-L2_distance / (2 * bandwidth))

    return kernel_val


def mmd_loss(source_features, target_features, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    Calculate Maximum Mean Discrepancy (MMD) between source and target feature distributions

    Args:
        source_features: source domain features [batch_size, feature_dim]
        target_features: target domain features [batch_size, feature_dim]

    Returns:
        MMD distance value
    """
    batch_size = tf.shape(source_features)[0]

    # Calculate kernel values
    kernels = gaussian_kernel(
        source_features,
        target_features,
        kernel_mul=kernel_mul,
        kernel_num=kernel_num,
        fix_sigma=fix_sigma,
    )

    # Split kernels
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]

    # Calculate MMD
    loss = tf.reduce_mean(XX) + tf.reduce_mean(YY) - tf.reduce_mean(XY) - tf.reduce_mean(YX)

    return loss


def lmmd_loss(
    source_features,
    target_features,
    source_labels,
    target_pseudo_labels,
    num_classes,
    kernel_mul=2.0,
    kernel_num=5,
    fix_sigma=None,
):
    """
    Calculate Local Maximum Mean Discrepancy (LMMD) between source and target domains
    with class-conditional alignment

    Args:
        source_features: source domain features [batch_size, feature_dim]
        target_features: target domain features [batch_size, feature_dim]
        source_labels: one-hot encoded source labels [batch_size, num_classes]
        target_pseudo_labels: one-hot encoded target pseudo-labels [batch_size, num_classes]
        num_classes: number of classes

    Returns:
        LMMD loss value
    """
    batch_size = tf.shape(source_features)[0]

    # Initialize LMMD loss
    loss = 0.0

    # Handle case where classes have very few samples
    source_batch_size = tf.maximum(tf.shape(source_features)[0], 1)
    target_batch_size = tf.maximum(tf.shape(target_features)[0], 1)

    # Calculate class-specific loss for each class
    for c in range(num_classes):
        # Extract class-specific features using masks
        source_mask = source_labels[:, c]
        target_mask = target_pseudo_labels[:, c]

        # Check if we have samples for this class
        source_count = tf.reduce_sum(source_mask)
        target_count = tf.reduce_sum(target_mask)

        # Skip if no samples in either domain for this class
        if tf.equal(source_count, 0) or tf.equal(target_count, 0):
            continue

        # Expand dimensions for broadcasting
        source_mask = tf.expand_dims(source_mask, axis=1)
        target_mask = tf.expand_dims(target_mask, axis=1)

        # Apply mask to get class-specific features
        source_features_c = source_features * source_mask
        target_features_c = target_features * target_mask

        # Normalize to handle imbalance
        source_weight = 1.0 / (tf.maximum(source_count, 1) / tf.cast(source_batch_size, tf.float32))
        target_weight = 1.0 / (tf.maximum(target_count, 1) / tf.cast(target_batch_size, tf.float32))

        # Calculate class-conditional MMD
        mmd_c = mmd_loss(
            source_features_c,
            target_features_c,
            kernel_mul=kernel_mul,
            kernel_num=kernel_num,
            fix_sigma=fix_sigma,
        )

        # Add weighted class-conditional MMD to total loss
        loss += mmd_c * source_weight * target_weight

    # Normalize by number of classes
    loss = loss / tf.cast(num_classes, tf.float32)

    return loss


def get_lmmd_loss(num_classes, kernel_mul=2.0, kernel_num=5, fix_sigma=None, weight=1.0):
    """
    Create a TensorFlow-compatible loss function for LMMD

    Args:
        num_classes: number of emotion classes
        weight: weight for LMMD loss in the total loss

    Returns:
        loss function compatible with Keras
    """

    def loss_fn(
        y_true,
        y_pred,
        source_features,
        target_features,
        source_labels,
        target_pseudo_labels,
    ):
        # Calculate standard classification loss (categorical crossentropy)
        class_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)

        # Calculate LMMD loss
        adapt_loss = lmmd_loss(
            source_features,
            target_features,
            source_labels,
            target_pseudo_labels,
            num_classes,
            kernel_mul,
            kernel_num,
            fix_sigma,
        )

        # Return weighted sum
        return class_loss + (weight * adapt_loss)

    return loss_fn


# Keras custom loss function wrapper
class LMMDLoss(tf.keras.losses.Loss):
    def __init__(self, num_classes, weight=1.0, kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
        super(LMMDLoss, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.weight = weight
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        self.fix_sigma = fix_sigma

    def call(self, y_true, y_pred):
        # This is a placeholder - actual implementation requires source/target features and labels
        # which are provided during model training
        return tf.keras.losses.categorical_crossentropy(y_true, y_pred)

    def get_lmmd_loss_fn(self):
        return get_lmmd_loss(
            self.num_classes,
            self.kernel_mul,
            self.kernel_num,
            self.fix_sigma,
            self.weight,
        )
