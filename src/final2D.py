

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.callbacks import ReduceLROnPlateau

import pickle
from tensorflow.keras.callbacks import Callback
from tqdm import tqdm
#import scipy.ndimage as ndi
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
#import math
#import nibabel as nib
#import skimage.transform
import pathlib
#import pandas as pd
#from scipy import ndimage
#import skimage.feature, skimage.measure, skimage.morphology, skimage.segmentation
import glob
import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, \
    Conv2D, MaxPooling2D, ZeroPadding2D, Input, Embedding, \
    Lambda, UpSampling2D, Cropping2D, Conv2DTranspose, concatenate

from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, CSVLogger
# from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras import optimizers
from sklearn.model_selection import train_test_split
#from skimage.transform import resize
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras.utils import array_to_img, Sequence
import random
import tensorflow.keras.backend as K

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import scipy

# def focal_tversky_loss(alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-5):
#     def loss(y_true, y_pred):
#         y_true = tf.cast(y_true, tf.float32)
#         y_pred = tf.cast(y_pred, tf.float32)

#         tp = tf.reduce_sum(y_true * y_pred)
#         fp = tf.reduce_sum(y_pred) - tp
#         fn = tf.reduce_sum(y_true) - tp

#         tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
#         focal_tversky = tf.pow((1 - tversky), gamma)

#         return focal_tversky
#     return loss


def binary_cross_entropy_loss():
    def loss(y_true, y_pred):
        return tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)

    return loss


# def dice_loss(y_true, y_pred, smooth=1e-6):
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)
#     return 1 - (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# def focal_loss(y_true, y_pred, gamma=2):
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     epsilon = K.epsilon()
#     y_pred_f = K.clip(y_pred_f, epsilon, 1.0 - epsilon)
#     return -K.mean(y_true_f * K.pow(1 - y_pred_f, gamma) * K.log(y_pred_f))

# def dice_focal_loss(y_true, y_pred, lambda_weight=0.7, gamma=2):
#     d_loss = dice_loss(y_true, y_pred)
#     f_loss = focal_loss(y_true, y_pred, gamma=gamma)
#     return lambda_weight * d_loss + (1 - lambda_weight) * f_loss

def iou_coef(y_true, y_pred, smooth=1e-6):
    """
    Intersection over Union (IoU) metric for Keras.
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

#def dice_loss(y_true, y_pred, smooth=1):
    #y_true = tf.cast(y_true, 'float32')
    #y_pred = tf.cast(y_pred, 'float32')
    #y_true_f = tf.keras.backend.flatten(y_true)
    #y_pred_f = tf.keras.backend.flatten(y_pred)
    #intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    #return 1 - (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

#def focal_loss(y_true, y_pred, gamma=2):
#    y_true_f = tf.keras.backend.flatten(y_true)
#    y_pred_f = tf.keras.backend.flatten(y_pred)
#    focal = -y_true_f * tf.keras.backend.pow((1 - y_pred_f), gamma) * tf.keras.backend.log(y_pred_f)
#    return tf.keras.backend.mean(focal)

#def combined_dice_focal_loss(y_true, y_pred, lambda_weight=0.5, gamma=2, smooth=1):
#    dice = dice_loss(y_true, y_pred, smooth)
#    focal = focal_loss(y_true, y_pred, gamma)
#    return lambda_weight * dice + (1 - lambda_weight) * focal

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l2


# Dice Coefficient Metric

def dice_coef(y_true, y_pred, smooth=1e-6):
    """
    Dice coefficient metric for Keras.
    Usage: metrics=[dice_coef]
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# Sensitivity (Recall) Metric
def sensitivity_metric(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    true_positives = K.sum(y_true_f * y_pred_f)
    possible_positives = K.sum(y_true_f)
    return true_positives / (possible_positives + K.epsilon())

# Precision Metric
def precision_metric(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    true_positives = K.sum(y_true_f * y_pred_f)
    predicted_positives = K.sum(y_pred_f)
    return true_positives / (predicted_positives + K.epsilon())


def dice_loss(y_true, y_pred, smooth=1e-6):
    """
    Dice loss for Keras.
    Usage: loss=dice_loss
    """
    return 1 - dice_coef(y_true, y_pred, smooth)


# def DiceBCELoss(y_true, y_pred, smooth=1e-6):
#     # Cast to float32 data type for compatibility
#     y_true = tf.cast(y_true, 'float32')
#     y_pred = tf.cast(y_pred, 'float32')

#     # Flatten the tensors
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)

#     # Calculate Binary Cross-Entropy Loss
#     BCE = K.binary_crossentropy(y_true_f, y_pred_f)

#     # Calculate Dice Loss
#     dice = dice_coef_loss(y_true_f, y_pred_f)
#     dice_loss = 1 - dice

#     # Combine BCE and Dice Loss
#     Dice_BCE = BCE + dice_loss

#     return Dice_BCE

#def dice_loss(y_true, y_pred, smooth=1e-6):
#    y_true_f = K.flatten(y_true)
#    y_pred_f = K.flatten(y_pred)
#    intersection = K.sum(y_true_f * y_pred_f)
#    return 1 - (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_bce_loss(y_true, y_pred):
    dice = dice_loss(y_true, y_pred)
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return 0.7 * dice + 0.3 * bce

def sensitivity_specific_dice_loss(y_true, y_pred, beta=2, smooth=1):
    # Flatten the tensors
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)

    # Calculate True Positives, False Positives, and False Negatives
    TP = tf.keras.backend.sum(y_true_f * y_pred_f)
    FP = tf.keras.backend.sum(y_pred_f * (1 - y_true_f))
    FN = tf.keras.backend.sum((1 - y_pred_f) * y_true_f)

    # Calculate the Recall-Weighted Dice Coefficient
    weighted_dice = (1 + beta) * TP + smooth
    denominator = (1 + beta) * TP + FP + beta * FN + smooth
    dice_loss = 1 - (weighted_dice / denominator)

    return dice_loss

#convolutional block
def conv_block(x, kernelsize, filters, dropout, batchnorm=False):
    conv = layers.Conv2D(filters, (kernelsize, kernelsize),
                         kernel_initializer='he_normal', padding="same"
                        )(x)
    if batchnorm:
        conv = layers.BatchNormalization(axis=3)(conv)
    conv = layers.Activation("relu")(conv)
    if dropout > 0:
        conv = layers.Dropout(dropout)(conv)
    conv = layers.Conv2D(filters, (kernelsize, kernelsize),
                         kernel_initializer='he_normal', padding="same")(conv)
    if batchnorm:
        conv = layers.BatchNormalization(axis=3)(conv)
    conv = layers.Activation("relu")(conv)
    return conv



#residual convolutional block
def res_conv_block(x, kernelsize, filters, dropout, batchnorm=False):
    # First conv: add L2 regularization
    conv1 = layers.Conv2D(filters, (kernelsize, kernelsize),
                          kernel_initializer='he_normal', padding='same')(x)
    if batchnorm:
        conv1 = layers.BatchNormalization(axis=3)(conv1)
    conv1 = layers.Activation('relu')(conv1)

    # Second conv: also add L2 regularization
    conv2 = layers.Conv2D(filters, (kernelsize, kernelsize),
                          kernel_initializer='he_normal', padding='same')(conv1)
    if batchnorm:
        conv2 = layers.BatchNormalization(axis=3)(conv2)
        conv2 = layers.Activation('relu')(conv2)
    if dropout > 0:
        conv2 = layers.Dropout(dropout)(conv2)

    # Shortcut path: a 1x1 convolution with L2 regularization
    shortcut = layers.Conv2D(filters, kernel_size=(1, 1),
                             kernel_initializer='he_normal', padding='same')(x)
    if batchnorm:
        shortcut = layers.BatchNormalization(axis=3)(shortcut)
    shortcut = layers.Activation("relu")(shortcut)

    # Add the shortcut connection and return the result
    respath = layers.add([shortcut, conv2])
    return respath



#gating signal for attention unit
def gatingsignal(input, out_size, batchnorm=False):
    x = layers.Conv2D(out_size, (1, 1), padding='same')(input)
    if batchnorm:

        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

#attention unit/block based on soft attention
def attention_block(x, gating, inter_shape):
    shape_x = K.int_shape(x)
    shape_g = K.int_shape(gating)
    theta_x = layers.Conv2D(inter_shape, (2, 2), strides=(2, 2), kernel_initializer='he_normal', padding='same')(x)
    shape_theta_x = K.int_shape(theta_x)
    phi_g = layers.Conv2D(inter_shape, (1, 1), kernel_initializer='he_normal', padding='same')(gating)
    upsample_g = layers.Conv2DTranspose(inter_shape, (3, 3), strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]), kernel_initializer='he_normal', padding='same')(phi_g)
    concat_xg = layers.add([upsample_g, theta_x])
    act_xg = layers.Activation('relu')(concat_xg)
    psi = layers.Conv2D(1, (1, 1), kernel_initializer='he_normal', padding='same')(act_xg)
    sigmoid_xg = layers.Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(sigmoid_xg)
    upsample_psi = layers.UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)
    upsample_psi = layers.Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3), arguments={'repnum': shape_x[3]})(upsample_psi)
    y = layers.multiply([upsample_psi, x])
    result = layers.Conv2D(shape_x[3], (1, 1), kernel_initializer='he_normal', padding='same')(y)
    attenblock = layers.BatchNormalization()(result)
    return attenblock

def build_att_res_unet(input_shape=(256, 256, 1),
                       filters=[32, 64, 128, 256, 512],
                       kernelsize=3, upsample_size=2, dropout=0.5,
                       weights_path=None, batchnorm=True):
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Downsampling layers with regularization applied within each res_conv_block
    dn_1 = res_conv_block(inputs, kernelsize, filters[0], dropout, batchnorm)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(dn_1)

    dn_2 = res_conv_block(pool1, kernelsize, filters[1], dropout, batchnorm)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(dn_2)

    dn_3 = res_conv_block(pool2, kernelsize, filters[2], dropout, batchnorm)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(dn_3)

    dn_4 = res_conv_block(pool3, kernelsize, filters[3], dropout, batchnorm)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(dn_4)

    dn_5 = res_conv_block(pool4, kernelsize, filters[4], dropout, batchnorm)

    # Upsampling layers: (you can also add regularization in the upsampling blocks if desired)
    gating_5 = gatingsignal(dn_5, filters[3], batchnorm)
    att_5 = attention_block(dn_4, gating_5, filters[3])
    up_5 = layers.UpSampling2D(size=(upsample_size, upsample_size), data_format="channels_last")(dn_5)
    up_5 = layers.concatenate([up_5, att_5], axis=3)
    up_conv_5 = res_conv_block(up_5, kernelsize, filters[3], dropout, batchnorm)

    gating_4 = gatingsignal(up_conv_5, filters[2], batchnorm)
    att_4 = attention_block(dn_3, gating_4, filters[2])
    up_4 = layers.UpSampling2D(size=(upsample_size, upsample_size), data_format="channels_last")(up_conv_5)
    up_4 = layers.concatenate([up_4, att_4], axis=3)
    up_conv_4 = res_conv_block(up_4, kernelsize, filters[2], dropout, batchnorm)

    gating_3 = gatingsignal(up_conv_4, filters[1], batchnorm)
    att_3 = attention_block(dn_2, gating_3, filters[1])
    up_3 = layers.UpSampling2D(size=(upsample_size, upsample_size), data_format="channels_last")(up_conv_4)
    up_3 = layers.concatenate([up_3, att_3], axis=3)
    up_conv_3 = res_conv_block(up_3, kernelsize, filters[1], dropout, batchnorm)

    gating_2 = gatingsignal(up_conv_3, filters[0], batchnorm)
    att_2 = attention_block(dn_1, gating_2, filters[0])
    up_2 = layers.UpSampling2D(size=(upsample_size, upsample_size), data_format="channels_last")(up_conv_3)
    up_2 = layers.concatenate([up_2, att_2], axis=3)
    up_conv_2 = res_conv_block(up_2, kernelsize, filters[0], dropout, batchnorm)

    conv_final = layers.Conv2D(1, kernel_size=(1, 1), kernel_initializer='he_normal', padding="same")(up_conv_2)
    conv_final = layers.BatchNormalization(axis=3)(conv_final)
    outputs = layers.Activation('sigmoid')(conv_final)

    model = models.Model(inputs=inputs, outputs=outputs)

###########################################################
    if weights_path:
        try:
            model.load_weights(weights_path)
            print(f"Weights loaded from: {weights_path}")
        except Exception as e:
            print(f"Error loading weights: {e}")
#########################################################
    return model

class LungNoduleDataGenerator2(tf.keras.utils.Sequence):
    def __init__(self, pos_ct_paths, pos_mask_paths, empty_ct_paths, empty_mask_paths,
                 batch_size=16, empty_ratio=0.25, shuffle=True, augment=False):
        self.pos_ct_paths = pos_ct_paths
        self.pos_mask_paths = pos_mask_paths
        self.empty_ct_paths = empty_ct_paths
        self.empty_mask_paths = empty_mask_paths
        self.batch_size = batch_size
        self.empty_ratio = empty_ratio
        self.shuffle = shuffle
        self.augment = augment



    def __len__(self):
        return len(self.pos_ct_paths) // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            # You can shuffle the lists, but DON'T shorten them!
            pass

    def __getitem__(self, index):
        n_empty = int(self.batch_size * self.empty_ratio)
        n_pos = self.batch_size - n_empty

        if len(self.pos_ct_paths) == 0 or len(self.empty_ct_paths) == 0:
            print("ERROR: Data lists are empty!")
            x_batch = np.zeros((self.batch_size, 256, 256, 1), dtype=np.float32)
            y_batch = np.zeros((self.batch_size, 256, 256, 1), dtype=np.float32)
            return x_batch, y_batch

        pos_indices = np.random.choice(len(self.pos_ct_paths), n_pos, replace=True)
        empty_indices = np.random.choice(len(self.empty_ct_paths), n_empty, replace=True)
        x_batch, y_batch = [], []

        for i in pos_indices:
            x = np.load(self.pos_ct_paths[i]).astype(np.float32)
            y = np.load(self.pos_mask_paths[i]).astype(np.float32)
            if self.augment:
                x, y = self.apply_light_augmentation(x, y)
            x = np.expand_dims(x, -1)
            y = np.expand_dims(y, -1)
            x_batch.append(x)
            y_batch.append(y)
        for i in empty_indices:
            x = np.load(self.empty_ct_paths[i]).astype(np.float32)
            y = np.load(self.empty_mask_paths[i]).astype(np.float32)
            if self.augment:
                x, y = self.apply_light_augmentation(x, y)
            x = np.expand_dims(x, -1)
            y = np.expand_dims(y, -1)
            x_batch.append(x)
            y_batch.append(y)
        idx = np.random.permutation(len(x_batch))
        x_batch = np.array(x_batch)[idx]
        y_batch = np.array(y_batch)[idx]
        return x_batch, y_batch



    def apply_light_augmentation(self, x, y):
        # Random horizontal flip (20% probability)
        if random.random() < 0.2:
            x = np.fliplr(x)
            y = np.fliplr(y)
        # Random rotation (20% probability)
        if random.random() < 0.2:
            import cv2
            angle = random.uniform(-20, 20)
            matrix = cv2.getRotationMatrix2D((x.shape[1] / 2, x.shape[0] / 2), angle, 1)
            x = cv2.warpAffine(x, matrix, (x.shape[1], x.shape[0]), flags=cv2.INTER_LINEAR)
            y = cv2.warpAffine(y, matrix, (y.shape[1], y.shape[0]), flags=cv2.INTER_NEAREST)
        return x, y


# --- Make sure to import or define your custom data generator class above! ---

if __name__ == "__main__":
    # Paths to positive CT and masks
    path = "/gpfs/ddn/users/zafara/arman/Data/Corrected_Sliced/CT/"
    mask_path = "/gpfs/ddn/users/zafara/arman/Data/Corrected_Sliced/Mask/"
    output_dir = "/gpfs/ddn/users/zafara/arman/"
    os.makedirs(output_dir, exist_ok=True)

    ct_paths = sorted(glob.glob(os.path.join(path, "*.npy")))
    mask_paths = sorted(glob.glob(os.path.join(mask_path, "*.npy")))
    assert len(ct_paths) == len(mask_paths), "Mismatch between CT and mask files!"

    # Paths to empty CT and masks
    empty_ct_folder = "/gpfs/ddn/users/zafara/arman/Data/Corrected_Sliced/SelectedEmptySlices/CT/"
    empty_mask_folder = "/gpfs/ddn/users/zafara/arman/Data/Corrected_Sliced/SelectedEmptySlices/Mask/"
    empty_ct_paths = sorted(glob.glob(os.path.join(empty_ct_folder, "*.npy")))
    empty_mask_paths = sorted(glob.glob(os.path.join(empty_mask_folder, "*.npy")))
    assert len(empty_ct_paths) == len(empty_mask_paths), "Mismatch between empty CT and mask files!"

    # Split positive data by patient
    patient_ids = list(set([os.path.basename(f).split('_')[0] for f in ct_paths]))
    train_patient_ids, val_patient_ids = train_test_split(
        patient_ids, test_size=0.1, random_state=32, shuffle=True
    )
    train_ct_paths = [f for f in ct_paths if os.path.basename(f).split('_')[0] in train_patient_ids]
    train_mask_paths = [f for f in mask_paths if os.path.basename(f).split('_')[0] in train_patient_ids]
    val_ct_paths = [f for f in ct_paths if os.path.basename(f).split('_')[0] in val_patient_ids]
    val_mask_paths = [f for f in mask_paths if os.path.basename(f).split('_')[0] in val_patient_ids]

    # Split empty data randomly (since they're not patient-dependent)
    train_empty_ct_paths, val_empty_ct_paths = train_test_split(
        empty_ct_paths, test_size=0.1, random_state=32, shuffle=True
    )
    train_empty_mask_paths = [f.replace('/CT/', '/Mask/') for f in train_empty_ct_paths]
    val_empty_mask_paths = [f.replace('/CT/', '/Mask/') for f in val_empty_ct_paths]

    # Sanity check: All pairs match in basename
    for ct, mask in zip(train_ct_paths, train_mask_paths):
        assert os.path.basename(ct).replace("CT", "") == os.path.basename(mask).replace("Mask", ""), f"Mismatch: {ct}, {mask}"

    for ct, mask in zip(train_empty_ct_paths, train_empty_mask_paths):
        assert os.path.basename(ct).replace("CT", "") == os.path.basename(mask).replace("Mask", ""), f"Mismatch: {ct}, {mask}"

    # Model setup
    learning_rate = 0.0001
    batch_size = 16
    epochs = 500

    lr_schedule = CosineDecay(initial_learning_rate=0.0001, decay_steps=10, alpha=1e-4)

    model = build_att_res_unet()
    opt = optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss=dice_bce_loss,
                  metrics=[dice_coef, iou_coef, sensitivity_metric, precision_metric])
    model.summary()

    filepath = os.path.join(output_dir, "modelwithempty_Lr0.0001-data25%empty-1-diceBCEloss-bs16.h5.keras")
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    
    
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',        # or 'val_dice_coef' if you prefer
        factor=0.5,
        patience=15,               # number of epochs with no improvement to wait
        verbose=1,
        min_lr=1e-6
    )
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=25,
        restore_best_weights=True,
        verbose=1
    )

    # Generators (pass both CT and mask paths for positives and empties!)
    train_generator = LungNoduleDataGenerator2(
        train_ct_paths, train_mask_paths,
        train_empty_ct_paths, train_empty_mask_paths,
        batch_size=batch_size, empty_ratio=0.25, shuffle=True, augment=True
    )
    val_generator = LungNoduleDataGenerator2(
        val_ct_paths, val_mask_paths,
        val_empty_ct_paths, val_empty_mask_paths,
        batch_size=batch_size, empty_ratio=0.25, shuffle=False, augment=False
    )

    # Optional: Sanity check a batch
    x, y = train_generator[0]
    print("Train batch shape:", x.shape, y.shape)
    print("Num empty masks in batch:", sum(np.sum(mask) == 0 for mask in y))
    print("Num non-empty masks in batch:", batch_size - sum(np.sum(mask) == 0 for mask in y))

    # Training
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        verbose=1,
        callbacks=[checkpoint, reduce_lr]
    )

# Plot Loss
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss vs. Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "Dicesenloss_best.png"), dpi=200)
plt.show()

# Plot Dice Coefficient
plt.figure(figsize=(8, 5))
plt.plot(history.history['dice_coef'], label='Training')
plt.plot(history.history['val_dice_coef'], label='Validation')
plt.title('Dice Coefficient vs. Epochs')
plt.xlabel('Epoch')
plt.ylabel('Dice Coefficient')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "Dicesen_best.png"), dpi=200)
plt.show()
