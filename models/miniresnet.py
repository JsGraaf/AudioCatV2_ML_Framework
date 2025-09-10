# /*---------------------------------------------------------------------------------------------
#  * Copyright 2015 The TensorFlow Authors.
#  * Copyright (c) 2022 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import os
import sys
import warnings
from pathlib import Path

import tensorflow as tf
from keras import layers, regularizers
from keras.applications import ResNet50
from keras.models import Model, load_model

"""NOTE : Most of this implementation is adapted from the Tensorflow implementation of ResNets."""
# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import tensorflow as tf
from tensorflow import keras


def add_head(
    n_classes,
    backbone,
    add_flatten=True,
    trainable_backbone=True,
    activation=None,
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    functional=True,
    dropout=0,
):
    """
    Adds a classification head to a backbone.
    This classification head consists of a dense layer and an activation function.

    Inputs
    ------
    n_classes : int, number of neurons in the classification head
    backbone : tf.keras.Model : Backbone upon which to add the classification head
    add_flatten : bool, if True add a Flatten layer between the backbone and the head
    trainable_backbone : bool, if True unfreeze all backbone weights.
        If False freeze all backbone weights
    activation : str, activation function of the classification head. Usually "softmax" or "sigmoid"
    kernel_regularizer : tf.keras.regularizers.Regularizer,
        kernel regularizer to add to the classification head.
    bias_regularizer : tf.keras.regularizers.Regularizer,
        bias regularizer to add to the classification head.
    activity_regularizer : tf.keras.regularizers.Regularizer,
        activity regularizer to add to the classification head.
    functional : bool, if True return a tf.keras functional (as opposed to Sequential) model.
        Recommended.
    dropout : float, if >0 adds dropout to the classification head with the specified rate.

    Outputs
    -------
    model : Model with the attached classification head

    """
    if functional:
        if not trainable_backbone:
            for layer in backbone.layers:
                layer.trainable = False
        x = backbone.output
        if add_flatten:
            x = tf.keras.layers.Flatten()(x)
        if dropout:
            x = tf.keras.layers.Dropout(dropout)(x)
        if activation is None:
            out = tf.keras.layers.Dense(
                units=n_classes,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer,
                name="new_head",
            )(x)
        else:
            out = tf.keras.layers.Dense(
                units=n_classes,
                activation=activation,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer,
                name="new_head",
            )(x)
        func_model = tf.keras.models.Model(inputs=backbone.input, outputs=out)
        return func_model

    else:

        if not trainable_backbone:
            for layer in backbone.layers:
                layer.trainable = False
        seq_model = tf.keras.models.Sequential()
        seq_model.add(backbone)
        if add_flatten:
            seq_model.add(tf.keras.layers.Flatten())
        if dropout:
            seq_model.add(tf.keras.layers.Dropout(dropout))
        if activation is None:
            seq_model.add(
                tf.keras.layers.Dense(
                    units=n_classes,
                    kernel_regularizer=kernel_regularizer,
                    bias_regularizer=bias_regularizer,
                    activity_regularizer=activity_regularizer,
                )
            )
        else:
            seq_model.add(
                tf.keras.layers.Dense(
                    units=n_classes,
                    activation=activation,
                    kernel_regularizer=kernel_regularizer,
                    bias_regularizer=bias_regularizer,
                    activity_regularizer=activity_regularizer,
                )
            )

        return seq_model


def _block1_custom(x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
    """A residual block.
    Args:
      x: input tensor.
      filters: integer, filters of the bottleneck layer.
      kernel_size: default 3, kernel size of the bottleneck layer.
      stride: default 1, stride of the first layer.
      conv_shortcut: default True, use convolution shortcut if True,
          otherwise identity shortcut.
      name: string, block label.
    Returns:
      Output tensor for the residual block.
    """
    bn_axis = 3

    if conv_shortcut:
        shortcut = layers.Conv2D(filters, 1, strides=stride, name=name + "_0_conv")(x)
        shortcut = layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + "_0_bn"
        )(shortcut)
    else:
        shortcut = x

    x = layers.Conv2D(filters, 1, strides=stride, name=name + "_1_conv")(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + "_1_bn")(
        x
    )
    x = layers.Activation("relu", name=name + "_1_relu")(x)

    x = layers.Conv2D(filters, kernel_size, padding="SAME", name=name + "_2_conv")(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + "_2_bn")(
        x
    )
    x = layers.Activation("relu", name=name + "_2_relu")(x)

    x = layers.Add(name=name + "_add")([shortcut, x])
    x = layers.Activation("relu", name=name + "_out")(x)
    return x


def _MiniResNet(
    n_stacks: int = 1,
    weights: str = None,
    input_shape: tuple = None,
    pooling: str = "avg",
):
    """
    Instantiates a miniature ResNet architecture model.
    Adapted from tf.keras implementation source.

    Inputs
    ------
    n_stacks : int, number of residual block stacks to include in the model.
    weights : str, path to pretrained weight files. If None, initialize with random weights.
    input_shape : tuple, input shape of the model.
    pooling : str, one of "avg", "max" or None. The type of pooling applied
        to the output features of the model.

    Outputs
    -------
    resnet : tf.keras.Model : the appropriate MiniResNet model, without a classification head.
    """

    def _custom_stack(x, filters, blocks, stride1=2, name=None):
        x = _block1_custom(
            x, filters, stride=stride1, conv_shortcut=True, name=name + "_block1"
        )
        for i in range(2, blocks + 1):
            x = _block1_custom(
                x, filters, conv_shortcut=False, name=name + "_block" + str(i)
            )
        return x

    def _stack_fn(x):
        for i in range(n_stacks):
            x = _custom_stack(x, 64 * 2 ** (i), 2, name="conv{}".format(2 + i))
        return x

    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Conv2D(64, 7, strides=2, padding="same", name="conv1")(inputs)
    x = layers.BatchNormalization(name="conv1_bn")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

    x = _stack_fn(x)

    if pooling == "avg":
        x = layers.GlobalAveragePooling2D()(x)
    elif pooling == "max":
        x = layers.GlobalMaxPooling2D()(x)

    return Model(inputs, x, name="resnet_mini")


def _check_parameters(
    n_stacks: int = None, pooling: str = None, pretrained_weights: bool = None
):
    if pooling is None or pooling == "None":
        warnings.warn(
            message="Warning : pooling is set to None \
                        instead of 'avg' or 'max'. Ignore this warning if this is deliberate."
        )
    if pretrained_weights:
        assert int(n_stacks) in [1, 2, 3], (
            "n_stacks parameter must be one of (1, 2, 3)\n"
            ", current value is {}".format(n_stacks)
        )


def _get_scratch_model(
    input_shape: tuple = None,
    n_stacks: int = None,
    n_classes: int = None,
    pooling: str = "avg",
    use_garbage_class: bool = False,
    multi_label: bool = False,
    dropout: float = 0.0,
    kernel_regularizer=None,
    activity_regularizer=None,
):
    """
    Creates a MiniResNet model from scratch, i.e. with randomly initialized weights.
    Inputs
    ------
    input_shape : tuple, input shape of the model. Should be in format (n_mels, patch_length)
        and not (n_mels, patch_length, 1)
    n_stacks : int, number of residual block stacks to include in the model.
    n_classes : int, number of neurons of the classification head
    pooling : str, one of "avg", "max" or None. The type of pooling applied
    to the output features of the model.
    use_garbage_class : bool, if True an additional neuron is added to the classification head
        to accomodate for the "garbage" class.
    dropout : float, dropout probability applied to the classification head.
    multi_label : bool, set to True if output is multi-label. If True, activation function is a sigmoïd,
        if False it is a softmax instead.
    kernel_regularizer : tf.keras.Regularizer, kernel regularizer applied to the classification head.
        NOTE : Currently not parametrable by the user.
    activity_regularizer : tf.keras.Regularizer, activity regularizer applied to the classification head.
        NOTE : Currently not parametrable by the user.

    Outputs
    -------
    miniresnet : tf.keras.Model, MiniResNet model with the appropriate classification head.
    """

    input_shape = (input_shape[0], input_shape[1], 1)
    if use_garbage_class:
        n_classes += 1

    if multi_label:
        activation = "sigmoïd"
    else:
        activation = "softmax"

    backbone = _MiniResNet(
        n_stacks=n_stacks, input_shape=input_shape, weights=None, pooling=pooling
    )

    add_flatten = pooling == None or pooling == "None"
    miniresnet = add_head(
        backbone=backbone,
        n_classes=n_classes,
        trainable_backbone=True,
        add_flatten=add_flatten,
        functional=True,
        activation=activation,
        dropout=dropout,
        kernel_regularizer=kernel_regularizer,
        activity_regularizer=activity_regularizer,
    )
    return miniresnet


def _get_pretrained_model(
    n_stacks: int = None,
    n_classes: int = None,
    pooling: str = None,
    use_garbage_class: bool = False,
    multi_label: bool = False,
    dropout: float = 0.0,
    fine_tune: bool = False,
    kernel_regularizer=None,
    activity_regularizer=None,
):
    """
    Creates a MiniResNet model from ST provided pretrained weights
    Inputs
    ------
    n_stacks : int, number of residual block stacks to include in the model.
    n_classes : int, number of neurons of the classification head.
        Must be either 1, 2 or 3.
    pooling : str or None, if str must be "avg". Type of pooling applied to the pretrained backbone
    use_garbage_class : bool, if True an additional neuron is added to the classification head
        to accomodate for the "garbage" class.
    dropout : float, dropout probability applied to the classification head.
    fine_tune : bool, if True all the weights in the model are trainable.
        If False, only the classification head is trainable
    multi_label : bool, set to True if output is multi-label. If True, activation function is a sigmoïd,
        if False it is a softmax instead.
    kernel_regularizer : tf.keras.Regularizer, kernel regularizer applied to the classification head.
        NOTE : Currently not parametrable by the user.
    activity_regularizer : tf.keras.Regularizer, activity regularizer applied to the classification head.
        NOTE : Currently not parametrable by the user.

    Outputs
    -------
    miniresnet : tf.keras.Model, MiniResNet model with the appropriate classification head.
    """
    # Load model
    if pooling == "avg":
        miniresnet = tf.keras.models.load_model(
            Path(
                Path(__file__).parent.resolve(),
                "pooled_miniresnet_{}_stacks_backbone.h5".format(n_stacks),
            )
        )
        add_flatten = False
    elif pooling == None:
        miniresnet = tf.keras.models.load_model(
            Path(
                Path(__file__).parent.resolve(),
                "miniresnet_{}_stacks_backbone.h5".format(n_stacks),
            )
        )
        add_flatten = True
    else:
        raise NotImplementedError(
            "When using a pretrained backbone for miniresnet, 'pooling' must be either None or 'avg'"
        )

    # Add head
    if use_garbage_class:
        n_classes += 1
    if multi_label:
        activation = "sigmoïd"
    else:
        activation = "softmax"

    miniresnet = add_head(
        backbone=miniresnet,
        n_classes=n_classes,
        trainable_backbone=fine_tune,
        add_flatten=add_flatten,
        functional=True,
        activation=activation,
        dropout=dropout,
        kernel_regularizer=kernel_regularizer,
        activity_regularizer=activity_regularizer,
    )

    return miniresnet


def get_model(
    input_shape: tuple = None,
    n_stacks: int = None,
    n_classes: int = None,
    pooling: str = "avg",
    use_garbage_class: bool = False,
    multi_label: bool = False,
    dropout: float = 0.0,
    pretrained_weights: bool = None,
    fine_tune: bool = False,
    kernel_regularizer=None,
    activity_regularizer=None,
):
    """
    Instantiate a MiniResNet model, and perform basic sanity check on input parameters.
    Inputs
    ------
    input_shape : tuple, input shape of the model. Should be in format (n_mels, patch_length)
        and not (n_mels, patch_length, 1). Only used if pretrained_weights is False.
    n_stacks : int, number of residual block stacks to include in the model.
            Must be either 1, 2 or 3.
    n_classes : int, number of neurons of the classification head.
    pooling : str, one of "avg", "max" or None. The type of pooling applied
    to the output features of the model.
    use_garbage_class : bool, if True an additional neuron is added to the classification head
        to accomodate for the "garbage" class.
    dropout : float, dropout probability applied to the classification head.
    pretrained_weights : bool, use ST-provided pretrained weights is True, and
        create model from scratch if False.
    fine_tune : bool, if True all the weights in the model are trainable.
        If False, only the classification head is trainable
    multi_label : bool, set to True if output is multi-label. If True, activation function is a sigmoïd,
        if False it is a softmax instead.
    kernel_regularizer : tf.keras.Regularizer, kernel regularizer applied to the classification head.
        NOTE : Currently not parametrable by the user.
    activity_regularizer : tf.keras.Regularizer, activity regularizer applied to the classification head.
        NOTE : Currently not parametrable by the user.
    """

    _check_parameters(
        n_stacks=n_stacks, pooling=pooling, pretrained_weights=pretrained_weights
    )

    # Failsafe convert pooling None string to None type
    # This should happen earlier but just in case
    if pooling == "None":
        pooling = None

    if not pretrained_weights:
        miniresnet = _get_scratch_model(
            input_shape=input_shape,
            n_stacks=n_stacks,
            n_classes=n_classes,
            pooling=pooling,
            use_garbage_class=use_garbage_class,
            multi_label=multi_label,
            dropout=dropout,
            kernel_regularizer=kernel_regularizer,
            activity_regularizer=activity_regularizer,
        )
    else:
        miniresnet = _get_pretrained_model(
            n_stacks=n_stacks,
            n_classes=n_classes,
            pooling=pooling,
            use_garbage_class=use_garbage_class,
            multi_label=multi_label,
            dropout=dropout,
            fine_tune=fine_tune,
            kernel_regularizer=kernel_regularizer,
            activity_regularizer=activity_regularizer,
        )

    return miniresnet

