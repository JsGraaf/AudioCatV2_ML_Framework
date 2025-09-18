import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, layers


def build_miniresnet(
    input_shape,  # (n_mels, n_frames)
    n_classes,
    n_stacks=1,  # 1, 2, or 3 stacks
    pooling="avg",  # "avg", "max", or None
    multi_label=True,  # True -> sigmoid head, False -> softmax
    use_garbage_class=False,  # add an extra "garbage/other" neuron
    dropout=0.0,  # dropout rate on head
    kernel_regularizer=None,  # e.g., tf.keras.regularizers.l2(1e-4)
    bias_regularizer=None,
    activity_regularizer=None,
    name="miniresnet",
    lr=1e-3,
    gamma=1.0,
    alpha=0.45,
    loss="BCE",
):
    """
    MiniResNet (STM32-style) for 2D time–frequency inputs.

    input_shape: (n_mels, n_frames)
    n_stacks:    number of residual stacks (1–3). Each stack has 2 residual blocks.
    pooling:     "avg", "max", or None (if None => Flatten before Dense).
    multi_label: if True, uses sigmoid (multi-label); else softmax.
    """

    assert n_stacks in (1, 2, 3), "n_stacks must be 1, 2, or 3"
    if pooling == "None":
        pooling = None

    # --- Residual building blocks (STM32 zoo style) ---
    def residual_block(
        x, filters, kernel_size=3, stride=1, conv_shortcut=True, prefix=""
    ):
        bn_axis = 3  # channels_last

        if conv_shortcut:
            shortcut = layers.Conv2D(
                filters, 1, strides=stride, name=f"{prefix}_0_conv"
            )(x)
            shortcut = layers.BatchNormalization(
                axis=bn_axis, epsilon=1.001e-5, name=f"{prefix}_0_bn"
            )(shortcut)
        else:
            shortcut = x

        out = layers.Conv2D(filters, 1, strides=stride, name=f"{prefix}_1_conv")(x)
        out = layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=f"{prefix}_1_bn"
        )(out)
        out = layers.Activation("relu", name=f"{prefix}_1_relu")(out)

        out = layers.Conv2D(
            filters, kernel_size, padding="same", name=f"{prefix}_2_conv"
        )(out)
        out = layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=f"{prefix}_2_bn"
        )(out)
        out = layers.Activation("relu", name=f"{prefix}_2_relu")(out)

        out = layers.Add(name=f"{prefix}_add")([shortcut, out])
        out = layers.Activation("relu", name=f"{prefix}_out")(out)
        return out

    def stack(x, filters, blocks, stride1=2, name_prefix="conv"):
        # First block downsamples (stride1) with conv shortcut
        x = residual_block(
            x,
            filters,
            stride=stride1,
            conv_shortcut=True,
            prefix=f"{name_prefix}_block1",
        )
        # Remaining blocks keep shape (identity shortcut)
        for i in range(2, blocks + 1):
            x = residual_block(
                x, filters, conv_shortcut=False, prefix=f"{name_prefix}_block{i}"
            )
        return x

    # --- Input & stem ---
    inputs = layers.Input(
        shape=(input_shape[0], input_shape[1], 1), name="input"
    )  # expand channel=1 outside if you prefer
    x = layers.Conv2D(64, 7, strides=2, padding="same", name="conv1")(inputs)
    x = layers.BatchNormalization(name="conv1_bn")(x)
    x = layers.Activation("relu", name="conv1_relu")(x)
    x = layers.MaxPooling2D(3, strides=2, padding="same", name="pool1")(x)

    # --- Residual stacks ---
    # Stack i uses filters = 64 * (2**i), each with 2 blocks (STM32 MiniResNet style)
    for i in range(n_stacks):
        x = stack(x, filters=64 * (2**i), blocks=2, stride1=2, name_prefix=f"conv{2+i}")

    # --- Global pooling or flatten ---
    if pooling == "avg":
        x = layers.GlobalAveragePooling2D(name="gap")(x)
        add_flatten = False
    elif pooling == "max":
        x = layers.GlobalMaxPooling2D(name="gmp")(x)
        add_flatten = False
    else:
        x = layers.Flatten(name="flatten")(x)
        add_flatten = True  # not used further, but mirrors original logic

    if dropout and dropout > 0.0:
        x = layers.Dropout(dropout, name="head_dropout")(x)

    # --- Classification head ---
    out_units = n_classes + (1 if use_garbage_class else 0)
    activation = "sigmoid" if multi_label else "softmax"

    outputs = layers.Dense(
        out_units,
        activation=activation,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        name="new_head",
    )(x)

    model = Model(inputs=inputs, outputs=outputs, name=name)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=(
            keras.losses.BinaryFocalCrossentropy(
                alpha=alpha,
                gamma=gamma,
                name="binary_focal_crossentropy",
                from_logits=False,
            )
            if loss == "FOCAL"
            else keras.losses.BinaryCrossentropy(from_logits=False)
        ),
        # loss=keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=[
            # keras.metrics.AUC(name="pr_auc", curve="PR", class_id=1),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.RecallAtPrecision(precision=0.90, name="recall_at_p90"),
        ],
    )
    return model
