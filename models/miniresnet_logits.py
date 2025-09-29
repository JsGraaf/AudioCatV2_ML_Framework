import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, layers

from callbacks import RecallAtP90


def build_miniresnet(
    input_shape,  # (n_mels, n_frames)
    n_classes,
    n_stacks=1,  # 1, 2, or 3
    pooling="avg",  # "avg" | "max" | None
    multi_label=True,  # True -> multi-label, False -> single-label
    use_garbage_class=False,
    dropout=0.0,
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    name="miniresnet",
    lr=1e-3,
    gamma=1.0,
    alpha=0.45,
    loss="BCE",  # "BCE" | "FOCAL"
    logits=True,  # <<< NEW: if True, final Dense has no activation
):
    assert n_stacks in (1, 2, 3), "n_stacks must be 1, 2, or 3"
    if pooling == "None":
        pooling = None

    # --- blocks ---
    def residual_block(
        x, filters, kernel_size=3, stride=1, conv_shortcut=True, prefix=""
    ):
        bn_axis = 3
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
        x = residual_block(
            x,
            filters,
            stride=stride1,
            conv_shortcut=True,
            prefix=f"{name_prefix}_block1",
        )
        for i in range(2, blocks + 1):
            x = residual_block(
                x, filters, conv_shortcut=False, prefix=f"{name_prefix}_block{i}"
            )
        return x

    # --- stem ---
    inputs = layers.Input(shape=(input_shape[0], input_shape[1], 1), name="input")
    x = layers.Conv2D(64, 7, strides=2, padding="same", name="conv1")(inputs)
    x = layers.BatchNormalization(name="conv1_bn")(x)
    x = layers.Activation("relu", name="conv1_relu")(x)
    x = layers.MaxPooling2D(3, strides=2, padding="same", name="pool1")(x)

    # --- stacks ---
    for i in range(n_stacks):
        x = stack(x, filters=64 * (2**i), blocks=2, stride1=2, name_prefix=f"conv{2+i}")

    # --- head pre-pool ---
    if pooling == "avg":
        x = layers.GlobalAveragePooling2D(name="gap")(x)
    elif pooling == "max":
        x = layers.GlobalMaxPooling2D(name="gmp")(x)
    else:
        x = layers.Flatten(name="flatten")(x)

    if dropout and dropout > 0.0:
        x = layers.Dropout(dropout, name="head_dropout")(x)

    out_units = n_classes + (1 if use_garbage_class else 0)

    # Decide activation of the Dense head:
    #  - logits=True  -> activation=None (linear), train with from_logits=True
    #  - logits=False -> activation='sigmoid' (multi-label) or 'softmax' (single-label)
    if logits:
        head_activation = None
    else:
        head_activation = "sigmoid" if multi_label else "softmax"

    outputs = layers.Dense(
        out_units,
        activation=head_activation,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        name="new_head",
    )(x)

    model = Model(inputs=inputs, outputs=outputs, name=name)

    # Choose loss/metrics consistent with logits
    if multi_label:
        # Binary losses per class
        if loss.upper() == "FOCAL":
            criterion = keras.losses.BinaryFocalCrossentropy(
                alpha=alpha, gamma=gamma, from_logits=bool(logits), name="binary_focal"
            )
        else:
            criterion = keras.losses.BinaryCrossentropy(
                from_logits=bool(logits), name="binary_ce"
            )
    else:
        # Single-label (mutually exclusive) case
        if loss.upper() == "FOCAL":
            # If you need focal for single-label, use categorical focal (custom) or sparse focal.
            # Fallback to CE here:
            criterion = keras.losses.CategoricalCrossentropy(
                from_logits=bool(logits), name="categorical_ce"
            )
        else:
            criterion = keras.losses.CategoricalCrossentropy(
                from_logits=bool(logits), name="categorical_ce"
            )

    # Helpful metrics; probabilities only when logits=False. With logits=True, use AUC via from_logits.
    metrics = [RecallAtP90(from_logits=logits)]
    if multi_label:
        # AUC can accept logits if you pass from_logits=True in TF 2.12+
        try:
            metrics.append(
                keras.metrics.AUC(name="auc_pr", curve="PR", from_logits=bool(logits))
            )
        except TypeError:
            # Older TF: wrap a sigmoid for metrics via a Lambda layer if needed
            pass
        metrics += [
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ]
    else:
        metrics += [
            keras.metrics.CategoricalAccuracy(name="acc"),
        ]

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=criterion,
        metrics=metrics,
    )
    return model
