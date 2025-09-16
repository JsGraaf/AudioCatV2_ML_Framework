from tensorflow import keras


# --------- Simple binary CNN ----------
def build_dual_class_cnn(
    input_shape=(128, 64, 1),
    lr=1e-3,
    l2=1e-3,
    dropout=0.05,
    gamma=1.0,
    alpha=0.45,
):
    """
    Binary classifier for log-mel spectrograms (target vs non-target).
    Output: single sigmoid unit.
    Loss: Binary Focal Cross-Entropy (with safe fallbacks).
    """
    L2 = keras.regularizers.l2(l2)
    Conv = keras.layers.Conv2D

    inputs = keras.Input(shape=input_shape)

    # Block 1
    x = Conv(32, (3, 3), padding="same", kernel_regularizer=L2)(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = Conv(32, (3, 3), padding="same", kernel_regularizer=L2)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)  # 128x64 -> 64x32
    x = keras.layers.Dropout(dropout)(x)

    # Block 2
    x = Conv(64, (3, 3), padding="same", kernel_regularizer=L2)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = Conv(64, (3, 3), padding="same", kernel_regularizer=L2)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)  # 64x32 -> 32x16
    x = keras.layers.Dropout(dropout)(x)

    # Block 3
    x = Conv(96, (3, 3), padding="same", kernel_regularizer=L2)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = Conv(96, (3, 3), padding="same", kernel_regularizer=L2)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)  # 32x16 -> 16x8
    x = keras.layers.Dropout(dropout)(x)

    # Head
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(128, activation="relu", kernel_regularizer=L2)(x)
    x = keras.layers.Dropout(dropout)(x)
    outputs = keras.layers.Dense(2, activation="softmax")(x)  # binary

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        # loss=keras.losses.BinaryFocalCrossentropy(
        #     alpha=alpha,
        #     gamma=gamma,
        #     name="binary_focal_crossentropy",
        #     from_logits=False,
        # ),
        loss=keras.losses.CategoricalCrossentropy(from_logits=False),
        # loss=keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[
            # keras.metrics.AUC(name="pr_auc", curve="PR", class_id=1),
            keras.metrics.Precision(name="precision", class_id=1),
            keras.metrics.Recall(name="recall", class_id=1),
            keras.metrics.RecallAtPrecision(
                precision=0.90, name="recall_at_p90", class_id=1
            ),
        ],
    )
    return model
