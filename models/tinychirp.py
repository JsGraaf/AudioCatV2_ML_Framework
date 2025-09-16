from tensorflow import keras


def build_cnn_mel(
    input_shape=(None, None, 1),  # variable (freq, time, channels)
    conv_filters=(4, 4),  # to mirror your 4-channel convs
    kernel_size=(3, 3),
    pool_size=(2, 2),
    dense_units=8,
    lr=1e-3,
    padding="valid",  # "valid" mimics your shrinking dims; "same" keeps size
    alpha=0.3,
    gamma=2,
):
    Conv = keras.layers.Conv2D

    inputs = keras.Input(shape=input_shape)

    # Block 1: 3x3 Conv + ReLU -> MaxPool
    x = Conv(conv_filters[0], kernel_size, padding=padding, activation="relu")(inputs)
    x = keras.layers.MaxPooling2D(pool_size)(x)

    # Block 2: 3x3 Conv + ReLU -> MaxPool
    x = Conv(conv_filters[1], kernel_size, padding=padding, activation="relu")(x)
    x = keras.layers.MaxPooling2D(pool_size)(x)

    # "Reshape/Flatten -> FC" equivalent for variable size: use GAP, then Dense
    x = keras.layers.GlobalAveragePooling2D()(x)  # shape -> (batch, filters_last)
    x = keras.layers.Dense(dense_units, activation="relu")(x)  # FC + ReLU (8)

    # Binary head (single sigmoid instead of 2-way softmax)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.BinaryFocalCrossentropy(
            alpha=alpha,
            gamma=gamma,
            name="binary_focal_crossentropy",
            from_logits=False,
        ),
        # loss=keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[
            keras.metrics.AUC(name="pr_auc", curve="PR"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.MeanSquaredError(name="Brier Score"),
            keras.metrics.RecallAtPrecision(precision=0.90, name="recall_at_p90"),
        ],
    )
    return model
