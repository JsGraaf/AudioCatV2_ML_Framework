from tensorflow import keras

from misc import hp_audio_and_aug


def model_builder(hp):
    """
    Binary classifier for log-mel spectrograms (target vs non-target).
    Output: single sigmoid unit.
    Loss: Binary Focal Cross-Entropy
    """

    hp_cfg = hp_audio_and_aug(hp)

    input_shape = (hp.get("n_mels"), hp_cfg["data"]["audio"]["n_frames"], 1)

    lr = hp.Choice("lr", values=[1e-2, 1e-3, 1e-4])
    l2 = hp.Choice("l2", values=[1e-2, 1e-3, 1e-4, 1e-5])
    dropout = hp.Float("dropout", min_value=0, max_value=0.5, step=0.05)
    gamma = hp.Int("gamma", min_value=1, max_value=4, step=1)
    alpha = hp.Float("alpha", min_value=0.1, max_value=0.5, step=0.05)
    loss = hp.Choice("loss", ["BCE", "FOCAL"])

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
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Dropout(dropout)(x)

    # Block 2
    x = Conv(64, (3, 3), padding="same", kernel_regularizer=L2)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = Conv(64, (3, 3), padding="same", kernel_regularizer=L2)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Dropout(dropout)(x)

    # Block 3
    x = Conv(96, (3, 3), padding="same", kernel_regularizer=L2)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = Conv(96, (3, 3), padding="same", kernel_regularizer=L2)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Dropout(dropout)(x)

    # Head
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(128, activation="relu", kernel_regularizer=L2)(x)
    x = keras.layers.Dropout(dropout)(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)  # binary

    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=(
            keras.losses.BinaryFocalCrossentropy(
                alpha=alpha,
                gamma=gamma,
                name="binary_focal_crossentropy",
                from_logits=False,
            )
            if hp.get("loss") == "FOCAL"
            else keras.losses.BinaryCrossentropy(from_logits=False)
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
