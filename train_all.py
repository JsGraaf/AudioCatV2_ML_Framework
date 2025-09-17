import logging
import os
import sys

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from dataset_loaders import get_birdclef_datasets, get_birdset_dataset
from init import init
from misc import load_config
from models.binary_cnn import build_binary_cnn
from models.dual_class_cnn import build_dual_class_cnn
from models.miniresnet import build_miniresnet
from models.tinychirp import build_cnn_mel

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    # Load the config
    CONFIG_PATH = "config.yaml"
    logging.info(f"Loading config from {CONFIG_PATH}")
    config = load_config(CONFIG_PATH)

    if config is None:
        exit(1)

    logging.info(
        f"Running Experiment {config['exp']['name']} for {config['exp']['target']}"
    )

    # Initialize the framework
    init(config["exp"]["random_state"])

    if config["data"]["use_birdset"]:
        logging.info("[Info] Loading birdset!")
        datasets = get_birdset_dataset(config)
    else:
        logging.info("[Info] Loading Birdclef")
        datasets = get_birdclef_datasets(config)

    # Train the model
    if config["exp"]["one_hot"]:
        model = build_dual_class_cnn(
            input_shape=(
                config["data"]["audio"]["n_mels"],
                config["data"]["audio"]["n_frames"],
                1,
            )
        )
    else:
        # model = build_binary_cnn(
        #     input_shape=(
        #         config["data"]["audio"]["n_mels"],
        #         config["data"]["audio"]["n_frames"],
        #         1,
        #     )
        # )
        model = build_miniresnet(
            input_shape=(
                config["data"]["audio"]["n_mels"],
                config["data"]["audio"]["n_frames"],
                1,
            ),
            n_classes=1,
            loss=config["ml"]["loss"],
        )

    # model = build_cnn_mel(
    #     input_shape=(
    #         config["data"]["audio"]["n_mels"],
    #         config["data"]["audio"]["n_frames"],
    #         1,
    #     )
    # )

    # model = get_model(
    #     input_shape=(
    #         config["data"]["audio"]["n_mels"],
    #         config["data"]["audio"]["n_frames"],
    #         1,
    #     ),
    #     stacks=2,
    # )

    early = EarlyStopping(
        monitor="val_recall_at_p90",
        mode="max",
        patience=8,
        min_delta=1e-3,
        restore_best_weights=True,
        verbose=1,
        start_from_epoch=10,
    )
    ckpt = ModelCheckpoint(
        "output/best_train_all.keras",
        monitor="val_recall_at_p90",
        mode="max",
        save_best_only=True,
        verbose=1,
    )

    rlrop = ReduceLROnPlateau(
        monitor="val_recall_at_p90",
        mode="max",
        factor=0.5,
        patience=4,
        min_delta=0.001,
        cooldown=0,
        min_lr=1e-6,
        verbose=1,
    )

    model.fit(
        datasets["train_ds"],
        epochs=config["ml"]["epochs_per_fold"],
        steps_per_epoch=int(
            np.ceil(datasets["train_size"] / config["ml"]["batch_size"])
        ),
        validation_data=datasets["val_ds"],
        verbose=1,
        callbacks=[early, ckpt, rlrop],
    )

    results = model.evaluate(
        datasets["test_ds"],
        verbose=1,
        return_dict=True,
    )

    model.save("output/full_training_model.keras")

    print(results)

    pd.DataFrame(results, index=[0]).to_csv("Full_Training_Results.csv", index=False)
