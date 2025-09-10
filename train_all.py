import logging
import os

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from cross_validation import make_cv_splits
from init import init
from load_birdclef import load_and_clean_birdclef
from misc import load_config
from models.binary_cnn import build_binary_cnn
from models.miniresnet import get_model
from models.tinychirp import build_cnn_mel
from tf_datasets import build_final_dataset

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

    # Load the dataset
    logging.info(f"Loading birdclef data from {config['data']['birdclef_path']}")

    birdclef_df = load_and_clean_birdclef(
        config["data"]["birdclef_path"],
        config["data"]["min_per_class"],
    )

    # Check that all the paths exist
    birdclef_df.apply(
        lambda x: print(f"Failed {x}") if not os.path.isfile(x["path"]) else {}, axis=1
    )

    # Check if the target is in the df
    if config["exp"]["target"] not in birdclef_df["primary_label"].unique():
        logging.error("Target Category not in df!")
        exit(1)

    # Create a dataset containing all positives and pos_neg_ratio times negatives
    logging.info(f"Making the final dataset")
    datasets = build_final_dataset(birdclef_df, config)

    # Train the model

    model = build_binary_cnn(
        input_shape=(
            config["data"]["audio"]["n_mels"],
            config["data"]["audio"]["n_frames"],
            1,
        )
    )

    # model = build_cnn_mel(
    #     input_shape=(
    #         config["data"]["audio"]["n_mels"],
    #         config["data"]["audio"]["n_frames"],
    #         1,
    #     )
    # )

    early = EarlyStopping(
        monitor="val_pr_auc",
        mode="max",
        patience=8,
        min_delta=1e-3,
        restore_best_weights=True,
        verbose=1,
    )
    ckpt = ModelCheckpoint(
        "output/best_train_all.keras",
        monitor="val_pr_auc",
        mode="max",
        save_best_only=True,
        verbose=1,
    )

    print(datasets["val_size"])

    model.fit(
        datasets["train_ds"],
        epochs=config["ml"]["epochs_per_fold"],
        steps_per_epoch=int(
            np.ceil(datasets["train_size"] / config["ml"]["batch_size"])
        ),
        validation_data=datasets["val_ds"],
        verbose=1,
        callbacks=[early, ckpt],
    )

    results = model.evaluate(datasets["test_ds"], verbose=1, return_dict=True)

    model.save("output/full_training_model.keras")

    print(results)

    pd.DataFrame(results, index=[0]).to_csv("Full_Training_Results.csv", index=False)
