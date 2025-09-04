import logging
import os

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cross_validation import make_cv_splits
from init import init
from load_birdclef import load_and_clean_birdclef
from misc import load_config
from models.binary_cnn import build_binary_cnn
from tf_datasets import build_file_lists

if __name__ == "__main__":
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

    # Make the splits
    logging.info(f"Making the CV Splits")
    cv_sets = make_cv_splits(
        birdclef_df,
        target=config["exp"]["target"],
        n_splits=config["data"]["n_splits"],
        random_state=config["exp"]["random_state"],
    )

    # Create the TF datasets
    logging.info(f"Making TF datasets for each fold")
    datasets = build_file_lists(birdclef_df, cv_sets, config=config)

    # Train the model
    scores = []
    for fold in datasets[:3]:

        logging.info(f"Running fold: {fold['fold_id']}")
        model = build_binary_cnn(
            input_shape=(
                config["data"]["audio"]["n_mels"],
                config["data"]["audio"]["n_frames"],
                1,
            )
        )

        model.fit(
            fold["train_ds"],
            epochs=config["ml"]["epochs_per_fold"],
            steps_per_epoch=int(
                np.ceil(fold["train_size"] / config["ml"]["batch_size"])
            ),
            verbose=1,
        )

        results = model.evaluate(fold["test_ds"], verbose=1, return_dict=True)
        results["id"] = fold["fold_id"]

        scores.append(results)

    print(scores)

    # Write results to a csv
    pd.DataFrame(scores).to_csv("Training_Results.csv", index=False)

    # Print the average metrics

    del datasets
