import logging
import os

import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from cross_validation import make_cv_splits_with_validation
from init import init
from load_birdclef import load_and_clean_birdclef
from metric_utils import get_scores_per_class, plot_pr_with_thresholds
from misc import load_config
from models.binary_cnn import build_binary_cnn
from models.tinychirp import build_cnn_mel
from tf_datasets import build_file_lists
from models.miniresnet import build_miniresnet

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
    cv_sets = make_cv_splits_with_validation(
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
    prediction_df = pd.DataFrame()
    for i, fold in enumerate(datasets):

        logging.info(f"Running fold: {fold['fold_id']}")
        # model = build_binary_cnn(
        #     input_shape=(
        #         config["data"]["audio"]["n_mels"],
        #         config["data"]["audio"]["n_frames"],
        #         1,
        #     ),
        #     alpha=0.3,
        #     gamma=2,
        # )
        # model = build_cnn_mel(
        #     input_shape=(
        #         config["data"]["audio"]["n_mels"],
        #         config["data"]["audio"]["n_frames"],
        #         1,
        #     ),
        #     alpha=0.3,
        #     gamma=2,
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

        # Callbacks
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

        lr = ReduceLROnPlateau(
            monitor="val_recall_at_p90",
            mode="max",
            factor=0.5,
            patience=5,
            min_delta=0.001,
            cooldown=0,
            min_lr=1e-6,
            verbose=1,
            start_from_epoch=10,
        )

        model.fit(
            fold["train_ds"],
            epochs=config["ml"]["epochs_per_fold"],
            steps_per_epoch=int(
                np.ceil(fold["train_size"] / config["ml"]["batch_size"])
            ),
            validation_data=fold["val_ds"],
            verbose=1,
            callbacks=[early, ckpt],
        )

        results = model.evaluate(fold["test_ds"], verbose=1, return_dict=True)
        results["id"] = fold["fold_id"]

        predictions = model.predict(fold["test_ds"], verbose=1).ravel()

        fold["test_df"]["predictions"] = predictions

        prediction_df = pd.concat([prediction_df, fold["test_df"]])

        scores.append(results)

    prediction_df["true_label"] = prediction_df["primary_label"].apply(
        lambda x: 1 if x == config["exp"]["target"] else 0
    )
    print(get_scores_per_class(prediction_df))

    plot_pr_with_thresholds(
        prediction_df["true_label"],
        prediction_df["predictions"],
        marks=(0.01, 1.0),
    )

    # Write results to a csv
    pd.DataFrame(scores).to_csv("Training_Results.csv", index=False)

    del datasets
