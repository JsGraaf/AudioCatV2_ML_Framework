import logging
import os
from typing import Dict

import keras_tuner as kt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from init import init
from load_birdclef import load_and_clean_birdclef
from misc import hp_audio_and_aug, load_config
from models.binary_cnn_hyper import model_builder
from tf_datasets import build_final_dataset


class DataAwareHyperband(kt.Hyperband):
    def run_trial(self, trial, *args, **kwargs):
        hp = trial.hyperparameters

        hp_cfg = hp_audio_and_aug(hp)

        # Load the dataset
        logging.info(f"Loading birdclef data from {hp_cfg['data']['birdclef_path']}")

        birdclef_df = load_and_clean_birdclef(
            hp_cfg["data"]["birdclef_path"],
            hp_cfg["data"]["min_per_class"],
        )

        # Check that all the paths exist
        birdclef_df.apply(
            lambda x: print(f"Failed {x}") if not os.path.isfile(x["path"]) else {},
            axis=1,
        )

        # Check if the target is in the df
        if hp_cfg["exp"]["target"] not in birdclef_df["primary_label"].unique():
            logging.error("Target Category not in df!")
            exit(1)

        # Create a dataset containing all positives and pos_neg_ratio times negatives
        logging.info(f"Making the final dataset")

        print(hp_cfg)
        datasets = build_final_dataset(birdclef_df, hp_cfg)

        early = EarlyStopping(
            monitor="val_recall_at_p90",
            mode="max",
            patience=8,
            min_delta=1e-3,
            restore_best_weights=True,
            verbose=1,
        )

        ret = super().run_trial(
            trial,
            x=datasets["train_ds"],
            validation_data=datasets["val_ds"],
            epochs=hp_cfg["ml"]["epochs_per_fold"],
            callbacks=[early],
            **kwargs,
        )

        del datasets

        return ret


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

    # Train the model
    tuner = DataAwareHyperband(
        hypermodel=model_builder,
        objective=kt.Objective("val_recall_at_p90", direction="max"),
        max_epochs=100,
        factor=3,
        seed=config["exp"]["random_state"],
        directory="my_dir",
        project_name="binary_cnn_v2",
    )

    tuner.search()

    print(tuner.results_summary())

    # Get the best trial’s hyperparameters
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

    with open("best_hparams.yaml", "w") as f:
        yaml.safe_dump(best_hp.values, f, sort_keys=False)

    # Get the best model
    best_model = tuner.get_best_models(num_models=1)[0]
    best_model.save("best_model.keras")

    # model = model_builder(best_hps[0])
    #
    # Build the dataset with the best hyperparameters

    # # Retrain the model
    # history = model.fit(
    #     datasets["train_ds"], epochs=50, validation_data=datasets["val_ds"]
    # )
    #
    # val_pr_auc_per_epoch = history.history["val_pr_auc"]
    # best_epoch = val_pr_auc_per_epoch.index(max(val_pr_auc_per_epoch)) + 1
    # print(f"Best epoch: {best_epoch}")
    #
    # hypermodel = tuner.hypermodel.build(best_hps)
    # hypermodel.fit(datasets["train_ds"], epochs=50, validation_data=datasets["val_ds"])
    #
    # eval_result = hypermodel.evaluate(datasets["test_ds"])
    # print(eval_result)
