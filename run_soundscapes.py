import logging
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from init import init
from load_birdclef import load_and_clean_soundscapes
from metric_utils import plot_confusion_matrix
from misc import load_config
from models.binary_cnn import build_binary_cnn
from tf_datasets import make_soundscape_dataset

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    # Load the config
    CONFIG_PATH = Path("config.yaml")
    logging.info(f"Loading config from {CONFIG_PATH}")
    config = load_config(CONFIG_PATH)

    if config is None:
        sys.exit(1)

    logging.info(
        f"Running Experiment {config['exp']['name']} for {config['exp']['target']}"
    )

    # Initialize the framework
    init(config["exp"]["random_state"])

    # Load the soundscapes metadata
    logging.info(f"Loading the soundscapes from {config['data']['soundscape_path']}")

    soundscapes_df = load_and_clean_soundscapes(
        config["data"]["birdclef_path"], config["data"]["soundscape_path"]
    )

    if soundscapes_df is None:
        sys.exit(1)

    soundscapes_df.apply(
        lambda x: print(f"Failed {x}") if not os.path.isfile(x["path"]) else {}, axis=1
    )

    # Create the TF dataset
    soundscape_ds = make_soundscape_dataset(soundscapes_df, config)

    # Load the model
    model = build_binary_cnn(
        input_shape=(
            config["data"]["audio"]["n_mels"],
            config["data"]["audio"]["n_frames"],
            1,
        )
    )

    # Load the weights
    model.load_weights(config["ml"]["model_path"])

    predictions = model.predict(soundscape_ds)

    soundscapes_df["predictions"] = predictions

    soundscapes_df["true_label"] = soundscapes_df["primary_label"].apply(
        lambda x: 1 if x == config["exp"]["target"] else 0
    )

    soundscapes_df["predicted_label"] = soundscapes_df["predictions"].apply(
        lambda x: 1 if x >= 0.37 else 0
    )

    soundscapes_df.to_csv("soundscape_predictions.csv", index=False)

    print(
        soundscapes_df[
            soundscapes_df["predicted_label"] == soundscapes_df["true_label"]
        ].shape[0]
        / soundscapes_df.shape[0]
    )

    # ax = plot_confusion_matrix(soundscapes_df["true_label"], predicted_labels)
    # ax.set_title(f"t={0.37}")
    # plt.show()
