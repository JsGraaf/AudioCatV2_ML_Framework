import logging
import os
import pprint
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from callbacks import BestF1OnVal
from init import init
from load_birdclef import load_and_clean_soundscapes
from metric_utils import (
    get_scores_per_class,
    get_scores_per_class_one_hot,
    plot_confusion_matrix,
)
from misc import load_config
from models.binary_cnn import build_binary_cnn
from models.dual_class_cnn import build_dual_class_cnn
from models.miniresnet import build_miniresnet
from models.tinychirp import build_cnn_mel
from tf_datasets import make_soundscape_dataset


def get_birdclef_soundscapes(config):
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

    return soundscapes_df, soundscape_ds


def get_birdset_soundscapes(config):
    # Load the soundscapes metadata
    logging.info(
        f"Loading the soundscapes from {config['data']['birdset_soundscape_path']}"
    )

    soundscapes_df = pd.read_csv(config["data"]["birdset_soundscape_path"])

    if soundscapes_df is None:
        sys.exit(1)

    soundscapes_df.apply(
        lambda x: print(f"Failed {x}") if not os.path.isfile(x["path"]) else {}, axis=1
    )

    # Drop all segments longer than seconds
    soundscapes_df["duration"] = soundscapes_df["end"] - soundscapes_df["start"]
    soundscapes_df = soundscapes_df[
        soundscapes_df["duration"] <= config["data"]["audio"]["seconds"]
    ]

    print(
        soundscapes_df[soundscapes_df["primary_label"] == config["exp"]["target"]].shape
    )

    # Create the TF dataset
    soundscape_ds = make_soundscape_dataset(soundscapes_df, config)

    return soundscapes_df, soundscape_ds


if __name__ == "__main__":
    pp = pprint.PrettyPrinter(indent=4)
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

    if config["data"]["use_birdset"]:
        soundscapes_df, soundscape_ds = get_birdset_soundscapes(config)
    else:
        soundscapes_df, soundscape_ds = get_birdclef_soundscapes(config)

    # Load the model
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
        )

    # Load the weights
    model.load_weights(config["ml"]["model_path"])

    predictions = model.predict(soundscape_ds)

    soundscapes_df["true_label"] = soundscapes_df["primary_label"].apply(
        lambda x: 1 if x == config["exp"]["target"] else 0
    )

    if config["exp"]["one_hot"]:
        soundscapes_df["p_class0"] = predictions[:, 0]
        soundscapes_df["p_class1"] = predictions[:, 1]
        pp.pprint(get_scores_per_class_one_hot(soundscapes_df))

    else:
        soundscapes_df["predictions"] = predictions
        pp.pprint(get_scores_per_class(soundscapes_df))

    soundscapes_df.to_csv("soundscape_predictions.csv", index=False)
