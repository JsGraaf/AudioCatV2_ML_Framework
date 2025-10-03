import logging
import os
import sys

import pandas as pd

from load_birdclef import load_and_clean_birdclef
from tf_datasets import build_final_dataset


def get_birdclef_datasets(config):
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
        sys.exit(1)

    # Create a dataset containing all positives and pos_neg_ratio times negatives
    logging.info(f"Making the final dataset")
    datasets = build_final_dataset(birdclef_df, config)

    return datasets


def get_birdset_dataset(config):
    # Load the dataset
    logging.info(f"Loading birdclef data from {config['data']['birdset_path']}")

    birdset_df = pd.read_csv(config["data"]["birdset_path"])

    # Check that all the paths exist
    birdset_df.apply(
        lambda x: print(f"Failed {x}") if not os.path.isfile(x["path"]) else {}, axis=1
    )

    # Check if the target is in the df
    if config["exp"]["target"] not in birdset_df["primary_label"].unique():
        logging.error("Target Category not in df!")
        sys.exit(1)

    # Create a dataset containing all positives and pos_neg_ratio times negatives
    logging.info(f"Making the final dataset")
    datasets = build_final_dataset(birdset_df, config)

    return datasets


def get_custom_dataset(config, percentage: float = 1.0):
    # Load the dataset
    logging.info(f"Loading custom data from {config['data']['custom_data_path']}")

    custom_df = load_and_clean_birdclef(
        config["data"]["custom_data_path"],
        config["data"]["min_per_class"],
    )

    # Check that all the paths exist
    custom_df.apply(
        lambda x: print(f"Failed {x}") if not os.path.isfile(x["path"]) else {}, axis=1
    )

    # Check if the target is in the df
    if config["exp"]["target"] not in custom_df["primary_label"].unique():
        logging.error("Target Category not in df!")
        sys.exit(1)

    # Create a dataset containing all positives and pos_neg_ratio times negatives
    logging.info(f"Making the final dataset")
    datasets = build_final_dataset(custom_df, config, percentage=percentage)

    return datasets
