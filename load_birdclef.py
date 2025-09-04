import logging
import os
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import pandas as pd


def load_birdclef(path: Path, min_per_class: int = 0) -> Optional[pd.DataFrame]:
    """
    Loads the birdclef dataset
    """
    if not os.path.isfile(path):
        logging.error("Invalid path for birdclef")
        return None
    df = pd.read_csv(path)

    # Drop rare classes
    if min_per_class > 0:
        keep_labels = df["primary_label"].value_counts()
        keep_labels = keep_labels[keep_labels >= min_per_class].index
        df = df[df["primary_label"].isin(keep_labels)].reset_index(drop=True)

    # Drop unnamed column
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # Add group key based on auther + time to prevent straddeling
    df["group_key"] = df["author"] + df["time"]
    return df


def create_label_mapping(df: pd.DataFrame) -> dict:
    """
    The primary labels are stored seperately from the scientific names and need
    to be mapped so that we can trace back the birdnet requirements for the future
    soundscape predictions.
    """
    mapping = df.groupby("scientific_name")["primary_label"].first().to_dict()
    return mapping


def expand_birdclef(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Expands the dataframe so we can maximize the dataset. It creates
    a new dataset in which each row is a single detection.
    """

    rows = []

    def expand_row(row):
        # Map the cell to a list
        preds = eval(row["predictions"], {"OrderedDict": OrderedDict})

        if len(preds) == 0:
            return

        new_rows = []
        for pred in preds:
            # Get the first ordered dict item
            first = next(iter(pred[1].items()))
            r = {
                "path": row["path"],
                "file": row["filename"],
                "start": pred[0][0],
                "end": pred[0][1],
                "scientific_name": first[0].split("_")[0],
                "common_name": first[0].split("_")[1],
                "confidence": first[1],
                "group_key": row["group_key"],
            }
            new_rows.append(r)
        rows.extend(new_rows)

    df.apply(expand_row, axis=1)

    return pd.DataFrame(rows)


def load_and_clean_birdclef(
    path: Path, min_per_class: int = 0, allow_cache=True
) -> Optional[pd.DataFrame]:
    """
    This function:
    - Loads the dataframe
    - Maps the birdnet predictions to individual rows with primary labels
    """

    # First check if we have it cached
    if allow_cache and os.path.isfile(Path("cache/cleaned_birdnet_clef.csv")):
        logging.info("Found cached birndet_clef dataframe!")
        return pd.read_csv("cache/cleaned_birdnet_clef.csv")

    birdclef_df = load_birdclef(path, min_per_class=min_per_class)

    if birdclef_df is None:
        return None

    label_mapping = create_label_mapping(birdclef_df)

    expanded_df = expand_birdclef(birdclef_df)

    if expanded_df is None:
        return None

    # Add the primary labels to each row
    expanded_df["primary_label"] = expanded_df.apply(
        (
            lambda row: (
                label_mapping[row["scientific_name"]]
                if row["scientific_name"] in label_mapping.keys()
                else None
            )
        ),
        axis=1,
    )
    # Drop the none columns
    expanded_df = expanded_df.dropna(subset=["primary_label"])

    # Output to cache
    os.makedirs("cache", exist_ok=True)
    expanded_df.to_csv("cache/cleaned_birdnet_clef.csv")

    return expanded_df
