import logging
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from audio_processing import audio_pipeline


def plan_epoch_counts(n_pos_train: int, neg_pos_ratio: float = 2.0) -> Tuple[int, int]:
    pos_count = int(n_pos_train)
    neg_count = int(round(neg_pos_ratio * pos_count))
    return pos_count, neg_count


def sample_train_negatives(
    neg_all: Sequence[str], n_neg: int, seed: int
) -> Sequence[str]:
    n = min(n_neg, len(neg_all))
    rng = random.Random(seed)
    return rng.sample(list(neg_all), n) if n > 0 else []


def make_fixed_val_negatives(
    neg_all: Sequence[str], n_pos_val: int, neg_pos_ratio: int, seed: int
) -> Sequence[str]:
    n_neg = min(neg_pos_ratio * n_pos_val, len(neg_all))
    rng = random.Random(seed)
    return rng.sample(list(neg_all), n_neg) if n_neg > 0 else []


def build_train_dataset(
    pos_files: List[Tuple[Sequence[str], float, float]],
    neg_files: List[Tuple[Sequence[str], float, float]],
    seed: int,
    config: Dict,
    shuffle: bool = True,
) -> tf.data.Dataset:
    labels_pos = tf.ones([len(pos_files)], dtype=tf.float32)
    labels_neg = tf.zeros([len(neg_files)], dtype=tf.float32)

    def _map_pos(filename, start, end, label):
        return audio_pipeline((filename, start, end), config, augments=True), label

    def _map_neg(filename, start, end, label):
        return audio_pipeline((filename, start, end), config, augments=True), label

    def _map_pos_one_hot(filename, start, end, label):
        return audio_pipeline(
            (filename, start, end), config, augments=True
        ), tf.one_hot(tf.cast(label, tf.int32), depth=2)

    def _map_neg_one_hot(filename, start, end, label):
        return audio_pipeline(
            (filename, start, end), config, augments=True
        ), tf.one_hot(tf.cast(label, tf.int32), depth=2)

    ds_pos = tf.data.Dataset.from_tensor_slices(
        (
            list(f[0] for f in pos_files),
            list(f[1] for f in pos_files),
            list(f[2] for f in pos_files),
            labels_pos,
        )
    )
    if config["exp"]["one_hot"]:
        ds_pos = ds_pos.map(_map_pos_one_hot, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds_pos = ds_pos.map(_map_pos, num_parallel_calls=tf.data.AUTOTUNE)

    ds_neg = tf.data.Dataset.from_tensor_slices(
        (
            list(f[0] for f in neg_files),
            list(f[1] for f in neg_files),
            list(f[2] for f in neg_files),
            labels_neg,
        )
    )
    if config["exp"]["one_hot"]:
        ds_neg = ds_neg.map(_map_neg_one_hot, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds_neg = ds_neg.map(_map_neg, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds_pos.concatenate(ds_neg)
    if shuffle:
        ds = ds.shuffle(
            buffer_size=len(pos_files) + len(neg_files),
            seed=seed,
            reshuffle_each_iteration=True,
        )
    # ds = ds.batch(batch_size).map(lambda x, y: mixup_batch_binary(x, y), num_parallel_calls=tf.data.AUTOTUNE).cache().repeat()
    ds = ds.batch(config["ml"]["batch_size"]).cache()
    return ds


def build_val_dataset(
    pos_files: Sequence[str], neg_files: Sequence[str], config: Dict
) -> tf.data.Dataset:

    labels_pos = tf.ones([len(pos_files)], dtype=tf.float32)
    labels_neg = tf.zeros([len(neg_files)], dtype=tf.float32)

    ds_pos = tf.data.Dataset.from_tensor_slices(
        (
            list(f[0] for f in pos_files),
            list(f[1] for f in pos_files),
            list(f[2] for f in pos_files),
            labels_pos,
        )
    )
    if config["exp"]["one_hot"]:
        ds_pos = ds_pos.map(
            lambda f, s, e, y: (
                audio_pipeline((f, s, e), config, augments=False),
                tf.one_hot(tf.cast(y, tf.int32), depth=2),
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    else:
        ds_pos = ds_pos.map(
            lambda f, s, e, y: (audio_pipeline((f, s, e), config, augments=False), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    ds_neg = tf.data.Dataset.from_tensor_slices(
        (
            list(f[0] for f in neg_files),
            list(f[1] for f in neg_files),
            list(f[2] for f in neg_files),
            labels_neg,
        )
    )
    if config["exp"]["one_hot"]:
        ds_neg = ds_neg.map(
            lambda f, s, e, y: (
                audio_pipeline((f, s, e), config, augments=False),
                tf.one_hot(tf.cast(y, tf.int32), depth=2),
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    else:
        ds_neg = ds_neg.map(
            lambda f, s, e, y: (
                audio_pipeline((f, s, e), config, augments=False),
                y,
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    return ds_pos.concatenate(ds_neg).batch(config["ml"]["batch_size"]).cache()


def make_epoch_train_dataset(
    pos_tr_all: Tuple[Sequence[str], float, float],
    neg_tr_all: Tuple[Sequence[str], float, float],
    config: Dict,
    seed: int,
) -> tf.data.Dataset:
    P_pos, P_neg = plan_epoch_counts(len(pos_tr_all), config["data"]["pos_neg_ratio"])
    neg_epoch = sample_train_negatives(neg_tr_all, P_neg, seed=seed)
    return build_train_dataset(
        pos_tr_all, neg_epoch, config=config, shuffle=True, seed=seed
    )


def make_fixed_test_dataset(
    pos_test_all: Sequence[str],
    neg_test_all: Sequence[str],
    config: Dict,
    seed: int,
):
    neg_fixed = make_fixed_val_negatives(
        neg_test_all,
        len(pos_test_all),
        neg_pos_ratio=config["data"]["pos_neg_ratio"],
        seed=seed,
    )
    return (build_val_dataset(pos_test_all, neg_fixed, config=config), neg_fixed)


def build_file_list(df: pd.DataFrame, idx, target: str):
    """
    Extracts the rows from this folds from the dataframe and collects the information for the tensor.
    It also splits the fold into its positive anbd negative parts
    """
    sub = df.iloc[idx]
    pos = sub[sub["primary_label"] == target].apply(
        lambda x: (x["path"], x["start"], x["end"]), axis=1
    )
    neg = sub[sub["primary_label"] != target].apply(
        lambda x: (x["path"], x["start"], x["end"]), axis=1
    )
    return pos, neg


def build_file_lists(
    df: pd.DataFrame,
    folds: dict,
    config: Dict,
) -> List[Dict]:
    """
    For every fold, it collects the selected items and creates a list of tuples that
    contain (filepath, start, end) in a tensorflow dataset format
    """
    out = []
    for fold in folds:
        fold_id = int(fold["fold"])
        fold_train_idx = fold["train_idx"]
        fold_test_idx = fold["test_idx"]

        # Build the actual file lists

        pos_tr_all, neg_tr_all = build_file_list(
            df,
            fold_train_idx,
            config["exp"]["target"],
        )
        pos_test_all, neg_test_all = build_file_list(
            df, fold_test_idx, config["exp"]["target"]
        )

        # Build the actual tensorflow sets by taking all assigned positives and
        # a subset of the negatives based on the ratio
        # TODO: Test if this can be changed to select new ones every epoch
        train_ds = make_epoch_train_dataset(
            pos_tr_all,
            neg_tr_all,
            config=config,
            seed=fold_id,
        )
        test_ds, test_idx = make_fixed_test_dataset(
            pos_test_all,
            neg_test_all,
            config=config,
            seed=fold_id,
        )

        test_df = []
        for i in test_idx:
            test_df.append(
                df[
                    (df["path"] == i[0]) & (df["start"] == i[1]) & (df["end"] == i[2])
                ].iloc[0]
            )
        for i in pos_test_all:
            test_df.append(
                df[
                    (df["path"] == i[0]) & (df["start"] == i[1]) & (df["end"] == i[2])
                ].iloc[0]
            )

        out.append(
            {
                "fold_id": fold_id,
                "train_ds": train_ds,
                "test_ds": test_ds,
                "test_df": pd.DataFrame(test_df),
                "train_size": (1 + config["data"]["pos_neg_ratio"]) * len(pos_tr_all),
                "test_size": (1 + config["data"]["pos_neg_ratio"]) * len(pos_test_all),
            }
        )
        print(
            f"Pos: {len(pos_tr_all)}, Neg: {len(neg_tr_all)}, Size: {out[-1]['train_size']}"
        )
    return out


def build_final_dataset(df: pd.DataFrame, config: Dict):
    """
    Create the final dataset, this contains all the training samples and
    x times negative samples. A 80-10-10 train, val, test split will be made
    """

    # Build file lists for all files
    all_pos = (
        df[df["primary_label"] == config["exp"]["target"]]
        .apply(lambda x: (x["path"], x["start"], x["end"]), axis=1)
        .tolist()
    )

    all_neg = (
        df[df["primary_label"] != config["exp"]["target"]]
        .apply(lambda x: (x["path"], x["start"], x["end"]), axis=1)
        .tolist()
    )

    all_pos = [(x, 1) for x in all_pos]
    all_neg = [(x, 0) for x in all_neg]

    all_data = all_pos + all_neg

    all_idx = [x[0] for x in all_data]
    all_labels = [x[1] for x in all_data]

    # Split into train and test
    X_trainval, X_test, y_train, y_test = train_test_split(
        all_idx,
        all_labels,
        test_size=0.1,
        random_state=config["exp"]["random_state"],
        stratify=all_labels,
    )

    # Split train further into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_train,
        test_size=0.1,
        random_state=config["exp"]["random_state"],
        stratify=y_train,
    )

    # Recreate the tuples
    X_train = list(zip(X_train, y_train))
    X_val = list(zip(X_val, y_val))
    X_test = list(zip(X_test, y_test))

    pos_train = len([x for x in X_train if x[1] == 1])
    pos_val = len([x for x in X_val if x[1] == 1])
    pos_test = len([x for x in X_test if x[1] == 1])

    logging.info("Final training set")
    logging.info(
        f"Train: {len(X_train)}, Pos: {pos_train}, Neg: {len([x for x in X_train if x[1] == 0])}"
    )
    logging.info(
        f"Validation: {len(X_val)}, Pos: {pos_val}, Neg: {len([x for x in X_val if x[1] == 0])}"
    )
    logging.info(
        f"Testing: {len(X_test)}, Pos {pos_test}, Neg: {len([x for x in X_test if x[1] == 0])}"
    )

    train_ds = make_epoch_train_dataset(
        [x[0] for x in X_train if x[1] == 1],
        [x[0] for x in X_train if x[1] == 0],
        config=config,
        seed=config["exp"]["random_state"],
    )
    val_ds, _ = make_fixed_test_dataset(
        [x[0] for x in X_val if x[1] == 1],
        [x[0] for x in X_val if x[1] == 0],
        config=config,
        seed=config["exp"]["random_state"],
    )
    test_ds, _ = make_fixed_test_dataset(
        [x[0] for x in X_test if x[1] == 1],
        [x[0] for x in X_test if x[1] == 0],
        config=config,
        seed=config["exp"]["random_state"],
    )

    return {
        "train_ds": train_ds,
        "val_ds": val_ds,
        "test_ds": test_ds,
        "train_size": (1 + config["data"]["pos_neg_ratio"]) * pos_train,
        "val_size": (1 + config["data"]["pos_neg_ratio"]) * pos_val,
        "test_size": (1 + config["data"]["pos_neg_ratio"]) * pos_test,
    }


def make_soundscape_dataset(df: pd.DataFrame, config: Dict):

    all_pos = (
        (df[df["primary_label"] == config["exp"]["target"]])
        .apply(lambda x: (x["path"], x["start"], x["end"]), axis=1)
        .tolist()
    )

    all_neg = (
        (df[df["primary_label"] != config["exp"]["target"]])
        .apply(lambda x: (x["path"], x["start"], x["end"]), axis=1)
        .tolist()
    )

    soundscape_ds = build_val_dataset(all_pos, all_neg, config)

    return soundscape_ds
