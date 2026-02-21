import logging
import random
from typing import Dict, List, Sequence, Tuple

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from audio_processing import audio_pipeline


def _sample_beta(alpha: float, dtype=tf.float32) -> tf.Tensor:
    """Scalar λ ~ Beta(alpha, alpha) without tfp."""
    if alpha <= 0.0:
        return tf.constant(1.0, dtype)
    a = tf.constant(alpha, dtype)
    g1 = tf.random.gamma(shape=[], alpha=a, dtype=dtype)  # Gamma(a,1)
    g2 = tf.random.gamma(shape=[], alpha=a, dtype=dtype)
    return g1 / (g1 + g2 + 1e-8)


def _mixup_batch(
    x: tf.Tensor, y: tf.Tensor, alpha: float
) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Pairwise Mixup on a batched (x,y).
    x: [B,H,W,C] or [B,H,W] -> returns [B,H,W,C]
    y: [B] (binary) or [B,K] (one-hot / multi-label)
    """
    y = tf.cast(y, tf.float32)

    b = tf.shape(x)[0]

    def no_mix():
        return x, y

    def do_mix():
        idx = tf.random.shuffle(tf.range(b))
        x2 = tf.gather(x, idx, axis=0)
        y2 = tf.gather(y, idx, axis=0)

        lam = _sample_beta(alpha, dtype=x.dtype)  # scalar λ
        xm = lam * x + (1.0 - lam) * x2
        ym = lam * y + (1.0 - lam) * y2
        return xm, ym

    return tf.cond(b > 1, do_mix, no_mix)


def mixup_map_fn(alpha: float, prob: float = 1.0):
    """
    tf.data map-fn applying Mixup with probability `prob` per batch.
    Place AFTER .batch(...) and BEFORE .prefetch(...).
    """

    def _fn(x, y):
        y = tf.cast(y, tf.float32)
        if prob >= 1.0:
            return _mixup_batch(x, y, alpha)
        r = tf.random.uniform([])
        return tf.cond(r < prob, lambda: _mixup_batch(x, y, alpha), lambda: (x, y))

    return _fn


def _sample_beta_batch(alpha: float, b: tf.Tensor, dtype) -> tf.Tensor:
    """[B] ~ Beta(alpha, alpha) without tfp."""
    if alpha <= 0.0:
        return tf.ones([b], dtype)
    a = tf.cast(alpha, dtype)
    g1 = tf.random.gamma(shape=[b], alpha=a, dtype=dtype)
    g2 = tf.random.gamma(shape=[b], alpha=a, dtype=dtype)
    return g1 / (g1 + g2 + 1e-8)  # [B]


def _mixup_batch_per_sample(x: tf.Tensor, y: tf.Tensor, alpha: float):
    """
    Pairwise MixUp with a different lambda per example.
    x: [B,H,W] or [B,H,W,C]
    y: [B] or [B,K]
    """
    x = tf.convert_to_tensor(x)
    y = tf.cast(y, tf.float32)
    b = tf.shape(x)[0]

    idx = tf.random.shuffle(tf.range(b))
    x2 = tf.gather(x, idx, axis=0)
    y2 = tf.gather(y, idx, axis=0)

    lam = _sample_beta_batch(alpha, b, x.dtype)  # [B]
    lam = tf.maximum(lam, 1.0 - lam)

    # Broadcast lam across image dims
    if x.shape.rank == 4:
        lam_x = tf.reshape(lam, [b, 1, 1, 1])
    else:  # [B,H,W]
        lam_x = tf.reshape(lam, [b, 1, 1])

    xm = lam_x * x + (1.0 - lam_x) * x2

    if y.shape.rank == 1:
        ym = lam * y + (1.0 - lam) * y2  # [B]
    else:
        ym = lam[:, None] * y + (1.0 - lam)[:, None] * y2  # [B,K]
    return xm, ym


def mixup_map_fn_per_sample(alpha: float, prob: float = 1.0):
    """tf.data map-fn: per-sample lambda; apply with probability `prob` per batch."""

    def _fn(x, y):
        r = tf.random.uniform([])
        return tf.cond(
            r < prob,
            lambda: _mixup_batch_per_sample(x, y, alpha),
            lambda: (x, tf.cast(y, tf.float32)),
        )

    return _fn


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

    ds = ds.batch(config["ml"]["batch_size"]).cache()

    # Conditional mixup
    mix_cfg = config["ml"].get("mixup", {})
    if mix_cfg.get("prob", 0.0) > 0.0:
        alpha = float(mix_cfg.get("alpha", 0.4))  # typical 0.2–0.4
        prob = float(mix_cfg.get("prob", 1.0))  # chance to apply per batch

        if prob >= 1.0:
            ds = ds.map(
                mixup_map_fn_per_sample(alpha), num_parallel_calls=tf.data.AUTOTUNE
            )
        else:
            # Randomly apply with probability 'prob'
            def maybe_mix(x, y):
                do = tf.less(tf.random.uniform([], 0, 1), prob)
                xm, ym = _mixup_batch_per_sample(x, y, alpha)
                return tf.cond(do, lambda: (xm, ym), lambda: (x, y))

            ds = ds.map(maybe_mix, num_parallel_calls=tf.data.AUTOTUNE)

    # Prefetch stays last for performance
    ds = ds.prefetch(tf.data.AUTOTUNE)
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
        fold_val_idx = fold["val_idx"]
        fold_test_idx = fold["test_idx"]

        # Build the actual file lists
        pos_tr_all, neg_tr_all = build_file_list(
            df,
            fold_train_idx,
            config["exp"]["target"],
        )
        pos_val_all, neg_val_all = build_file_list(
            df, fold_val_idx, config["exp"]["target"]
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

        val_ds, _ = make_fixed_test_dataset(
            pos_val_all,
            neg_val_all,
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
                "val_ds": val_ds,
                "test_ds": test_ds,
                "test_df": pd.DataFrame(test_df),
                "train_size": (1 + config["data"]["pos_neg_ratio"]) * len(pos_tr_all),
                "val_size": (1 + config["data"]["pos_neg_ratio"]) * len(pos_val_all),
                "test_size": (1 + config["data"]["pos_neg_ratio"]) * len(pos_test_all),
            }
        )
        print(
            f"Pos: {len(pos_tr_all)}, Neg: {len(neg_tr_all)}, Size: {out[-1]['train_size']}"
        )
    return out


def build_final_dataset(df: pd.DataFrame, config: Dict, percentage: float = 1.0):
    """
    Create the final dataset, this contains all the training samples and
    x times negative samples. A 80-10-10 train, val, test split will be made.
    If percentage < 1.0, only use a percentage of the entire dataset
    """

    # Build file lists for all files
    all_pos = (
        df[df["primary_label"] == config["exp"]["target"]]
        .apply(
            lambda x: ((x["path"], x["start"], x["end"]), x["primary_label"]), axis=1
        )
        .tolist()
    )

    all_neg = (
        df[df["primary_label"] != config["exp"]["target"]]
        .apply(
            lambda x: ((x["path"], x["start"], x["end"]), x["primary_label"]), axis=1
        )
        .tolist()
    )

    # Take only a percentage of the data if requested
    # Since we calculate the amount of negatives based on the positives,
    # we only need to limit the positives
    if percentage < 1.0:
        n_pos = int(len(all_pos) * percentage)
        random.seed(config["exp"]["random_state"])
        all_pos = random.sample(all_pos, n_pos) if n_pos > 0 else []

    all_data = all_pos + all_neg
    all_idx = [x[0] for x in all_data]
    all_labels = [x[1] for x in all_data]

    # Split into train and test/val
    X_train, X_testval, y_train, y_testval = train_test_split(
        all_idx,
        all_labels,
        test_size=0.2,
        random_state=config["exp"]["random_state"],
        stratify=all_labels,
    )

    # Split testval into test and validation
    X_val, X_test, y_val, y_test = train_test_split(
        X_testval,
        y_testval,
        test_size=0.5,
        random_state=config["exp"]["random_state"],
        stratify=y_testval,
    )

    # Recreate the tuples
    X_train = list(
        zip(X_train, [1 if x == config["exp"]["target"] else 0 for x in y_train])
    )
    X_val = list(zip(X_val, [1 if x == config["exp"]["target"] else 0 for x in y_val]))
    X_test = list(
        zip(X_test, [1 if x == config["exp"]["target"] else 0 for x in y_test])
    )

    pos_train = len([x for x in X_train if x[1] == 1])
    pos_val = len([x for x in X_val if x[1] == 1])
    pos_test = len([x for x in X_test if x[1] == 1])

    logging.info("Final training set")
    logging.info(
        f"Train: {len(X_train)}, Pos: {pos_train}, Neg: {len([x for x in X_train if x[1] == 0])}, Used: {(1 + config['data']['pos_neg_ratio']) * pos_train}"
    )
    logging.info(
        f"Validation: {len(X_val)}, Pos: {pos_val}, Neg: {len([x for x in X_val if x[1] == 0])}, Used: {(1 + config['data']['pos_neg_ratio']) * pos_val}"
    )
    logging.info(
        f"Testing: {len(X_test)}, Pos {pos_test}, Neg: {len([x for x in X_test if x[1] == 0])}, Used: {(1 + config['data']['pos_neg_ratio']) * pos_test}"
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
