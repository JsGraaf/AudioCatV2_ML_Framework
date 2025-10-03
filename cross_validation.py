from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit


def make_cv_splits(
    df: pd.DataFrame,
    target: str,
    n_splits: int,
    random_state: int,
):
    """
    Build nested CV folds for binary BirdCLEF: target species = 1, others = 0.
    Uses StratifiedGroupKFold for both outer and inner splits to avoid leakage across groups.
    """

    # Binary labels for stratification
    y_all = (df["primary_label"] == target).astype(int).values
    groups_all = df["group_key"].astype(str).fillna("NA").values
    idx_all = np.arange(len(df))

    folder = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )

    splits = []
    for k, (tr_idx, te_idx) in enumerate(
        folder.split(idx_all, y=y_all, groups=groups_all), start=1
    ):
        # Outer train/validation and test sets
        trval_idx = idx_all[tr_idx]
        test_idx = idx_all[te_idx]

        splits.append(
            {
                "fold": k,
                "train_idx": trval_idx,
                "test_idx": test_idx,
            }
        )

    return splits


def make_cv_splits_with_validation(
    df: pd.DataFrame,
    target: str,
    n_splits: int,
    random_state: int,
):
    """
    Build CV folds for binary BirdCLEF with group-aware stratification.

    For each outer fold:
      - Outer: StratifiedGroupKFold -> ~80% train, ~20% hold-out (if n_splits=5)
      - Inner-on-holdout: StratifiedGroupKFold(n_splits=2) -> split hold-out
        into equal halves: ~10% validation, ~10% test.

    Notes
    -----
    * To achieve ~80/20 train/(val+test), use n_splits=5.
    * Inner split preserves both stratification and groups on the outer hold-out.
    * If the inner stratified split fails (too few positives/groups), we fall back
      to GroupShuffleSplit (group-aware but not stratified).
    """

    # Binary labels + groups
    y_all = (df["primary_label"] == target).astype(int).values
    groups_all = df["group_key"].astype(str).fillna("NA").values
    idx_all = np.arange(len(df))

    outer = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )

    splits = []
    for k, (tr_idx, hold_idx) in enumerate(
        outer.split(idx_all, y=y_all, groups=groups_all), start=1
    ):
        tr_idx = np.asarray(tr_idx)
        hold_idx = np.asarray(hold_idx)

        # Prepare labels/groups for the outer hold-out only
        y_hold = y_all[hold_idx]
        g_hold = groups_all[hold_idx]
        i_hold = np.arange(len(hold_idx))  # local indices inside hold-out

        # Try a stratified+grouped 2-fold split on the hold-out to get val/test
        inner = StratifiedGroupKFold(
            n_splits=2, shuffle=True, random_state=random_state + k
        )
        got_inner = False
        try:
            # Take the FIRST split: (inner_tr, inner_te)
            # We'll use inner_tr as validation and inner_te as test (or vice-versa).
            inner_splits = list(inner.split(i_hold, y=y_hold, groups=g_hold))
            if len(inner_splits) >= 1:
                val_local, test_local = inner_splits[0][0], inner_splits[0][1]
                val_idx = hold_idx[val_local]
                test_idx = hold_idx[test_local]
                got_inner = True
        except Exception:
            got_inner = False

        # Fallback: group-aware 50/50 on hold-out (not stratified)
        if not got_inner:
            gss = GroupShuffleSplit(
                n_splits=1, test_size=0.5, random_state=random_state + k
            )
            ((val_local, test_local),) = gss.split(i_hold, groups=g_hold)
            val_idx = hold_idx[val_local]
            test_idx = hold_idx[test_local]

        # Sanity: sort indices for reproducibility
        tr_idx = np.sort(tr_idx)
        val_idx = np.sort(val_idx)
        test_idx = np.sort(test_idx)

        # Metrics summary per split
        def pos_frac(ix):
            n = len(ix)
            return float((y_all[ix] == 1).sum()) / n if n > 0 else 0.0

        split = {
            "fold": k,
            "train_idx": tr_idx,
            "val_idx": val_idx,
            "test_idx": test_idx,
        }
        splits.append(split)

    return splits
