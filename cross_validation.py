from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


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
                "train_pos_ratio": (y_all[trval_idx] == 1).sum(),
                "test_pos_ratio": (y_all[test_idx] == 1).sum(),
            }
        )

    return splits


def simple_group_stratified_holdout_and_cv(
    df: pd.DataFrame,
    target: str,
    random_state: int = 42,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
    cv_splits: int = 5,
) -> Dict:
    """
    80/10/10 scheme with group-aware stratification:
      - Use StratifiedGroupKFold with n_splits ~= 1/(val_frac+test_frac) to take a 20% holdout.
      - Split that holdout at the GROUP level into val/test with a stratified shuffle (50/50).
      - Run SGKF cross-validation on the remaining 80%.

    Requires df columns:
      - 'primary_label' (str): class label per sample
      - 'group_key'     (str): grouping id to prevent leakage (e.g., recording/site)
    """
    total_holdout = val_frac + test_frac
    if not (0 < val_frac < 1 and 0 < test_frac < 1 and total_holdout < 1):
        raise ValueError("val_frac and test_frac must be in (0,1) and sum to < 1.")

    y_all = (df["primary_label"] == target).astype(int).to_numpy()
    groups_all = df["group_key"].astype(str).fillna("NA").to_numpy()
    idx_all = np.arange(len(df))

    # 1) Take ~20% holdout with SGKF (pick the first split)
    outer_n = max(2, int(round(1.0 / total_holdout)))  # e.g., 0.2 -> 5
    outer = StratifiedGroupKFold(
        n_splits=outer_n, shuffle=True, random_state=random_state
    )
    train_pool_pos, holdout_pos = next(outer.split(idx_all, y=y_all, groups=groups_all))
    train_pool_idx = idx_all[train_pool_pos]
    holdout_idx = idx_all[holdout_pos]

    # 2) Stratified *group-level* split of holdout into 50/50 -> 10% val, 10% test
    # Build one row per group inside the holdout with a binary label indicating if group contains any positives.
    hold_groups = pd.DataFrame(
        {
            "group": groups_all[holdout_idx],
            "y": y_all[holdout_idx],
        }
    )
    grp_agg = (
        hold_groups.groupby("group")["y"].max().reset_index()
    )  # group label: any positive in group
    grp_labels = grp_agg["y"].to_numpy()
    grp_ids = grp_agg["group"].to_numpy()

    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=0.5, random_state=random_state + 1
    )
    grp_train, grp_test = next(sss.split(grp_ids, grp_labels))
    val_groups = set(grp_ids[grp_train])  # 50% of holdout -> validation groups
    test_groups = set(grp_ids[grp_test])  # 50% of holdout -> test groups

    val_mask = np.array([g in val_groups for g in groups_all[holdout_idx]])
    test_mask = np.array([g in test_groups for g in groups_all[holdout_idx]])
    val_idx = holdout_idx[val_mask]
    test_idx = holdout_idx[test_mask]

    # 3) CV on the remaining 80% (train_pool) with SGKF
    y_tr = y_all[train_pool_idx]
    g_tr = groups_all[train_pool_idx]
    cv = StratifiedGroupKFold(
        n_splits=cv_splits, shuffle=True, random_state=random_state + 2
    )

    cv_folds: List[Dict] = []
    for i, (tr_pos, va_pos) in enumerate(
        cv.split(train_pool_idx, y=y_tr, groups=g_tr), 1
    ):
        tr_idx = train_pool_idx[tr_pos]
        va_idx = train_pool_idx[va_pos]
        cv_folds.append(
            {
                "fold": i,
                "train_idx": tr_idx,
                "val_idx": va_idx,
                "counts": {
                    "train_pos": int(y_all[tr_idx].sum()),
                    "train_neg": len(tr_idx) - int(y_all[tr_idx].sum()),
                    "val_pos": int(y_all[va_idx].sum()),
                    "val_neg": len(va_idx) - int(y_all[va_idx].sum()),
                },
            }
        )

    # Sanity counts for fixed val/test
    val_pos = int(y_all[val_idx].sum())
    test_pos = int(y_all[test_idx].sum())

    return {
        "train_pool_idx": train_pool_idx,  # 80%
        "fixed_val_idx": val_idx,  # 10%
        "fixed_test_idx": test_idx,  # 10%
        "fixed_counts": {
            "val_pos": val_pos,
            "val_neg": len(val_idx) - val_pos,
            "test_pos": test_pos,
            "test_neg": len(test_idx) - test_pos,
        },
        "cv_folds": cv_folds,  # SGKF folds on the 80% pool
    }
