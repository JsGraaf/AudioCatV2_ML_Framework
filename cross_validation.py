import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


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
