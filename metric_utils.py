import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
)

plt.rcParams.update({"font.size": 20})


def threshold_at_precision(y_true, y_score, target=0.90):
    """Return the highest-recall threshold achieving >= target precision.
    y_true: (N,) in {0,1}
    y_score: (N,) sigmoid probabilities/logits passed through sigmoid.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    # precision/recall are length M, thresholds length M-1 (aligned to precision[1:])
    precision = precision[1:]
    recall = recall[1:]
    ok = np.where(precision >= target)[0]
    if ok.size == 0:
        # No threshold reaches the target precision; fall back to best achievable precision
        idx = np.argmax(precision)
        return thresholds[idx], precision[idx], recall[idx], False
    # among those, pick the one with max recall
    idx = ok[np.argmax(recall[ok])]
    return thresholds[idx], precision[idx], recall[idx], True


def get_predictions(dataset, predictions):
    y_true_batches = []
    for _, yb in dataset:
        y_true_batches.append(yb.numpy())
    y_true = np.concatenate(y_true_batches, axis=0)

    y_pred = predictions  # shape (N,)

    df_pred = pd.DataFrame(
        {
            "y_true": y_true.squeeze(),
            "y_pred": y_pred.squeeze(),  # keep the score for ROC/PR or threshold sweeping
        }
    )
    return df_pred


def confusion_at_threshold(y_true, y_prob, t):
    y_pred = (y_prob >= t).astype(np.int32)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    return dict(tp=tp, tn=tn, fp=fp, fn=fn)


def plot_confusion_matrix_preprocessed(
    cm,
    labels,
    class_names=None,
    normalize=True,
    cmap="Blues",
    title="",
):
    """
    Plot a confusion matrix with fixed shape using all known labels.
    The axes will be ordered as: target, other (x: left→right, y: bottom→top).
    """
    if class_names is None:
        class_names = [str(l) for l in labels]

    plt.rcParams.update({"font.size": 22})

    # Ensure labels and class_names are in the desired order: target, other
    # Reverse both axes so that y-axis is bottom-to-top: target, other
    # and x-axis is left-to-right: target, other
    cm = np.asarray(cm)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError("`cm` must be a square 2D array.")
    n = cm.shape[0]
    if labels is None:
        labels = list(range(n))

    # Reverse the order for both axes (if needed)
    # Here, we assume the desired order is [target, other], i.e., [1, 0]
    # So, we reverse if the current order is [0, 1]
    if class_names[0] != labels[0]:
        cm = cm[::-1, ::-1]
        class_names = class_names[::-1]
        labels = labels[::-1]

    if normalize:
        row_sums = cm.sum(axis=0, keepdims=True)
        cm_norm = cm.astype(np.float32) / np.where(row_sums == 0, 1, row_sums) * 100
        cm_norm = np.nan_to_num(cm_norm, nan=0.0)

    fig, ax = plt.subplots(figsize=(11, 9))
    if normalize:
        im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    else:
        im = ax.imshow(cm, cmap=cmap, vmin=0, vmax=np.max(cm), aspect="auto")

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(
        "Counts" if not normalize else "Proportion(%)", rotation=-90, va="bottom"
    )
    if normalize:
        cbar.ax.set_ylim(0, 100)
    else:
        cbar.ax.set_ylim(0, 1)

    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="Predicted label",
        xlabel="True label",
    )
    plt.setp(ax.get_xticklabels(), rotation=0)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            if normalize:
                val_norm = cm_norm[i, j]
                text_val = f"{val_norm:.2f}% \n({val})"
            else:
                text_val = str(int(val))

            # Get background color from the colormap
            if normalize:
                rgba = plt.get_cmap(cmap)(
                    val_norm / cm_norm.max() if cm_norm.max() > 0 else 0
                )
            else:
                rgba = plt.get_cmap(cmap)(val / cm.max() if cm.max() > 0 else 0)
            r, g, b = rgba[:3]
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            text_color = "black" if luminance > 0.5 else "white"

            ax.text(j, i, text_val, ha="center", va="center", color=text_color)

    fig.tight_layout()
    ax.set_title(title)
    return fig, ax


def plot_pr_with_thresholds(
    y_true, y_score, marks=(0.01, 1.0), pick=("min_precision", 0.7)
):
    """
    y_true:  (N,) array of {0,1}
    y_score: (N,) array of predicted probabilities for class 1
    marks:   iterable of thresholds to highlight on the curve
    pick:    'max_f1' or ('min_precision', value) to choose a threshold
    """
    # PR curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_score, pos_label=1)
    ap = average_precision_score(y_true, y_score)

    # thresholds has length len(precision) - 1; the (p,r) at index i+1 corresponds to thresholds[i]
    pr_points = []
    for i, t in enumerate(thresholds):
        pr_points.append((t, precision[i + 1], recall[i + 1]))
    pr_points = np.array(pr_points)  # shape (K, 3): [t, P, R]

    # Helper: metrics at threshold t
    def metrics_at_threshold(t):
        y_pred = (y_score >= t).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=[0, 1], average=None, zero_division=0.0
        )
        return prec[1], rec[1], f1[1]

    # Choose a threshold
    if pick == "max_f1":
        # Sweep same thresholds used in PR curve
        f1_vals = []
        for t in thresholds:
            p, r, f1 = metrics_at_threshold(t)
            f1_vals.append(f1)
        best_idx = int(np.argmax(f1_vals))
        best_t = thresholds[best_idx]
        best_p, best_r, best_f1 = metrics_at_threshold(best_t)
        chosen = ("Best F1", best_t, best_p, best_r, best_f1)
    elif isinstance(pick, tuple) and pick[0] == "min_precision":
        min_p = float(pick[1])
        # Among thresholds meeting precision≥min_p, pick the one with max recall
        candidates = []
        for t in thresholds:
            p, r, f1 = metrics_at_threshold(t)
            if p >= min_p:
                candidates.append((r, p, f1, t))
        if candidates:
            r, p, f1, best_t = max(candidates)  # max recall
            best_p, best_r, best_f1 = p, r, f1
            chosen = (f"P≥{min_p}", best_t, best_p, best_r, best_f1)
        else:
            chosen = (f"P≥{min_p}", None, None, None, None)
    else:
        raise ValueError("pick must be 'max_f1' or ('min_precision', value)")

    # Compute and print metrics at requested marks and chosen threshold
    print(f"Average Precision (AP): {ap:.4f}")
    if chosen[1] is not None:
        print(
            f"{chosen[0]} at t={chosen[1]:.3f}: P={chosen[2]:.3f}, R={chosen[3]:.3f}, F1={chosen[4]:.3f}"
        )
    for t in marks:
        p, r, f1 = metrics_at_threshold(t)
        print(f"t={t:.3f}  →  P={p:.3f}, R={r:.3f}, F1={f1:.3f}")

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, lw=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision–Recall (AP={ap:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle=":", alpha=0.6)

    # Mark: best/constraint threshold
    if chosen[1] is not None:
        # Find nearest PR point for chosen threshold
        idx = np.argmin(np.abs(pr_points[:, 0] - chosen[1]))
        ax.scatter(pr_points[idx, 2], pr_points[idx, 1], s=60)  # (recall, precision)
        ax.annotate(
            f"{chosen[0]}\nt={chosen[1]:.3f}",
            (pr_points[idx, 2], pr_points[idx, 1]),
            textcoords="offset points",
            xytext=(6, -12),
        )

    # Mark: specific thresholds (e.g., 0.01 and 1.0)
    for t in marks:
        idx = np.argmin(np.abs(pr_points[:, 0] - t))
        ax.scatter(pr_points[idx, 2], pr_points[idx, 1], s=50)
        ax.annotate(
            f"t={t:.2f}",
            (pr_points[idx, 2], pr_points[idx, 1]),
            textcoords="offset points",
            xytext=(6, 6),
        )

    plt.show()


def get_scores_per_class_one_hot(df: pd.DataFrame, min_precision: float = 0.3):
    best_scores = {}

    # Prepare score array
    print(df["true_label"])
    for c in df["true_label"].unique():
        best_scores[c] = {
            "precision": {"t": 0, "val": -1},
            "recall": {"t": 0, "val": -1},
            "f1": {"t": 0, "val": -1},
            "p90": {"t": 0, "val": -1},
        }

    for t in np.arange(0.01, 1.0, 0.01):
        df["predicted_label"] = df["p_class1"].apply(lambda x: 1 if x >= t else 0)

        # Calculate the overall accuracy
        acc = (df[df["predicted_label"] == df["true_label"]].shape[0] / df.shape[0],)

        report = classification_report(
            df["true_label"],
            df["predicted_label"],
            zero_division=0.0,
        )

        # Calculate the scores
        prec, rec, f1, support = precision_recall_fscore_support(
            df["true_label"],
            df["predicted_label"],
            labels=[0, 1],
            average=None,
            zero_division=0.0,
        )

        for c, p, r, f1 in zip([0, 1], prec, rec, f1):
            # Search for the best metrics
            if p >= best_scores[c]["precision"]["val"]:
                if (
                    p > (best_scores[c]["precision"]["val"])
                    or r > best_scores[c]["recall"]["val"]
                ):
                    best_scores[c]["precision"]["t"] = t
                    best_scores[c]["precision"]["val"] = p
                    best_scores[c]["precision"]["recall"] = r
            if r >= best_scores[c]["recall"]["val"]:
                if (
                    r > best_scores[c]["recall"]["val"]
                    or p > best_scores[c]["precision"]["val"]
                ):
                    best_scores[c]["recall"]["t"] = t
                    best_scores[c]["recall"]["val"] = r
                    best_scores[c]["recall"]["precision"] = p
            if f1 > best_scores[c]["f1"]["val"]:
                best_scores[c]["f1"]["t"] = t
                best_scores[c]["f1"]["val"] = f1
            # Search for recall at p90
            if p >= 0.7 and r > best_scores[c]["p90"]["val"]:
                best_scores[c]["p90"]["t"] = t
                best_scores[c]["p90"]["val"] = r
                best_scores[c]["p90"]["precision"] = p

    return best_scores


def get_scores_per_class(df: pd.DataFrame, min_precision: float = 0.9):
    best_scores = {}

    # Prepare score array
    print(df["true_label"].unique())
    for c in df["true_label"].unique():
        best_scores[c] = {
            "precision": {"t": 0, "val": -1},
            "recall": {"t": 0, "val": -1},
            "f1": {"t": 0, "val": -1},
            "p90": {"t": 0, "val": -1},
        }

    for t in np.arange(0.01, 1.0, 0.01):
        df["predicted_label"] = df["predictions"].apply(lambda x: 1 if x >= t else 0)

        # Calculate the overall accuracy
        acc = (df[df["predicted_label"] == df["true_label"]].shape[0] / df.shape[0],)

        report = classification_report(
            df["true_label"],
            df["predicted_label"],
            zero_division=0.0,
        )

        # Calculate the scores
        prec, rec, f1, support = precision_recall_fscore_support(
            df["true_label"],
            df["predicted_label"],
            labels=[0, 1],
            average=None,
            zero_division=0.0,
        )

        for c, p, r, f1 in zip([0, 1], prec, rec, f1):
            # Search for the best metrics
            if p >= best_scores[c]["precision"]["val"]:
                if (
                    p > (best_scores[c]["precision"]["val"])
                    or r > best_scores[c]["recall"]["val"]
                ):
                    best_scores[c]["precision"]["t"] = t
                    best_scores[c]["precision"]["val"] = p
                    best_scores[c]["precision"]["recall"] = r
            if r >= best_scores[c]["recall"]["val"]:
                if (
                    r > best_scores[c]["recall"]["val"]
                    or p > best_scores[c]["precision"]["val"]
                ):
                    best_scores[c]["recall"]["t"] = t
                    best_scores[c]["recall"]["val"] = r
                    best_scores[c]["recall"]["precision"] = p
            if f1 > best_scores[c]["f1"]["val"]:
                best_scores[c]["f1"]["t"] = t
                best_scores[c]["f1"]["val"] = f1
            # Search for recall at p90
            if p >= min_precision and r > best_scores[c]["p90"]["val"]:
                best_scores[c]["p90"]["t"] = t
                best_scores[c]["p90"]["val"] = r
                best_scores[c]["p90"]["precision"] = p

    return best_scores
