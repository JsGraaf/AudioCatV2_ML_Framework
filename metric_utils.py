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


def plot_confusion_matrix(
    y_true, y_pred, labels=None, class_names=None, normalize=True, cmap="Blues"
):
    """
    Plot a confusion matrix with fixed shape using all known labels.
    """
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    if class_names is None:
        class_names = [str(l) for l in labels]

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = cm.astype(np.float32) / np.where(row_sums == 0, 1, row_sums)
        cm = np.nan_to_num(cm, nan=0.0)

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(
        "Counts" if not normalize else "Proportion", rotation=-90, va="bottom"
    )

    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=0)

    fmt = ".2f" if normalize else "d"
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            text_val = f"{val:.2f}" if normalize else f"{int(val)}"

            # Get background color from the colormap
            rgba = plt.get_cmap(cmap)(val / cm.max() if cm.max() > 0 else 0)
            r, g, b = rgba[:3]
            # Perceptual luminance (sRGB)
            luminance = 0.299 * r + 0.587 * g + 0.114 * b

            text_color = "black" if luminance > 0.5 else "white"

            ax.text(j, i, text_val, ha="center", va="center", color=text_color)

    fig.tight_layout()
    return ax


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


def get_scores_per_class(df: pd.DataFrame, min_precision: float = 0.5):
    best_scores = {}

    # Prepare score array
    for c in df["true_label"].unique():
        best_scores[c] = {
            "precision": {"t": 0, "val": -1},
            "recall": {"t": 0, "val": -1},
            "f1": {"t": 0, "val": -1},
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
            if p < min_precision:
                continue
            if p > best_scores[c]["precision"]["val"]:
                best_scores[c]["precision"]["t"] = t
                best_scores[c]["precision"]["val"] = p
            if r > best_scores[c]["recall"]["val"]:
                best_scores[c]["recall"]["t"] = t
                best_scores[c]["recall"]["val"] = r
            if f1 > best_scores[c]["f1"]["val"]:
                best_scores[c]["f1"]["t"] = t
                best_scores[c]["f1"]["val"] = f1

    return best_scores
