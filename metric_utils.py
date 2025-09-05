import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


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
