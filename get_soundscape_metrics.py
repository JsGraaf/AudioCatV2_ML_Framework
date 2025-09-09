import argparse
import glob
import os
import pprint
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.metrics import classification_report, precision_recall_fscore_support

from metric_utils import get_scores_per_class

pp = pprint.PrettyPrinter(indent=4)

parser = argparse.ArgumentParser()
parser.add_argument("--input_csv", default=None, help="Path to the soundscape CSV")
parser.add_argument("--input_dir", default=None, help="Path to Experiments folder")


args = parser.parse_args()


def single_soundscape(path: Path, do_print=False):
    soundscape_df = pd.read_csv(path)

    soundscape_df["true_label"] = soundscape_df["primary_label"].apply(
        lambda x: 1 if x == "rucwar" else 0
    )

    best_scores = get_scores_per_class(soundscape_df)

    if do_print:
        pp.pprint(best_scores)

    soundscape_df["predicted_label"] = soundscape_df["predictions"].apply(
        lambda x: 1 if x >= best_scores[0]["recall"]["t"] else 0
    )

    return best_scores


def multiple_soundscapes(dirpath: Path):
    # Try to find all soundscape_predictions.csv in (sub)directories
    soundscapes = {}
    files = glob.glob(os.path.join(dirpath, "*/soundscape_predictions.csv"))
    if not files:
        print("No soundscape_predictions.csv found")
        return

    for f in files:
        exp_name = Path(f).parent.name  # robust experiment name
        best_scores = single_soundscape(Path(f), do_print=False)
        soundscapes[exp_name] = best_scores
    pp.pprint(soundscapes)

    # Collect classes and metrics
    classes = [0, 1]  # or list(next(iter(soundscapes.values())).keys())
    metrics = ["f1", "recall", "precision"]

    n_rows = len(metrics)
    n_cols = len(classes)
    fig, axs = plt.subplots(
        nrows=n_rows, ncols=n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False
    )

    # --- Sort experiment names alphabetically ---
    exp_names = sorted(soundscapes.keys())
    n_exps = len(exp_names)
    colors = plt.cm.tab10.colors  # up to 10 distinct colors

    # Use a wider bar width so the group fills ~75% of the axis
    total_width = 0.75
    bar_width = total_width / n_exps
    x0 = np.arange(1)  # only one group per subplot

    for row, m in enumerate(metrics):
        for col, c in enumerate(classes):
            ax = axs[row][col]
            for j, exp in enumerate(exp_names):
                v = soundscapes[exp].get(c, {}).get(m, {}).get("val", 0.0)
                xpos = x0 + j * bar_width - total_width / 2
                # Use a no-legend label so axes don't collect entries
                ax.bar(
                    xpos,
                    v,
                    width=bar_width,
                    color=colors[j % len(colors)],
                    label="_nolegend_",
                )
                ax.text(
                    xpos,
                    v / 2,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white",
                    fontweight="bold",
                )

            ax.set_ylim(0, 1)
            ax.set_xticks([])
            ax.set_ylabel(m)
            ax.set_title(f"Class {c} — {m.upper()}")
            ax.grid(True, axis="y", linestyle=":", alpha=0.5)

    # --- Build proxy legend handles once, then place figure legend ---
    legend_handles = [
        Patch(facecolor=colors[j % len(colors)], edgecolor="none", label=exp_names[j])
        for j in range(len(exp_names))
    ]

    fig.subplots_adjust(top=0.88)  # leave space for legend
    fig.legend(handles=legend_handles, loc="upper center", ncol=min(len(exp_names), 6))
    plt.show()


if __name__ == "__main__":
    if args.input_csv is not None:
        single_soundscape(args.input_csv, do_print=True)
    if args.input_dir is not None:
        multiple_soundscapes(args.input_dir)
