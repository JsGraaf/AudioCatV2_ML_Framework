import argparse
import glob
import os
import pprint
import re
from pathlib import Path
from typing import Dict, List, Tuple

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

    if not soundscapes:
        print("No results parsed from soundscapes.")
        return

    target_key = 1

    metrics = ["f1", "recall", "precision"]

    # Prepare per-metric sorted lists of (exp_name, value)
    data_by_metric = {}
    for m in metrics:
        items = []
        for exp, scores in soundscapes.items():
            v = max(scores.get(target_key, {}).get(m, {}).get("val", 0.0), 0.0)
            items.append((exp, float(v)))
        # sort descending so highest at top
        items.sort(key=lambda kv: kv[1], reverse=True)
        data_by_metric[m] = items

    # --- Plot: 3 horizontal bar charts, one per metric (stacked vertically for readability) ---
    n_rows, n_cols = 3, 1
    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(10, max(6, 0.4 * len(soundscapes) * n_rows)),
        squeeze=False,
    )
    axs = axs.ravel()

    for i, m in enumerate(metrics):
        ax = axs[i]
        items = data_by_metric[m]
        exp_names = [kv[0] for kv in items]
        values = [kv[1] for kv in items]

        y_pos = np.arange(len(exp_names))
        bars = ax.barh(y_pos, values)

        # Labels: experiment names on y-axis
        ax.set_yticks(y_pos)
        ax.set_yticklabels(exp_names)

        # Keep highest at top
        ax.invert_yaxis()

        # X Axis
        ax.set_xlim(0, 1)
        ax.set_xlabel(m.upper())
        ax.grid(True, axis="x", linestyle=":", alpha=0.5)

        # Annotate value at the end of each bar
        for rect, v in zip(bars, values):
            x = rect.get_width()
            y = rect.get_y() + rect.get_height() / 2
            ax.text(x + 0.01, y, f"{v:.2f}", va="center", ha="left", fontsize=9)

        ax.set_title(f"Target Class — {m.upper()}")

    fig.tight_layout()
    plt.show()


# def multiple_soundscapes(dirpath: Path):
#     # Try to find all soundscape_predictions.csv in (sub)directories
#     soundscapes = {}
#     files = glob.glob(os.path.join(dirpath, "*/soundscape_predictions.csv"))
#     if not files:
#         print("No soundscape_predictions.csv found")
#         return
#
#     for f in files:
#         exp_name = Path(f).parent.name  # robust experiment name
#         best_scores = single_soundscape(Path(f), do_print=False)
#         soundscapes[exp_name] = best_scores
#     pp.pprint(soundscapes)
#
#     # Collect classes and metrics
#     classes = [0, 1]  # or list(next(iter(soundscapes.values())).keys())
#     metrics = ["f1", "recall", "precision"]
#
#     n_rows = len(metrics)
#     n_cols = len(classes)
#     fig, axs = plt.subplots(
#         nrows=n_rows, ncols=n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False
#     )
#
#     # --- Sort experiment names alphabetically ---
#     exp_names = sorted(soundscapes.keys())
#     n_exps = len(exp_names)
#     colors = plt.cm.tab10.colors  # up to 10 distinct colors
#
#     # Use a wider bar width so the group fills ~75% of the axis
#     total_width = 0.75
#     bar_width = total_width / n_exps
#     x0 = np.arange(1)  # only one group per subplot
#
#     for row, m in enumerate(metrics):
#         for col, c in enumerate(classes):
#             ax = axs[row][col]
#             for j, exp in enumerate(exp_names):
#                 v = soundscapes[exp].get(c, {}).get(m, {}).get("val", 0.0)
#                 xpos = x0 + j * bar_width - total_width / 2
#                 # Use a no-legend label so axes don't collect entries
#                 ax.bar(
#                     xpos,
#                     v,
#                     width=bar_width,
#                     color=colors[j % len(colors)],
#                     label="_nolegend_",
#                 )
#                 ax.text(
#                     xpos,
#                     v / 2,
#                     f"{v:.2f}",
#                     ha="center",
#                     va="center",
#                     fontsize=9,
#                     color="white",
#                     fontweight="bold",
#                 )
#
#             ax.set_ylim(0, 1)
#             ax.set_xticks([])
#             ax.set_ylabel(m)
#             ax.set_title(f"Class {c} — {m.upper()}")
#             ax.grid(True, axis="y", linestyle=":", alpha=0.5)
#
#     # --- Build proxy legend handles once, then place figure legend ---
#     legend_handles = [
#         Patch(facecolor=colors[j % len(colors)], edgecolor="none", label=exp_names[j])
#         for j in range(len(exp_names))
#     ]
#
#     fig.subplots_adjust(top=0.88)  # leave space for legend
#     fig.legend(handles=legend_handles, loc="upper center", ncol=min(len(exp_names), 6))
#     plt.show()


if __name__ == "__main__":
    if args.input_csv is not None:
        single_soundscape(args.input_csv, do_print=True)
    if args.input_dir is not None:
        multiple_soundscapes(args.input_dir)
