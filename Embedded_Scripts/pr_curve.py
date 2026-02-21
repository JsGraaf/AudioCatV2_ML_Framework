#!/usr/bin/env python3
"""
Example: Precision–Recall curve with Average Precision and Recall@P90 marker.

- Generates synthetic logits/probabilities
- Computes precision, recall, thresholds
- Finds the highest recall where precision >= 0.90
- Saves figure to output/pr_curve_example.svg and .png
"""

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

# Reproducibility
RNG = np.random.default_rng(42)

def make_synthetic_scores(n=2000, pos_frac=0.25):
    """Create toy ground-truth labels and calibrated-ish probabilities."""
    y_true = (RNG.random(n) < pos_frac).astype(int)
    # Scores: positives skew higher, negatives lower
    pos_scores = RNG.beta(a=5, b=2, size=(y_true == 1).sum())
    neg_scores = RNG.beta(a=2, b=8, size=(y_true == 0).sum())
    y_scores = np.empty(n, dtype=float)
    y_scores[y_true == 1] = pos_scores
    y_scores[y_true == 0] = neg_scores
    return y_true, y_scores

def recall_at_precision(precision, recall, thresholds, target_p=0.90):
    """
    Return (best_recall, threshold) for the highest recall achieving
    precision >= target_p. If none, returns (nan, nan).
    """
    mask = precision[:-1] >= target_p  # thresholds aligns with precision[:-1], recall[:-1]
    if not np.any(mask):
        return np.nan, np.nan
    idx = np.argmax(recall[:-1][mask])  # among qualifying points, pick max recall
    # map back to original indices
    true_idx = np.flatnonzero(mask)[idx]
    return float(recall[true_idx]), float(thresholds[true_idx])

def main():
    # 1) Synthetic data
    y_true, y_scores = make_synthetic_scores(n=3000, pos_frac=0.2)

    # 2) PR curve + AP
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)

    # 3) Operating point at precision >= 0.90 (P90)
    target_p = 0.90
    r_at_p90, thr_p90 = recall_at_precision(precision, recall, thresholds, target_p)

    # 4) Plot
    fig, ax = plt.subplots(figsize=(6.8, 4.2))

    ax.plot(recall, precision, lw=2, label=f"PR curve (AP = {ap:.3f})")
    ax.fill_between(recall, precision, step="pre", alpha=0.15)

    # Mark P90 point if it exists
    if not np.isnan(r_at_p90):
        # Find the precision value at the same point
        # thresholds aligns with precision[:-1], recall[:-1]
        # so we get the nearest index
        pr_idx = np.searchsorted(thresholds[::-1], thr_p90, side="left")
        # Safer: compute the precision at the threshold directly:
        # pick the index where thresholds == thr_p90, or nearest lower index
        # For visualization, we can simply find the closest recall point to r_at_p90
        idx_closest = int(np.argmin(np.abs(recall - r_at_p90)))
        p_at_p90 = float(precision[idx_closest])

        ax.plot([r_at_p90], [p_at_p90], "o", ms=7, label=f"P≥{int(target_p*100)} op.\nR={r_at_p90:.3f}, τ={thr_p90:.3f}")
        ax.vlines(r_at_p90, 0, p_at_p90, linestyles="dotted", alpha=0.7)
        ax.hlines(p_at_p90, 0, r_at_p90, linestyles="dotted", alpha=0.7)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, ls=":", lw=0.8, alpha=0.6)
    ax.legend(loc="lower left", frameon=False)
    ax.set_title("Example Precision–Recall Curve")

    # 5) Save
    outdir = Path("output")
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "pr_curve_example.svg", bbox_inches="tight")
    plt.close(fig)

    print("Saved:", outdir / "pr_curve_example.svg")

if __name__ == "__main__":
    main()
