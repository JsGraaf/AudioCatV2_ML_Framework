#!/usr/bin/env python3
"""
Bar plots for precision / recall / recall_at_p90 from an Excel file.

- X axis: variant names (categorical) as bars
- Output: SVGs with tight layout in ./output/
- Also writes one <stem>_plots.tex that \includesvg's all three figures.

Assumes your LaTeX preamble already loads whatever you use to include SVGs.
"""

import argparse
import pathlib
import sys
from typing import Dict, List

import pandas as pd
import matplotlib.pyplot as plt
import ast


# ---------- Column mapping helpers ----------
CANON = {
    "variant": ["variant", "name", "model", "config"],
    "precision": ["precision", "prec", "p"],
    "recall": ["recall", "r"],
    "recall_at_p90": ["recall_at_p90", "rp90", "recall@p90", "recall_at_90p", "recall_at90p"],
}

def _norm_key(s: str) -> str:
    return s.lower().replace(" ", "").replace("-", "_")

def find_col(df: pd.DataFrame, keys: List[str]) -> str:
    lower = {_norm_key(c): c for c in df.columns}
    for k in keys:
        k2 = _norm_key(k)
        if k2 in lower:
            return lower[k2]
    # fallback: substring search
    for k in keys:
        k2 = _norm_key(k)
        for c in df.columns:
            if k2 in _norm_key(c):
                return c
    raise KeyError(f"Could not find any of {keys} in columns: {list(df.columns)}")

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Assume: first col = variant, second col = dict-like metrics
    metric_col = df.columns[1]
    df = df.copy()
    df["Variant"] = df[df.columns[0]]

    def get_metrics(row):
        d = ast.literal_eval(row[metric_col]) if isinstance(row[metric_col], str) else row[metric_col]
        for k in ("precision", "recall", "recall_at_p90"):
            row[k] = d.get(k, float("nan"))
            if "Redone" in row and pd.notna(row["Redone"]):
                subd = ast.literal_eval(row["Redone"])
                row[k] += subd[k]
                row[k] = row[k] / 2
        return row

    df = df.apply(get_metrics, axis=1)
    return df[["Variant", "precision", "recall", "recall_at_p90"]]

# ---------- Plot helpers ----------
def assign_colors(variants: List[str]) -> Dict[str, str]:
    # Deterministic color per variant across all figures
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not cycle:
        cycle = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
                 "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]
    return {v: cycle[i % len(cycle)] for i, v in enumerate(variants)}

def bar_plot(df: pd.DataFrame, metric: str, colors: Dict[str, str], out_path: pathlib.Path,
             title: str, ylabel: str):
    x = list(range(len(df)))
    xticks = df["Variant"].tolist()
    vals = df[metric].values
    cs = [colors[v] for v in df["Variant"]]

    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    bars = ax.bar(x, vals, color=cs, edgecolor="none")
    ax.set_xlim(-0.5, len(df) - 0.5)
    ax.set_ylim(0.98, 1.0)  # metrics are probabilities
    ax.set_xticks(x)
    ax.set_xticklabels(xticks, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.8, alpha=0.6)

    # Label each bar halfway up the visible portion (between y-min and bar height)
    y_min = ax.get_ylim()[0]
    for rect, v in zip(bars, vals):
        x_center = rect.get_x() + rect.get_width() / 2
        y_mid = y_min + 0.5 * (v - y_min)
        ax.text(x_center, y_mid, f"{v:.4f}",
                ha="center", va="center",
                fontsize=9, color="white",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="none", edgecolor="none"))

    fig.tight_layout()
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Bar plots for precision/recall/rp90 from an Excel sheet.")
    parser.add_argument("xlsx", nargs="?", default="input/results.xlsx",
                        help="Path to .xlsx (default: input/results.xlsx)")
    parser.add_argument("--sheet", default=0, help="Worksheet index/name (default: 0)")
    parser.add_argument("--outdir", default="output", help="Output directory (default: output)")
    args = parser.parse_args()

    xlsx = pathlib.Path(args.xlsx)
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(xlsx, sheet_name=args.sheet)
    df = normalize_columns(df)

    # Consistent colors per variant across all figures
    variants = df["Variant"].tolist()
    color_map = assign_colors(variants)

    stem = xlsx.stem
    prec_svg = outdir / f"{stem}_precision_bar.svg"
    rec_svg  = outdir / f"{stem}_recall_bar.svg"
    rp90_svg = outdir / f"{stem}_rp90_bar.svg"

    bar_plot(df, "precision",      color_map, prec_svg, "Precision per Variant", "Precision")
    bar_plot(df, "recall",         color_map, rec_svg,  "Recall per Variant",    "Recall")
    bar_plot(df, "recall_at_p90",  color_map, rp90_svg, "Recall at 90% Precision", "Recall@P=0.90")

    # Emit a single .tex file that brings all figures in.
    tex_path = outdir / f"{stem}_plots.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% Auto-generated: include the three SVG figures.\n")
        f.write("\\begin{figure}[t]\n\\centering\n")
        f.write(f"\\includesvg[width=\\linewidth]{{{prec_svg.name}}}\n")
        f.write("\\caption{Precision per variant.}\\label{fig:" + stem + "-precision}\n")
        f.write("\\end{figure}\n\n")

        f.write("\\begin{figure}[t]\n\\centering\n")
        f.write(f"\\includesvg[width=\\linewidth]{{{rec_svg.name}}}\n")
        f.write("\\caption{Recall per variant.}\\label{fig:" + stem + "-recall}\n")
        f.write("\\end{figure}\n\n")

        f.write("\\begin{figure}[t]\n\\centering\n")
        f.write(f"\\includesvg[width=\\linewidth]{{{rp90_svg.name}}}\n")
        f.write("\\caption{Recall at 90\\% precision per variant.}\\label{fig:" + stem + "-rp90}\n")
        f.write("\\end{figure}\n")
    print(f"Wrote:\n  {prec_svg}\n  {rec_svg}\n  {rp90_svg}\n  {tex_path}")

if __name__ == "__main__":
    main()
