# For every recording, run birdnet and verify the predictions.

import os
import glob
import argparse
import sys
from pathlib import Path
import tempfile
import librosa
import numpy as np

import re
import pandas as pd
from typing import Dict, Any, Optional


sys.path.append("../BirdNET-Analyzer")
sys.path.append("..")

from birdnet.audio_based_prediction_mp import predict_species_within_audio_files_mp
from metric_utils import plot_confusion_matrix_preprocessed

import numpy as np
import librosa
from load_predictions import get_predictions_df
import soundfile as sf
from tqdm import tqdm
from typing import Tuple
from multiprocessing import Pool
import pandas as pd


def rms(x: np.ndarray, eps: float = 1e-12) -> float:
    """RMS of a 1-D mono signal."""
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x * x) + eps))


def normalize_rms_mono(
    x: np.ndarray,
    target_rms: float = 0.125,
    max_gain_db: float = 24.0,
    eps: float = 1e-12,
    clip: bool = True,
) -> Tuple[np.ndarray, float]:
    """
    Normalize mono audio chunk to target RMS with max gain clamp.

    Returns:
      y (np.float32 array): normalized audio (same shape)
      applied_gain_db (float): gain actually applied in dB (<= max_gain_db)
    """
    x = np.asarray(x, dtype=np.float64)
    current_rms = rms(x, eps=eps)

    # If effectively silent, do nothing
    if current_rms <= eps:
        return x.astype(np.float32), 0.0

    desired_gain = target_rms / current_rms  # linear gain
    desired_gain_db = 20.0 * np.log10(desired_gain)  # convert to dB

    # clamp to max_gain_db
    applied_gain_db = float(min(desired_gain_db, max_gain_db))
    applied_gain = 10.0 ** (applied_gain_db / 20.0)

    y = x * applied_gain

    if clip:
        if np.max(np.abs(y)) > 1.0:
            np.clip(y, -1.0, 1.0, out=y)

    return y.astype(np.float32), applied_gain_db


def process_recording(args):
    rec, tmpdir = args
    tmp_path = Path(tmpdir) / f"{os.path.basename(rec)}"
    y, sr = librosa.load(rec)
    # y = normalize_rms_mono(y, target_rms=0.125, max_gain_db=12.0)[0]
    sf.write(tmp_path, y, sr)
    return str(tmp_path)


import pandas as pd


def metrics_from_df(df, pred_col="embedded_present", label_col="birdnet_present"):
    """
    Compute confusion counts, recall, precision, accuracy from DataFrame boolean columns.
    Returns (counts_dict, recall, precision, accuracy).
    """
    p = df[pred_col].astype(bool)
    L = df[label_col].astype(bool)

    tp = int(((p) & (L)).sum())
    tn = int(((~p) & (~L)).sum())
    fp = int(((p) & (~L)).sum())
    fn = int(((~p) & (L)).sum())

    total = tp + tn + fp + fn

    # safe division: if denominator 0, define metric as 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0

    counts = dict(tp=tp, tn=tn, fp=fp, fn=fn)
    return counts, recall, precision, accuracy


# regex to capture ('Label_name', 0.12345) pairs inside the string
_PAIR_RE = re.compile(r"\('([^']+)'\s*,\s*([0-9.eE+-]+)\)")


def check_species_row(
    row: pd.Series, *, species: str = "Pica pica", threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Inspect one dataframe row that looks like:
      filename,score,"OrderedDict([('Pica pica_Eurasian Magpie', 0.9308)])"
      or
      filename,score,[]

    Returns a dict with:
      - pred_prob: probability for the target species (None if not present)
      - predicted_present: bool, whether pred_prob >= threshold
      - label_present: bool, whether species appears in the OrderedDict
      - match: bool, whether predicted_present == label_present
      - top_label, top_prob: best-scoring label found (or None)
    Usage: df_results = df.apply(lambda r: pd.Series(check_species_row(r, threshold=0.5)), axis=1)
    """
    birdnet_preds = row["birdnet"]
    b_target_in_row = False
    b_target_prob = 0.0

    if birdnet_preds != []:
        # Check if the target species is present
        for key in birdnet_preds.keys():
            # Target found
            if species in key:
                b_target_in_row = True
                b_target_prob = birdnet_preds[key]

    # Determine if the model predicted this based on the threshold
    e_target_in_row = row["score"] >= threshold
    e_target_prob = row["score"]

    match = b_target_in_row == e_target_in_row

    return {
        "rec": row["rec"],
        "birdnet_present": b_target_in_row,
        "birdnet_prob": b_target_prob,
        "embedded_present": e_target_in_row,
        "embedded_prob": e_target_prob,
        "match": match,
        "birdnet": row["birdnet"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run BirdNET on recordings and verify predictions."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing audio recordings.",
    )
    args = parser.parse_args()

    recordings = sorted(glob.glob(args.input_dir + "/**/*.wav", recursive=True))

    birdnet_predictions = []

    with tempfile.TemporaryDirectory() as tmpdir:
        args_list = [(rec, tmpdir) for rec in recordings]
        with Pool() as pool:
            recordings = list(
                tqdm(
                    pool.imap(process_recording, args_list),
                    total=len(recordings),
                    desc="Processing recordings",
                )
            )

        birdnet_predictions = list(
            predict_species_within_audio_files_mp(
                [Path(x) for x in recordings], min_confidence=0.7
            )
        )
        birdnet_predictions = {
            os.path.basename(pred[0]): pred[1] for pred in birdnet_predictions
        }

    # Load the model predictions
    embedded_preds = pd.DataFrame()
    for hour_predictions in glob.glob(
        args.input_dir + "/**/pred_*.txt", recursive=True
    ):
        embedded_preds = pd.concat(
            [embedded_preds, get_predictions_df(hour_predictions)]
        )

    def get_birdnet_prediction(row):
        if row["rec"] in birdnet_predictions:
            pred = list(birdnet_predictions[row["rec"]].items())[0][1]
            if len(pred) > 0:
                return pred
        return []

    embedded_preds["birdnet"] = embedded_preds.apply(get_birdnet_prediction, axis=1)

    embedded_preds.to_csv("output/verification.csv", index=False)

    # Check the predictions missed by embedded platform
    results = embedded_preds.apply(
        lambda r: pd.Series(check_species_row(r, species="Pica pica", threshold=0.5)),
        axis=1,
    )

    detections = results[results["birdnet"].apply(lambda x: x != [])]
    detections.to_csv("output/detections.csv")
    print(f"Total detections: {detections.shape[0]}")

    # using & with parentheses
    df_true_pos = results[
        (results["birdnet_present"] == True) & (results["embedded_present"] == True)
    ].copy()
    df_true_neg = results[
        (results["birdnet_present"] == False) & (results["embedded_present"] == False)
    ].copy()

    print(f"True Positives: {df_true_pos.shape[0]}")
    print(f"True Negatives: {df_true_neg.shape[0]}")

    df_false_pos = results[
        (results["match"] == False) & (results["embedded_present"] == True)
    ].copy()
    df_false_neg = results[
        (results["match"] == False) & (results["embedded_present"] == False)
    ].copy()

    # Group the false positives based on the birdnet prediction
    print("False Positives")
    df_false_pos["birdnet"] = df_false_pos["birdnet"].apply(
        lambda x: list(x.keys())[0] if x != [] else "None"
    )
    print(df_false_pos)
    print(df_false_pos["birdnet"].value_counts())

    print("False Negatives")
    df_false_neg["birdnet"] = df_false_neg["birdnet"].apply(
        lambda x: list(x.keys())[0] if x != [] else "None"
    )
    print(df_false_neg)
    print(df_false_neg["birdnet"].value_counts())

    results.to_csv("output/verification_results.csv", index=False)

    counts, recall, precision, accuracy = metrics_from_df(results)
    print(f"recall={recall:.4f}, precision={precision:.4f}, accuracy={accuracy:.4f}")
    # Convert counts dict to confusion matrix array
    confusion_matrix = np.array(
        [[counts["tp"], counts["fp"]], [counts["fn"], counts["tn"]]]
    )

    fig, _ = plot_confusion_matrix_preprocessed(
        confusion_matrix,
        labels=["Pica pica", "Other"],
        title="Confusion Matrix for Pica pica",
    )
    fig.savefig(os.path.join("output", os.path.basename(args.input_dir) + ".svg"))

    spec_root = Path(args.input_dir, "Spectrograms")
    stamp_re = re.compile(r"(Y\d{4}M\d{2}D\d{2}_\d{2}h\d{2}m\d{2}s\d{3})", re.I)

    def find_spec(rec_name: str):
        m = stamp_re.search(rec_name)
        if not m:
            return None
        stamp = m.group(1)
        # pick the first match like spec_*<stamp>*.txt anywhere under Spectrograms
        matches = sorted(spec_root.rglob(f"*{stamp}*.txt"))
        return str(matches[0]) if matches else None

    df_false_neg["spec_path"] = df_false_neg["rec"].apply(find_spec)

    from compare_spectrograms import read_device_spec_matrix, _axes_extent
    import matplotlib.pyplot as plt

    extent = _axes_extent(283, 340, 32000, 400, 14999)

    for s in df_false_neg["spec_path"]:
        mat = read_device_spec_matrix(
            s,
            num_mels=80,
            num_frames=283,
            layout="mel-major",
            flip_frequency=False,
        )
        fig, ax = plt.subplots(figsize=(11, 9))
        im = librosa.display.specshow(            
            mat,
            x_axis="time",
            y_axis="mel",
            fmin=400,
            fmax=14999,
            cmap="gray_r",
            vmin=0,
            vmax=1,)
        
        ax.set_title(f"{os.path.basename(s)}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")

        cbar = fig.colorbar(im)
        cbar.set_label("Normalized dB (0..1)")
        cbar.ax.tick_params(labelsize=8)

        ax.set_xlim((0.0, 3.0))

        fig.savefig(
            os.path.join("output", os.path.basename(s).replace("txt", "svg")),
            bbox_inches="tight",
        )


if __name__ == "__main__":
    main()
