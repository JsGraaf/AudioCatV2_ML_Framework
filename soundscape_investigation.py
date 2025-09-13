import argparse
import os
import pprint
from pathlib import Path
from typing import Dict, Sequence

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from pydub import AudioSegment
from pydub.playback import play
from scipy import signal

from audio_processing_notched import audio_pipeline, generate_mel_spectrogram
from metric_utils import get_scores_per_class
from misc import load_config
from models.binary_cnn import build_binary_cnn

pp = pprint.PrettyPrinter(indent=4)

parser = argparse.ArgumentParser()
parser.add_argument("input_csv")
args = parser.parse_args()

# Get the experiment dir from the input_csv
CONFIG_PATH = os.path.join(args.input_csv, "config.yaml")

# Load the config
config = load_config(CONFIG_PATH)

if config is None:
    exit(1)

config["ml"]["batch_size"] = 1
config["data"]["audio"]["sample_rate"] = 22050
config["data"]["audio"]["fmin"] = 1000
config["data"]["audio"]["fmax"] = 22050
config["data"]["augments"]["band_low_freq"] = 500
config["data"]["augments"]["band_high_freq"] = 6000
config["data"]["audio"]["n_mels"] = 80
config["data"]["augments"]["p_loud"] = 0.0
config["data"]["augments"]["p_gaus"] = 0.0
config["data"]["augments"]["p_spec"] = 0.0


if config["data"]["audio"]["center"]:
    config["data"]["audio"]["n_frames"] = (
        1
        + (config["data"]["audio"]["seconds"] * config["data"]["audio"]["sample_rate"])
        // config["data"]["audio"]["hop_length"]
    )
else:
    config["data"]["audio"]["n_frames"] = (
        1
        + (
            config["data"]["audio"]["seconds"] * config["data"]["audio"]["sample_rate"]
            - config["data"]["audio"]["n_fft"]
        )
        // config["data"]["audio"]["hop_length"]
    )


def build_val_dataset_notched(
    pos_files: Sequence[str], neg_files: Sequence[str], config: Dict
) -> tf.data.Dataset:

    labels_pos = tf.ones([len(pos_files)], dtype=tf.float32)
    labels_neg = tf.zeros([len(neg_files)], dtype=tf.float32)

    ds_pos = tf.data.Dataset.from_tensor_slices(
        (
            list(f[0] for f in pos_files),
            list(f[1] for f in pos_files),
            list(f[2] for f in pos_files),
            list(f[3] for f in pos_files),
            labels_pos,
        )
    ).map(
        lambda f, s, e, n, y: (audio_pipeline((f, s, e, n), config), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    ds_neg = tf.data.Dataset.from_tensor_slices(
        (
            list(f[0] for f in neg_files),
            list(f[1] for f in neg_files),
            list(f[2] for f in neg_files),
            list(f[3] for f in neg_files),
            labels_neg,
        )
    ).map(
        lambda f, s, e, n, y: (
            audio_pipeline((f, s, e, n), config),
            y,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    return ds_pos.concatenate(ds_neg).batch(config["ml"]["batch_size"])


def make_soundscape_dataset_notched(df: pd.DataFrame, config: Dict):
    all_pos = (
        (df[df["primary_label"] == config["exp"]["target"]])
        .apply(lambda x: (x["path"], x["start"], x["end"], x["freqs"]), axis=1)
        .tolist()
    )

    all_neg = (
        (df[df["primary_label"] != config["exp"]["target"]])
        .apply(lambda x: (x["path"], x["start"], x["end"], x["freqs"]), axis=1)
        .tolist()
    )

    soundscape_ds = build_val_dataset_notched(all_pos, all_neg, config)

    return soundscape_ds


def get_median_across_time(y, sr):
    # Get FFT
    n_fft = 4096
    hop = 512
    Y = librosa.stft(y, n_fft=n_fft, hop_length=hop, window="hann", center=True)
    S = np.abs(Y)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    profile = np.median(S, axis=1)

    # Limit to your band of interest and find peaks by prominence
    fmin, fmax = config["data"]["audio"]["fmin"], config["data"]["audio"]["fmax"]
    mask = (freqs >= fmin) & (freqs <= fmax)
    prom_thresh = np.percentile(profile[mask], 85)  # adjust 80–95 if needed
    peaks, props = signal.find_peaks(profile[mask], prominence=prom_thresh)

    cand_freqs = freqs[mask][peaks]
    cand_prom = props["prominences"]

    # 4) Report the strongest few lines
    cands = sorted(zip(cand_freqs, cand_prom), key=lambda x: -x[1])[:5]
    return cands

    # # 5) (Optional) visualize
    # plt.figure(figsize=(7, 4))
    # plt.semilogy(freqs, profile)  # median spectrum
    # for f, _ in cands:
    #     plt.axvline(f, linestyle="--")
    # plt.xlim(fmin, fmax)
    # plt.xlabel("Frequency (Hz)")
    # plt.ylabel("Median |STFT| (log)")
    # plt.title("Stationary-band detector")
    # plt.tight_layout()
    # plt.show()


def get_mic_freqs_from_soundscapes(df: pd.DataFrame):
    for soundscape_path in df["path"].unique():
        y, sr = librosa.load(soundscape_path, sr=config["data"]["audio"]["sample_rate"])

        # Extract any static noise bands that have to be removed
        freqs = get_median_across_time(y, sr)

        if len(freqs) <= 0:
            df.loc[df["path"] == soundscape_path, "freqs"] = [None] * (
                df["path"] == soundscape_path
            ).sum()
            continue

        df.loc[df["path"] == soundscape_path, "freqs"] = [freqs[0][0]] * (
            df["path"] == soundscape_path
        ).sum()
    df = df.fillna(0.0)
    return df


# Load the soundscape predcitions
df = pd.read_csv(os.path.join(args.input_csv, "soundscape_predictions.csv"))

df = get_mic_freqs_from_soundscapes(df)

# # Get the best thresholds
# df["true_label"] = df["primary_label"].apply(
#     lambda x: 1 if x == config["exp"]["target"] else 0
# )
# scores = get_scores_per_class(df)
#
# threshold = scores[1]["precision"]["t"]
#
# print("Using Threshold: ", threshold)
# df["predicted_label"] = df["predicted_label"].apply(lambda x: 1 if x > threshold else 0)
#
# mislabeled_target = df[
#     (df["primary_label"] == config["exp"]["target"])
#     & (df["predicted_label"] != df["true_label"])
# ]
# print(mislabeled_target)
#
#
# # Make sure the DF isn't empty
# if len(mislabeled_target) == 0:
#     raise ValueError("No mislabeled rows found.")
#
# paths = mislabeled_target["path"].astype(str).to_numpy()
# starts = mislabeled_target["start"].astype(np.float32).to_numpy()
# ends = mislabeled_target["end"].astype(np.float32).to_numpy()
# notches = mislabeled_target["freqs"].astype(np.float32).to_numpy()
#
# ds_pos = (
#     tf.data.Dataset.from_tensor_slices((paths, starts, ends, notches))
#     .map(
#         lambda f, s, e, n: (audio_pipeline((f, s, e, n), config), n),
#         num_parallel_calls=tf.data.AUTOTUNE,
#     )
#     .prefetch(tf.data.AUTOTUNE)
# )

# Create the TF dataset
soundscape_ds = make_soundscape_dataset_notched(df, config)

# for i, y in soundscape_ds.take(1):
#     print(i.numpy())
#     print(y.numpy())
#     librosa.display.specshow(
#         np.squeeze(i.numpy(), axis=0),
#         x_axis="time",
#         y_axis="mel",
#         fmin=config["data"]["audio"]["fmin"],
#         fmax=config["data"]["audio"]["fmax"],
#     )
#
# plt.show()

# Load the model
model = build_binary_cnn(
    input_shape=(
        config["data"]["audio"]["n_mels"],
        config["data"]["audio"]["n_frames"],
        1,
    )
)

# Load the weights
model.load_weights(os.path.join(args.input_csv, "full_training_model.keras"))

predictions = model.predict(soundscape_ds).ravel()

df["predictions"] = predictions

print(df["predictions"])


df["true_label"] = df["primary_label"].apply(
    lambda x: 1 if x == config["exp"]["target"] else 0
)

pp.pprint(get_scores_per_class(df))


# soundscapes_df.to_csv("soundscape_predictions.csv", index=False)


# for i, n in ds_pos.take(1):
#     print(n)
#     librosa.display.specshow(i.numpy())
# plt.show()
