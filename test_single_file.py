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
from birdnet.audio_based_prediction import predict_species_within_audio_file
from pydub import AudioSegment
from pydub.playback import play
from scipy import signal

from audio_processing import generate_mel_spectrogram
from metric_utils import get_scores_per_class
from misc import load_config
from models.binary_cnn import build_binary_cnn
from models.miniresnet import build_miniresnet

pp = pprint.PrettyPrinter(indent=4)


def split_audio(y, sr, chunk_sec):
    chunk_len = int(chunk_sec * sr)
    n_chunks = int(np.ceil(len(y) / chunk_len))
    chunks = []
    for i in range(n_chunks):
        start = i * chunk_len
        end = min((i + 1) * chunk_len, len(y))
        y_chunk = y[start:end]
        # Pad last chunk to full length
        if len(y_chunk) < chunk_len:
            y_chunk = np.pad(y_chunk, (0, chunk_len - len(y_chunk)))
        chunks.append((y_chunk, start / sr, end / sr))
    return chunks


pp = pprint.PrettyPrinter(indent=4)

parser = argparse.ArgumentParser()
parser.add_argument("input_file")
args = parser.parse_args()

# Get the experiment dir from the input_csv
CONFIG_PATH = os.path.join("config.yaml")

# Load the config
config = load_config(CONFIG_PATH)

if config is None:
    exit(1)

audio_cfg: Dict = config["data"]["audio"]
win_sec = float(audio_cfg["seconds"])
sr = int(audio_cfg["sample_rate"])
n_fft = int(audio_cfg["n_fft"])
hop_length = int(audio_cfg["hop_length"])
n_mels = int(audio_cfg["n_mels"])
fmin = int(audio_cfg["fmin"])
fmax = int(audio_cfg["fmax"])
use_pcen = bool(audio_cfg.get("use_pcen", False))

# optional PCEN params
use_pcen = audio_cfg["use_pcen"]
pcen_time_constant = float(audio_cfg["pcen_time_constant"])
pcen_gain = float(audio_cfg["pcen_gain"])
pcen_bias = float(audio_cfg["pcen_bias"])
pcen_power = audio_cfg["pcen_power"]
pcen_eps = audio_cfg["pcen_eps"]


# Load the model
model = build_miniresnet(
    input_shape=(
        config["data"]["audio"]["n_mels"],
        config["data"]["audio"]["n_frames"],
        1,
    ),
    n_classes=1,
)

# Load the weights
model.load_weights(config["ml"]["model_path"])

# Load the audio
y, sr = librosa.load(args.input_file, sr=config["data"]["audio"]["sample_rate"])

# librosa.display.waveshow(y=y, sr=sr)
# plt.show()

# Predict with birdnet
birdnet_results = list(predict_species_within_audio_file(Path(args.input_file)))

# Split into chunks
chunks = split_audio(y, sr, config["data"]["audio"]["seconds"])

audio_config = config["data"]["audio"]

# Run model on each chunk
results = []
for i, (y_chunk, t0, t1) in enumerate(chunks):
    # Features (log-mel or PCEN via your function) -> [0,1], shape [n_mels, frames]
    spec = generate_mel_spectrogram(
        audio=y_chunk,
        sr=sr,
        n_fft=n_fft,
        n_mels=n_mels,
        hop_length=hop_length,
        win_length=None,
        window="hann",
        center=True,
        pad_mode="reflect",
        power=2.0,
        fmin=fmin,
        fmax=fmax,
        norm=None,
        use_pcen=use_pcen,
        pcen_time_constant=pcen_time_constant,
        pcen_gain=pcen_gain,
        pcen_bias=pcen_bias,
        pcen_power=pcen_power,
        pcen_eps=pcen_eps,
    )

    librosa.display.specshow(
        spec,
        x_axis="time",
        y_axis="mel",
        sr=config["data"]["audio"]["sample_rate"],
        hop_length=config["data"]["audio"]["hop_length"],
        fmin=config["data"]["audio"]["fmin"],
        fmax=config["data"]["audio"]["fmax"],
        cmap="magma",
        vmin=0,
        vmax=1,
    )
    # plt.show()

    H_expected = 80
    W_expected = 241

    if spec.ndim != 2:
        raise ValueError(f"Expected 2-D [n_mels, n_frames], got {S.shape}")

    H, W = spec.shape

    # If it's time-major (frames, mels), transpose to (mels, frames)
    if H == W_expected and W == H_expected:
        spec = spec.T
        H, W = spec.shape

    # Enforce mel count
    if H != H_expected:
        raise ValueError(
            f"n_mels mismatch: got {H}, expected {H_expected}. "
            "Compute the mel spec with n_mels set to the model's value."
        )

    # Pad/crop time axis to expected frames
    if W < W_expected:
        pad = np.zeros((H, W_expected - W), dtype=S.dtype)
        spec = np.concatenate([spec, pad], axis=1)
    elif W > W_expected:
        off = (W - W_expected) // 2
        S = spec[:, off : off + W_expected]

    # Add channel and batch dims: (H,W)->(H,W,1)->(1,H,W,1)
    x = spec.astype(np.float32)[..., None][None, ...]

    # Predict
    preds = model.predict(x, verbose=1)
    results.append((t0, t1, preds, birdnet_results[i]))

pp.pprint(results)

# # Print results
# for t0, t1, preds in results:
#     # Also play it
#
#     y, sr = librosa.load(args.input_file, offset=t0, duration=t1 - t0, sr=sr)
#
#     # Convert librosa waveform back to AudioSegment
#     audio = AudioSegment(
#         (y * 32767).astype(np.int16).tobytes(),
#         frame_rate=sr,
#         sample_width=2,
#         channels=1,
#     )
#
#     play(audio)
#
#     marked = input("Is this a bird (y/n)")
#     print(f"{t0:6.2f}-{t1:6.2f} sec : {preds}: Marked = {marked}")
