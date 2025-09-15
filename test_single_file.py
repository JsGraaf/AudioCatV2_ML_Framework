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

from audio_processing_notched import audio_pipeline, generate_mel_spectrogram
from metric_utils import get_scores_per_class
from misc import load_config
from models.binary_cnn import build_binary_cnn


# ---------- feature extraction ----------
def extract_features(y, sr, n_fft=1024, hop=256, n_mels=96, fmin=200, fmax=7500):
    """Compute log-Mel spectrogram with power=2 (your thesis convention)."""
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0,
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    # Normalize if needed (per training setup)
    return S_db.astype(np.float32)


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

# Load the model
model = build_binary_cnn(
    input_shape=(
        config["data"]["audio"]["n_mels"],
        config["data"]["audio"]["n_frames"],
        1,
    )
)
# Load the weights
model.load_weights(config["ml"]["model_path"])

# Load the audio
y, sr = librosa.load(args.input_file, sr=config["data"]["audio"]["sample_rate"])

librosa.display.waveshow(y=y, sr=sr)
plt.show()

# Predict with birdnet
print(predict_species_within_audio_file(Path(args.input_file)))

# Split into chunks
chunks = split_audio(y, sr, config["data"]["audio"]["seconds"])

audio_config = config["data"]["audio"]

# Run model on each chunk
results = []
for i, (y_chunk, t0, t1) in enumerate(chunks):
    # Extract features
    S = extract_features(
        y_chunk,
        sr,
        audio_config["n_fft"],
        audio_config["hop_length"],
        audio_config["n_mels"],
        audio_config["fmin"],
        audio_config["fmax"],
    )
    # Model expects batch, possibly channels-last (H, W, C)
    if len(S.shape) == 2:
        S_in = np.expand_dims(S, axis=-1)  # (mel, time, 1)
    else:
        S_in = S
    S_in = np.expand_dims(S_in, axis=0)  # batch dim

    # Predict
    preds = model.predict(S_in, verbose=0)[0]  # first row
    results.append((t0, t1, preds))

# Print results
for t0, t1, preds in results:
    # Also play it

    y, sr = librosa.load(args.input_file, offset=t0, duration=t1 - t0, sr=sr)

    # Convert librosa waveform back to AudioSegment
    audio = AudioSegment(
        (y * 32767).astype(np.int16).tobytes(),
        frame_rate=sr,
        sample_width=2,
        channels=1,
    )

    play(audio)

    marked = input("Is this a bird (y/n)")
    print(f"{t0:6.2f}-{t1:6.2f} sec : {preds}: Marked = {marked}")


print(y)
