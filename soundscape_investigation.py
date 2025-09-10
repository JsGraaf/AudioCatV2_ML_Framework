import argparse
import os

import pandas as pd
import yaml

from audio_processing import generate_mel_spectrogram
from metric_utils import get_scores_per_class
from misc import load_config

parser = argparse.ArgumentParser()
parser.add_argument("input_csv")
args = parser.parse_args()

# Load the config
CONFIG_PATH = "config.yaml"
config = load_config(CONFIG_PATH)

if config is None:
    exit(1)

# Load the soundscape predcitions
df = pd.read_csv(os.path.join(args.input_csv, "soundscape_predictions.csv"))

# Get the best thresholds
df["true_label"] = df["primary_label"].apply(
    lambda x: 1 if x == config["exp"]["target"] else 0
)
scores = get_scores_per_class(df, min_precision=0)

print(scores)
