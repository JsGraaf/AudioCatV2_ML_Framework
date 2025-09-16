from datetime import datetime
from pathlib import Path

import pandas as pd
import tensorflow as tf
from birdnet.audio_based_prediction_mp import predict_species_within_audio_files_mp

# --- CONFIG ---
AUDIO_ROOT = "datasets/birdclef_2021/train_short_audio"
META_CSV = "datasets/birdclef_2021/train_metadata.csv"
MIN_CONF = 0.9

# Tensorflow GPUs
print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))


# ---------- helpers ----------
def load_birdclef(audio_root, path):
    df = pd.read_csv(path)
    df["path"] = audio_root + "/" + df["primary_label"] + "/" + df["filename"]
    return df


birdclef_df = load_birdclef(
    "datasets/birdclef_2021/train_short_audio",
    "datasets/birdclef_2021/train_metadata.csv",
)


def process_row(row):
    # Gather all the info for a row
    sp = row["date"].split("-")

    year = max(int(sp[0]), 2001)
    month = max(int(sp[1]), 1)
    day = max(int(sp[2]), 1)

    recording = {
        "path": Path(row["path"]),
        "lat": row["latitude"],
        "lon": row["longitude"],
        "date": datetime(year=year, month=month, day=day),
        "min_conf": MIN_CONF,
    }
    return (recording, row["common_name"])


recordings = birdclef_df.apply(process_row, axis=1)

recordings_temp = [r[0]["path"] for r in recordings]

file_predictions = list(
    predict_species_within_audio_files_mp(
        recordings_temp, min_confidence=MIN_CONF, chunk_overlap_s=1.5
    )
)

predicts = []

for file, predictions in file_predictions:
    # Only keep the predictions that have a bird
    valid_preds = []
    for pred in list(predictions.items()):
        if len(pred[1].items()) > 0:
            valid_preds.append(pred)

    # Add valid preds to df
    predicts.append({"path": str(file), "predictions": valid_preds})

predicts = pd.DataFrame(predicts)


merged = pd.merge(birdclef_df, predicts, on=["path"])

# Output the new dataframe
merged.to_csv("datasets/birdclef_2021/train_metadata_birdnet.csv")
