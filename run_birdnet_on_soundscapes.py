import logging
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import tensorflow as tf
from birdnet.audio_based_prediction_mp import predict_species_within_audio_files_mp

# --- CONFIG ---
AUDIO_ROOT = "datasets/birdclef_2021/train_soundscapes"
META_CSV = "datasets/birdclef_2021/soundscape_metadata_birdnet.csv"
MIN_CONF = 0.7

# Tensorflow GPUs
print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))


def create_soundscapes_metadata(audio_root, path):
    """
    Creates a dataframe for the soundscapes adds the path for each file
    """
    if not os.path.isdir(audio_root):
        logging.error("Soundscape audio_root not valid!")

    df_rows = []
    for soundscape in os.listdir(audio_root):
        sp = soundscape.split("_")
        audio_id = sp[0]
        site = sp[1]
        date = sp[2]
        df_rows.append(
            {
                "audio_id": audio_id,
                "site": site,
                "date": date,
                "path": os.path.join(audio_root, soundscape),
            }
        )

    return pd.DataFrame(df_rows)


soundscape_df = create_soundscapes_metadata(AUDIO_ROOT, META_CSV)


def process_row(row):
    year = int(row["date"][:4])
    month = int(row["date"][4:6])
    day = int(row["date"][6:8])

    recording = {
        "path": Path(row["path"]),
        "date": datetime(year=year, month=month, day=day),
        "min_conf": MIN_CONF,
    }
    return recording


soundscapes = soundscape_df.apply(process_row, axis=1)

file_predictions = list(
    predict_species_within_audio_files_mp(
        [x["path"] for x in soundscapes], min_confidence=0.8
    )
)

predicts = []

for file, predictions in file_predictions:
    # Only keep the predictions that have a bird
    valid_preds = []
    for pred in list(predictions.items()):
        if len(pred[1].items()) > 0:
            valid_preds.append(pred)

        # For soundscapes we want to expand even the empty rows
        # valid_preds.append(pred)

    # Add valid preds to df
    predicts.append({"path": str(file), "predictions": valid_preds})

predicts = pd.DataFrame(predicts)


merged = pd.merge(soundscape_df, predicts, on=["path"])

merged.to_csv(META_CSV, index=False)
