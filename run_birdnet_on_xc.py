from pathlib import Path

import pandas as pd
import tensorflow as tf
from birdnet.audio_based_prediction_mp import predict_species_within_audio_files_mp

# --- CONFIG ---
AUDIO_ROOT = "datasets/custom_set/xc_dataset/"
METADATA_CSV = "datasets/custom_set/metadata.csv"
MIN_CONF = 0.9

# Tensorflow GPUs
print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

xc_df = pd.read_csv(METADATA_CSV)


xc_df["path"] = AUDIO_ROOT + xc_df["path"]


recordings = [Path(path) for path in xc_df["path"]]

file_predictions = list(
    predict_species_within_audio_files_mp(
        recordings, min_confidence=MIN_CONF, chunk_overlap_s=1.5
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

merged = pd.merge(xc_df, predicts, on=["path"])

# Output the new dataframe
merged.to_csv("datasets/custom_set/train_metadata_birdnet.csv")
