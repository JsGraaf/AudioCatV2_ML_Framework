import logging
from pathlib import Path
from typing import Optional

import keras_tuner as kt
import numpy as np
import yaml


def load_config(path: Path = Path("config.yaml")) -> Optional[dict]:
    """
    Loads the yaml config
    """
    with open(path, "r", encoding="utf-8") as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as exc:
            logging.error(f"Failed to load config: {exc}")
            return None


def hp_audio_and_aug(hp: kt.HyperParameters):

    # Load the normal config
    CONFIG_PATH = "config.yaml"
    logging.info(f"Loading config from {CONFIG_PATH}")
    cfg = load_config(CONFIG_PATH)

    if cfg is None:
        exit(1)
    # ----- Audio front-end (mel/STFT & framing) -----
    cfg["data"]["audio"]["sample_rate"] = hp.Choice(
        "sample_rate",
        [16000, 22050, 32000],
        default=16000,
    )

    # # Time window length used upstream (e.g., crop/pad seconds)
    # cfg["data"]["audio"]["seconds"] = hp.Choice("seconds", [2.0, 3.0, 5.0], default=3.0)

    # Spectrogram geometry
    cfg["data"]["audio"]["n_mels"] = hp.Choice("n_mels", [64, 80, 96, 128], default=128)
    # cfg["n_frames"] = hp.Choice("n_frames", [64, 94, 128], default=94)

    # STFT params
    cfg["data"]["audio"]["n_fft"] = hp.Choice("n_fft", [512, 1024], default=1024)
    cfg["data"]["audio"]["hop_length"] = hp.Int(
        "hop_length", min_value=128, max_value=512, step=16
    )

    # Let win_length follow n_fft or be slightly shorter
    # cfg["data"]["audio"]["win_length"] = hp.Choice("win_length", [512, 768, 1024])

    # Window function (map your "hamm" to "hamming")
    cfg["data"]["audio"]["window"] = hp.Choice(
        "window", ["hann", "hamming"], default="hamming"
    )
    cfg["data"]["audio"]["center"] = hp.Boolean("center", default=True)

    # Mel scaling / band limits
    cfg["data"]["audio"]["fmin"] = hp.Choice("fmin", [100, 150, 200, 300], default=200)
    cfg["data"]["audio"]["fmax"] = hp.Choice("fmax", [6000, 7500, 7999], default=7999)
    # cfg["data"]["audio"]["norm"] = hp.Choice(
    #     "mel_norm", [np.inf, "slaney"], default="slaney"
    # )

    # Power=2.0 (power-mel) is your default; optionally tune a tiny range
    cfg["data"]["audio"]["power"] = hp.Choice("power", [1.0, 2.0], default=2.0)

    # Optional frontend prefilter
    cfg["data"]["audio"]["butterworth_order"] = hp.Choice(
        "butterworth_order", [2, 4, 6, 8], default=4
    )

    # ----- Augmentations -----
    # Probability to apply loudness gain jitter
    cfg["data"]["augments"]["p_loud"] = hp.Float(
        "p_loud", min_value=0.0, max_value=0.8, default=0.0
    )

    # Loudness range in dB (min,max)
    cfg["data"]["augments"]["loud_min_db"] = hp.Float(
        "loud_min_db", -30.0, -12.0, step=2.0, default=-26.0
    )
    cfg["data"]["augments"]["loud_max_db"] = hp.Float(
        "loud_max_db", -24.0, -6.0, step=2.0, default=-18.0
    )

    # Gaussian noise
    cfg["data"]["augments"]["p_gaus"] = hp.Float(
        "p_gaus", min_value=0.0, max_value=0.8, default=0.0
    )
    cfg["data"]["augments"]["gaus_snr_min"] = hp.Int(
        "gaus_snr_min", 5, 20, step=5, default=10
    )
    cfg["data"]["augments"]["gaus_snr_max"] = hp.Int(
        "gaus_snr_max", 15, 35, step=5, default=20
    )

    # SpecAugment (masking only, since you already use it)
    cfg["data"]["augments"]["p_spec"] = hp.Float(
        "p_spec", min_value=0.01, max_value=0.8, default=0.5
    )
    cfg["data"]["augments"]["freq_masks"] = hp.Int("freq_masks", 0, 2, default=1)
    cfg["data"]["augments"]["time_masks"] = hp.Int("time_masks", 0, 2, default=1)
    cfg["data"]["augments"]["freq_mask_prop"] = hp.Float(
        "freq_mask_prop", 0.05, 0.2, step=0.05, default=0.1
    )
    cfg["data"]["augments"]["time_mask_prop"] = hp.Float(
        "time_mask_prop", 0.05, 0.2, step=0.05, default=0.1
    )

    # ----- Sampling / batching -----
    cfg["ml"]["batch_size"] = hp.Choice("batch_size", [16, 32, 64], default=32)

    # Class ratio in sampler (if you use it to build tf.data)
    cfg["ml"]["pos_neg_ratio"] = hp.Choice("pos_neg_ratio", [1, 2, 3, 4], default=3)

    # Training schedule (Hyperband will early-stop anyway)
    cfg["ml"]["epochs_per_fold"] = hp.Int(
        "epochs_per_fold", 30, 70, step=10, default=50
    )

    # Calculate the amount of frames
    if cfg["data"]["audio"]["center"]:
        cfg["data"]["audio"]["n_frames"] = 1 + (
            cfg["data"]["audio"]["seconds"] * hp.get("sample_rate")
        ) // hp.get("hop_length")
    else:
        cfg["data"]["audio"]["n_frames"] = 1 + (
            (cfg["data"]["audio"]["seconds"] * hp.get("sample_rate")) - hp.get("n_fft")
        ) // hp.get("hop_length")
    return cfg
