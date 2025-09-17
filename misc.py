import logging
from pathlib import Path
from typing import Optional

import keras_tuner as kt
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
    # Load base config
    CONFIG_PATH = "config.yaml"
    logging.info(f"Loading config from {CONFIG_PATH}")
    cfg = load_config(CONFIG_PATH)
    if cfg is None:
        raise SystemExit(1)

    # ----- Audio front-end (mel/STFT & framing) -----
    cfg["data"]["audio"]["sample_rate"] = hp.Choice(
        "sample_rate", [16000, 22050, 32000], default=16000
    )
    sr = cfg["data"]["audio"]["sample_rate"]

    # Spectrogram geometry
    cfg["data"]["audio"]["n_mels"] = hp.Choice("n_mels", [64, 80, 96, 128], default=128)

    # STFT params
    cfg["data"]["audio"]["n_fft"] = hp.Choice("n_fft", [512, 1024], default=1024)
    nfft = cfg["data"]["audio"]["n_fft"]

    # Hop depends on n_fft (keep reasonable geometries)
    if nfft == 512:
        cfg["data"]["audio"]["hop_length"] = hp.Int(
            "hop_length", 96, 320, step=16, default=192
        )
    else:  # 1024
        cfg["data"]["audio"]["hop_length"] = hp.Int(
            "hop_length", 128, 512, step=16, default=256
        )
    hop = cfg["data"]["audio"]["hop_length"]

    # Windowing / centering
    cfg["data"]["audio"]["window"] = hp.Choice(
        "window", ["hann", "hamming"], default="hamming"
    )
    cfg["data"]["audio"]["center"] = hp.Boolean("center", default=True)

    # Mel band limits – fmax must be ≤ Nyquist
    cfg["data"]["audio"]["fmin"] = hp.Choice("fmin", [100, 150, 200, 300], default=200)

    if sr == 16000:
        cfg["data"]["audio"]["fmax"] = hp.Choice("fmax", [6000, 7500], default=7500)
        cfg["data"]["audio"]["band_low_freq"] = hp.Choice(
            "band_low_freq", [500, 1000], default=1000
        )
        cfg["data"]["audio"]["band_high_freq"] = hp.Choice(
            "band_high_freq", [6000, 7500, 7999], default=7500
        )
    elif sr == 22050:
        cfg["data"]["audio"]["fmax"] = hp.Choice(
            "fmax", [7500, 9000, 11025], default=9000
        )
        cfg["data"]["audio"]["band_low_freq"] = hp.Choice(
            "band_low_freq", [200, 500, 1000], default=1000
        )
        cfg["data"]["audio"]["band_high_freq"] = hp.Choice(
            "band_high_freq", [9000, 11000], default=11000
        )
    else:  # 32000
        cfg["data"]["audio"]["fmax"] = hp.Choice(
            "fmax", [9000, 12000, 15000, 15999], default=12000
        )
        cfg["data"]["audio"]["band_low_freq"] = hp.Choice(
            "band_low_freq", [200, 500, 1000], default=1000
        )
        cfg["data"]["audio"]["band_high_freq"] = hp.Choice(
            "band_high_freq", [10000, 12000, 15000, 15999], default=15000
        )

    # Power-mel choice
    cfg["data"]["audio"]["power"] = hp.Choice("power", [1.0, 2.0], default=2.0)

    # Optional Butterworth bandpass (0 = disabled)
    cfg["data"]["audio"]["butterworth_order"] = hp.Choice(
        "butterworth_order", [0, 2, 4, 6, 8], default=4
    )
    if cfg["data"]["audio"]["butterworth_order"] > 0:
        cfg["data"]["augments"]["butter_fmin"] = hp.Choice(
            "butter_fmin", [100, 150, 200, 300], default=200
        )
        if sr == 16000:
            cfg["data"]["augments"]["band_high_freq"] = hp.Choice(
                "band_high_freq", [6000, 7000, 7500], default=7000
            )
        elif sr == 22050:
            cfg["data"]["augments"]["band_high_freq"] = hp.Choice(
                "band_high_freq", [8000, 10000, 11000], default=10000
            )
        else:  # 32000
            cfg["data"]["augments"]["band_high_freq"] = hp.Choice(
                "band_high_freq", [8000, 10000, 11000, 15000], default=10000
            )

    # ----- Augmentations (define knobs only if p>0) -----
    # Loudness jitter
    p_loud = hp.Choice("p_loud", [0.0, 0.25, 0.5, 0.75], default=0.0)
    cfg["data"]["augments"]["p_loud"] = p_loud
    if p_loud > 0.0:
        cfg["data"]["augments"]["loud_min_db"] = hp.Float(
            "loud_min_db", min_value=-36.0, max_value=-6.0, step=2.0, default=-26.0
        )
        cfg["data"]["augments"]["loud_max_db"] = hp.Float(
            "loud_max_db", min_value=-24.0, max_value=0.0, step=2.0, default=-18.0
        )

    # Gaussian noise (SNR in dB)
    p_gaus = hp.Choice("p_gaus", [0.0, 0.25, 0.5, 0.75], default=0.0)
    cfg["data"]["augments"]["p_gaus"] = p_gaus
    if p_gaus > 0.0:
        cfg["data"]["augments"]["gaus_snr_min"] = hp.Int(
            "gaus_snr_min", 0, 20, step=5, default=10
        )
        cfg["data"]["augments"]["gaus_snr_max"] = hp.Int(
            "gaus_snr_max", 15, 35, step=5, default=20
        )

    # SpecAugment (masking only)
    p_spec = hp.Choice("p_spec", [0.0, 0.25, 0.5, 0.75], default=0.5)
    cfg["data"]["augments"]["p_spec"] = p_spec
    if p_spec > 0.0:
        cfg["data"]["augments"]["freq_masks"] = hp.Int("freq_masks", 0, 2, default=1)
        cfg["data"]["augments"]["time_masks"] = hp.Int("time_masks", 0, 2, default=1)
        cfg["data"]["augments"]["spec_max_freq_width"] = hp.Int(
            "freq_width", min_value=2, max_value=64, step=8, default=24
        )
        cfg["data"]["augments"]["spec_max_time_width"] = hp.Int(
            "time_width", min_value=2, max_value=64, step=8, default=24
        )

    # ----- Sampling / batching -----
    cfg["ml"]["batch_size"] = hp.Choice("batch_size", [16, 32, 64], default=32)
    cfg["ml"]["pos_neg_ratio"] = hp.Choice("pos_neg_ratio", [1, 2, 3, 4], default=3)
    cfg["ml"]["epochs_per_fold"] = hp.Int(
        "epochs_per_fold", 30, 100, step=10, default=50
    )

    # ---- PCEN ----
    cfg["data"]["audio"]["use_pcen"] = hp.Boolean("use_pcen", default=True)
    if cfg["data"]["audio"]["use_pcen"]:
        cfg["data"]["audio"]["pcen_time_constant"] = hp.Float(
            "pcen_time_constant",
            min_value=0.20,
            max_value=0.80,
            step=0.05,
            default=0.40,
        )
        cfg["data"]["audio"]["pcen_gain"] = hp.Float(
            "pcen_gain", min_value=0.90, max_value=0.995, step=0.005, default=0.98
        )
        cfg["data"]["audio"]["pcen_bias"] = hp.Float(
            "pcen_bias", min_value=0.5, max_value=4.0, step=0.25, default=2.0
        )
        cfg["data"]["audio"]["pcen_power"] = hp.Float(
            "pcen_power", min_value=0.30, max_value=0.70, step=0.05, default=0.50
        )
        cfg["data"]["audio"]["pcen_eps"] = 1e-6
        cfg["data"]["audio"]["pcen_log1p"] = hp.Boolean("pcen_log1p", default=False)
    else:
        # sensible defaults so downstream code has the keys
        cfg["data"]["audio"].setdefault("pcen_time_constant", 0.40)
        cfg["data"]["audio"].setdefault("pcen_gain", 0.98)
        cfg["data"]["audio"].setdefault("pcen_bias", 2.0)
        cfg["data"]["audio"].setdefault("pcen_power", 0.50)
        cfg["data"]["audio"].setdefault("pcen_eps", 1e-6)
        cfg["data"]["audio"].setdefault("pcen_log1p", False)

    # ----- Derived: number of frames for your fixed-length window -----
    seconds = cfg["data"]["audio"]["seconds"]
    if cfg["data"]["audio"]["center"]:
        cfg["data"]["audio"]["n_frames"] = int(1 + (seconds * sr) // hop)
    else:
        cfg["data"]["audio"]["n_frames"] = int(1 + ((seconds * sr) - nfft) // hop)

    return cfg
