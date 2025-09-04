from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import librosa
import numpy as np
import tensorflow as tf
from librosa._typing import _STFTPad
from scipy import signal


def load_ogg_librosa(path: Sequence[str], start: float, end: float, sr: int):
    y, _ = librosa.load(
        path.decode("utf-8"), offset=start, duration=(end - start), sr=sr
    )
    return y


def generate_mel_spectrogram(
    audio,
    sr=16000,
    n_fft=1024,
    n_mels=128,
    hop_length=512,
    win_length=None,
    window: str = "hann",
    center=True,
    pad_mode: _STFTPad = "constant",
    power=2.0,
    fmin=200,
    fmax=8000,
    norm="slaney",
):
    # Generate mel spectrogram
    spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        n_mels=n_mels,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
        power=power,
        fmin=fmin,
        fmax=fmax,
        norm=norm,
    )
    # Convert to dB
    spec = librosa.power_to_db(spec, ref=np.max)
    return spec


def apply_with_prob(x, p, aug_fn):
    """aug_fn must be a zero-arg callable returning a tensor like x."""
    gen = tf.random.get_global_generator()
    u = gen.uniform((), 0, 1.0)
    return tf.cond(u < p, true_fn=aug_fn, false_fn=lambda: tf.identity(x))


def audio_pipeline(
    file_info: Tuple[Sequence[str], float, float],  # Filename, start time, end time
    config: Dict,
    augment: bool = False,
):
    # Get the tf random generator
    tf_g1 = tf.random.get_global_generator()

    # Load audio file as tensor
    audio_file = tf.numpy_function(
        load_ogg_librosa,
        [
            file_info[0],
            file_info[1],
            file_info[2],
            config["data"]["audio"]["sample_rate"],
        ],
        tf.float32,
    )

    # if augment:
    #     # Loudness Normalization
    #     target_dbfs = tf_g1.uniform((), loud_range[0], loud_range[1])
    #     processed = apply_with_prob(
    #         audio_file, p_loud, lambda: aug_loudness_norm_tf(processed, target_dbfs)
    #     )
    #
    #     # Add Gaussian noise
    #     gaussian_snr = tf_g1.uniform([], gaus_range[0], gaus_range[1])
    #     processed = apply_with_prob(
    #         audio_file, p_gaus, lambda: aug_gaussian_noise_tf(processed, gaussian_snr)
    #     )
    # else:
    processed = audio_file

    n = tf.shape(processed)[0]
    # If shorter than desired, pad or tile
    if config["data"]["audio"]["fill_type"] == "pad":
        pad = tf.maximum(
            0,
            config["data"]["audio"]["sample_rate"] * config["data"]["audio"]["seconds"]
            - n,
        )
        processed = tf.pad(processed, paddings=[[0, pad]])
    else:  # tile
        repeats = tf.maximum(
            1,
            tf.cast(
                tf.math.ceil(
                    (
                        config["data"]["audio"]["sample_rate"]
                        * config["data"]["audio"]["seconds"]
                    )
                    / tf.cast(n, tf.float32)
                ),
                tf.int32,
            ),
        )
        processed = tf.tile(processed, [repeats])
        processed = processed[
            : config["data"]["audio"]["sample_rate"]
            * config["data"]["audio"]["seconds"]
        ]

    # Pass through butterworth bandpass filter
    b, a = signal.butter(
        4, [200, 7999], fs=config["data"]["audio"]["sample_rate"], btype="band"
    )
    band_filter = tf.py_function(
        signal.lfilter, [b, a, processed], Tout=tf.float32, name="Filter"
    )

    db_mel_spectrogram = tf.numpy_function(
        generate_mel_spectrogram,
        [band_filter, config["data"]["audio"]["sample_rate"]],
        Tout=tf.float32,
    )

    db_mel_spectrogram = tf.ensure_shape(
        db_mel_spectrogram,
        shape=(config["data"]["audio"]["n_mels"], config["data"]["audio"]["n_frames"]),
    )

    return db_mel_spectrogram
