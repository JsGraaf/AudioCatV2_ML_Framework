from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import librosa
import numpy as np
import tensorflow as tf
from librosa._typing import _STFTPad
from scipy import signal

from augments import aug_gaussian_noise_tf, aug_loudness_norm_tf, aug_specaugment_tf


def load_ogg_librosa(path: Sequence[str], start: float, end: float, sr: int):
    y, _ = librosa.load(
        path.decode("utf-8"), offset=start, duration=(end - start), sr=sr
    )
    return y


def generate_mel_spectrogram(
    audio,
    sr,
    n_fft,
    n_mels,
    hop_length,
    win_length,
    window,
    center,
    pad_mode,
    power,
    fmin,
    fmax,
    norm,
):
    # Generate mel spectrogram
    spec = librosa.feature.melspectrogram(
        y=audio,
        sr=int(sr),
        n_fft=int(n_fft),
        n_mels=int(n_mels),
        hop_length=int(hop_length),
        win_length=None if win_length in (None, "None", b"None") else int(win_length),
        window=window if isinstance(window, str) else window.decode("utf-8"),
        center=bool(center),
        pad_mode=pad_mode if isinstance(pad_mode, str) else pad_mode.decode("utf-8"),
        power=float(power),
        fmin=int(fmin),
        fmax=int(fmax),
        norm=(
            None
            if norm in (None, "None", b"None")
            else (norm if isinstance(norm, str) else norm.decode("utf-8"))
        ),
    )

    # Convert to dB
    spec = librosa.power_to_db(spec, ref=np.max)

    bg = np.median(spec, axis=1, keepdims=True)
    spec = spec - bg

    # Clamp to a fixed window
    spec = np.clip(spec, -30.0, +15)

    # Scale to 0, 1
    spec = (spec + 30.0) / 45.0

    return spec.astype(np.float32)


def apply_with_prob(x, p, aug_fn):
    """aug_fn must be a zero-arg callable returning a tensor like x."""
    gen = tf.random.get_global_generator()
    u = gen.uniform((), 0.01, 1.0)
    return tf.cond(u < p, true_fn=aug_fn, false_fn=lambda: tf.identity(x))


def notch_lines(y, sr, freqs_hz, bw_hz=60.0):
    """
    Apply zero-phase IIR notch filters at given frequencies.
    bw_hz is the -3 dB bandwidth; Q = freqs_hz / bw_hz.
    """
    y_f = y
    if freqs_hz == 0:
        return y
    if freqs_hz >= sr / 2:
        return y

    Q = float(freqs_hz) / float(bw_hz)
    b, a = signal.iirnotch(w0=freqs_hz / (sr / 2), Q=Q)
    # zero-phase to avoid phase distortion of short syllables
    y_f = signal.filtfilt(b, a, y_f).astype(np.float32, copy=False)
    return y_f


def audio_pipeline(
    file_info: Tuple[
        Sequence[str], float, float, float
    ],  # Filename, start time, end time, notch
    config: Dict,
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

    # Loudness Normalization
    target_dbfs = tf_g1.uniform(
        (),
        config["data"]["augments"]["loud_range"][0],
        config["data"]["augments"]["loud_range"][1],
    )
    processed = apply_with_prob(
        audio_file,
        config["data"]["augments"]["p_loud"],
        lambda: aug_loudness_norm_tf(audio_file, target_dbfs),
    )

    # Add Gaussian noise
    gaussian_snr = tf_g1.uniform(
        [],
        config["data"]["augments"]["gaus_range"][0],
        config["data"]["augments"]["gaus_range"][1],
    )
    processed = apply_with_prob(
        processed,
        config["data"]["augments"]["p_gaus"],
        lambda: aug_gaussian_noise_tf(processed, gaussian_snr),
    )

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
        config["data"]["audio"]["butterworth_order"],
        [
            config["data"]["augments"]["band_low_freq"],
            config["data"]["augments"]["band_high_freq"],
        ],
        fs=config["data"]["audio"]["sample_rate"],
        btype="band",
    )
    band_filter = tf.py_function(
        signal.lfilter, [b, a, processed], Tout=tf.float32, name="Filter"
    )

    audio_config = config["data"]["audio"]

    db_mel_spectrogram = tf.numpy_function(
        generate_mel_spectrogram,
        [
            band_filter,
            audio_config["sample_rate"],
            audio_config["n_fft"],
            audio_config["n_mels"],
            audio_config["hop_length"],
            audio_config["win_length"],
            audio_config["window"],
            audio_config["center"],
            audio_config["pad_mode"],
            audio_config["power"],
            audio_config["fmin"],
            audio_config["fmax"],
            audio_config["norm"],
        ],
        Tout=tf.float32,
    )

    db_mel_spectrogram = tf.ensure_shape(
        db_mel_spectrogram,
        shape=(config["data"]["audio"]["n_mels"], config["data"]["audio"]["n_frames"]),
    )

    db_mel_spectrogram = apply_with_prob(
        db_mel_spectrogram,
        config["data"]["augments"]["p_spec"],
        lambda: aug_specaugment_tf(
            db_mel_spectrogram,
            config["data"]["augments"]["spec_freq_masks"],
            config["data"]["augments"]["spec_time_masks"],
            config["data"]["augments"]["spec_max_freq_width"],
            config["data"]["augments"]["spec_max_time_width"],
        ),
    )

    return db_mel_spectrogram
