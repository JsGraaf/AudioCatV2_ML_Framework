import numpy as np
import tensorflow as tf


# Augments
def aug_gaussian_noise_tf(audio, snr_db):
    """
    Add Gaussian noise to a waveform at a given SNR (dB).

    Args:
        audio (tf.Tensor): 1D or 2D waveform tensor (e.g. [time] or [batch, time]).
        snr_db (float): Target signal-to-noise ratio in dB.

    Returns:
        tf.Tensor: Noisy waveform with the specified SNR.
    """
    # Calculate RMS of the signal
    rms_signal = tf.sqrt(tf.reduce_mean(tf.square(audio), axis=-1, keepdims=True))

    # Convert SNR from dB to linear scale
    snr_linear = 10 ** (snr_db / 20.0)

    # Desired noise RMS
    rms_noise = rms_signal / snr_linear

    # Generate Gaussian noise
    noise = tf.random.get_global_generator().normal(
        shape=tf.shape(audio), dtype=tf.float32
    )

    # Normalize noise to unit RMS
    rms_noise_current = tf.sqrt(
        tf.reduce_mean(tf.square(noise), axis=-1, keepdims=True)
    )
    noise = noise * (rms_noise / (rms_noise_current + 1e-8))

    return audio + noise


def aug_loudness_norm_tf(audio, target_db=-20.0, eps=1e-8):
    rms = tf.sqrt(tf.reduce_mean(tf.square(audio), axis=-1, keepdims=True) + eps)
    rms_db = 20.0 * tf.math.log(rms + eps) / tf.math.log(10.0)
    gain = 10.0 ** ((target_db - rms_db) / 20.0)
    return audio * gain


def aug_specaugment_tf(
    spec, max_freq_masks=2, max_time_masks=2, max_freq_width=16, max_time_width=32
):
    spec = tf.identity(spec)

    def _mask_freq(s):
        f = tf.shape(s)[-2]
        t = tf.shape(s)[-1]
        w = tf.random.get_global_generator().uniform(
            (), 0, max_freq_width + 1, dtype=tf.int32
        )
        start = tf.random.get_global_generator().uniform(
            (), 0, tf.maximum(f - w, 1), dtype=tf.int32
        )
        mask = tf.concat(
            [tf.ones([start, t]), tf.zeros([w, t]), tf.ones([f - start - w, t])], axis=0
        )
        return s * mask

    def _mask_time(s):
        f = tf.shape(s)[-2]
        t = tf.shape(s)[-1]
        w = tf.random.get_global_generator().uniform(
            (), 0, max_time_width + 1, dtype=tf.int32
        )
        start = tf.random.get_global_generator().uniform(
            (), 0, tf.maximum(t - w, 1), dtype=tf.int32
        )
        mask = tf.concat(
            [tf.ones([f, start]), tf.zeros([f, w]), tf.ones([f, t - start - w])], axis=1
        )
        return s * mask

    for _ in range(2):  # loop vars must be static; use fixed counts
        spec = _mask_freq(spec)
        spec = _mask_time(spec)
    return spec
