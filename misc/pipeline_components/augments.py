import numpy as np

# ----------------------
# Utilities
# ----------------------
def _as_float32(x):
    return np.asarray(x, dtype=np.float32)

# ----------------------
# Augments (NumPy)
# ----------------------
def aug_gaussian_noise_np(audio, snr_db, rng=None, eps=1e-8):
    """
    Add Gaussian noise to a waveform at a given SNR (dB).

    Args:
        audio: np.ndarray, shape [T] or [B, T], float.
        snr_db: float, target SNR in dB (signal/noise).
        rng: optional np.random.Generator for reproducibility.
        eps: numerical epsilon.

    Returns:
        np.ndarray with noise added at specified SNR.
    """
    x = _as_float32(audio)
    if rng is None:
        rng = np.random.default_rng()

    # RMS of the signal over time axis
    rms_signal = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)

    # Convert SNR dB -> linear (amplitude ratio)
    snr_linear = 10.0 ** (snr_db / 20.0)

    # Desired noise RMS
    rms_noise = rms_signal / (snr_linear + eps)

    # Generate Gaussian noise with unit RMS, then scale
    noise = rng.normal(loc=0.0, scale=1.0, size=x.shape).astype(np.float32)
    rms_noise_current = np.sqrt(np.mean(noise**2, axis=-1, keepdims=True) + eps)
    noise = noise * (rms_noise / (rms_noise_current + eps))

    return x + noise


def aug_loudness_norm_np(audio, target_db=-20.0, eps=1e-8):
    """
    Loudness (RMS) normalization to a target dBFS-like value.
    Args:
        audio: np.ndarray [T] or [B, T]
        target_db: float, target level in dB (20*log10(RMS)).
        eps: numerical epsilon.
    """
    x = _as_float32(audio)
    rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)
    # 20*log10(rms)
    rms_db = 20.0 * np.log10(rms + eps)
    gain = 10.0 ** ((target_db - rms_db) / 20.0)
    return x * gain.astype(np.float32)


def aug_specaugment_np(
    spec,
    max_freq_masks=2,
    max_time_masks=2,
    max_freq_width=16,
    max_time_width=32,
    rng=None,
):
    """
    Apply SpecAugment-style frequency and time masking on a spectrogram.

    Args:
        spec: np.ndarray [F, T] (float).
        max_*_masks: int, how many masks of each type to apply.
        max_*_width: int, maximum mask width (inclusive of 0).
        rng: optional np.random.Generator.

    Returns:
        np.ndarray [F, T] with masks applied (copy).
    """
    if rng is None:
        rng = np.random.default_rng()

    s = _as_float32(spec).copy()
    F, T = s.shape

    # Frequency masks
    for _ in range(int(max_freq_masks)):
        if max_freq_width <= 0 or F == 0:
            break
        w = rng.integers(low=0, high=max_freq_width + 1)
        if w == 0:
            continue
        start = rng.integers(low=0, high=max(1, F - w + 1))
        s[start : start + w, :] = 0.0

    # Time masks
    for _ in range(int(max_time_masks)):
        if max_time_width <= 0 or T == 0:
            break
        w = rng.integers(low=0, high=max_time_width + 1)
        if w == 0:
            continue
        start = rng.integers(low=0, high=max(1, T - w + 1))
        s[:, start : start + w] = 0.0

    return s

def mixup_spec_power(S1, S2, lam: float):
    """
    Mix two power spectrograms linearly.
    S1, S2: [F, T] power (e.g., mel power), same shape
    lam: lambda in [0,1]
    """
    S1 = np.asarray(S1, dtype=np.float32)
    S2 = np.asarray(S2, dtype=np.float32)
    assert S1.shape == S2.shape, "Spectrograms must have same shape"
    lam = float(np.clip(lam, 0.0, 1.0))
    return lam * S1 + (1.0 - lam) * S2