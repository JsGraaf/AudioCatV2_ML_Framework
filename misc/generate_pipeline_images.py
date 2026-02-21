import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

from pipeline_components.augments import (
    aug_gaussian_noise_np,
    aug_loudness_norm_np,
    aug_specaugment_np,
    mixup_spec_power,
)

# ── Config ─────────────────────────────────────────────────────────────────────
FIGSIZE = (10, 3)
SECOND_FIGSIZE = (10, 3)
DPI = 300

def new_fig(figsize=None):
    # Use constrained layout from the start; consistent canvas size
    if (figsize is None):
        return plt.subplots(figsize=FIGSIZE, layout="constrained")
    else:
        return plt.subplots(figsize=figsize, layout="constrained")

def save_fig(fig, path):
    fig.savefig(path, dpi=DPI)  # no tight_layout(), no bbox_inches
    plt.close(fig)

# ── Prep ───────────────────────────────────────────────────────────────────────
os.makedirs("output/pipeline", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, required=True, help="Path to audio file (e.g., input/example.wav)")
parser.add_argument("--window", type=float, default=2.0, help="Window size in seconds")
args = parser.parse_args()

# ── Step 1: Audio Acquisition ──────────────────────────────────────────────────
y, sr = librosa.load(args.file, sr=None, mono=True)

duration = len(y) / sr
t = np.arange(len(y)) / sr
n_windows = int(np.ceil(duration / args.window))
boundaries = [i * args.window for i in range(n_windows + 1) if i * args.window <= duration]

fig, ax = new_fig()
librosa.display.waveshow(y, sr=sr, ax=ax)
ax.set_title("Raw Waveform")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")
ax.set_ylim(-1, 1)
ax.set_xlim(0, np.ceil(duration))
ax.set_xticks(boundaries)
save_fig(fig, "output/pipeline/stage1.svg")

# ── Step 2: Segmentation (overview) ────────────────────────────────────────────


fig, ax = new_fig()
librosa.display.waveshow(y, sr=sr, alpha=1.0, ax=ax)
for b in boundaries:
    ax.axvline(b, color="red", linestyle="--", linewidth=1)
ax.set_title(f"Segmentation into {args.window:.1f}-second windows")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")
ax.set_ylim(-1, 1)
ax.set_xticks(boundaries)
ax.set_xlim(0, np.ceil(duration))
save_fig(fig, "output/pipeline/stage2.svg")

# Select one window (2nd chunk)
w0 = int(1 * sr * args.window)
w1 = int(2 * sr * args.window)
window_1 = y[w0:w1]

fig, ax = new_fig()
librosa.display.waveshow(window_1, sr=sr, ax=ax)
ax.set_title("Window 2–4" if args.window == 2 else f"Window [{w0/sr:.1f}, {w1/sr:.1f}] s")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")
ax.set_ylim(-1, 1)
ax.set_xlim(0, 3)
save_fig(fig, "output/pipeline/stage2_window.svg")

fig, ax = new_fig(SECOND_FIGSIZE)
librosa.display.waveshow(window_1, sr=sr, ax=ax)
ax.set_title("Window 2–4" if args.window == 2 else f"Window [{w0/sr:.1f}, {w1/sr:.1f}] s")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")
ax.set_ylim(-1, 1)
ax.set_xlim(0, 3)
save_fig(fig, "output/pipeline/stage2_window_aspect.svg")

# ── Step 3: Augmentations (waveform) ───────────────────────────────────────────
# Loudness norm
loudness = aug_loudness_norm_np(window_1)
fig, ax = new_fig()
librosa.display.waveshow(loudness, sr=sr, ax=ax)
ax.set_title("Loudness Normalization")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")
ax.set_ylim(-1, 1)
ax.set_xlim(0, 3)
save_fig(fig, "output/pipeline/stage3_loud.svg")

# Gaussian noise
gaussian = aug_gaussian_noise_np(window_1, 6)
fig, ax = new_fig()
librosa.display.waveshow(gaussian, sr=sr, ax=ax)
ax.set_title("Gaussian Noise")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")
ax.set_ylim(-1, 1)
ax.set_xlim(0, 3)
save_fig(fig, "output/pipeline/stage3_gaus.svg")

# Combined (loudness + noise)
combined = aug_gaussian_noise_np(loudness, 6)
fig, ax = new_fig()
librosa.display.waveshow(combined, sr=sr, ax=ax)
ax.set_title("Loudness + Gaussian Noise")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")
ax.set_ylim(-1, 1)
ax.set_xlim(0, 3)
save_fig(fig, "output/pipeline/stage3_combined.svg")

# 3. STFT Power
D = librosa.stft(window_1)
S_power = np.abs(D) ** 2
fig, ax = new_fig(SECOND_FIGSIZE)
img = librosa.display.specshow(S_power, sr=sr, x_axis="time", y_axis="linear", ax=ax, cmap="gray_r")
cbar = fig.colorbar(img, ax=ax, aspect=20, pad=0.01)
ax.set_xlim(0, 3)
cbar.set_label("Power")
ax.set_ylabel("Frequency (Hz)")
ax.set_xlabel("Time (s)")
ax.set_title("STFT Power")
fig.savefig(
    "output/pipeline/stage_2_stft_power.svg",
)
plt.close(fig)

# 5. Mel Spectrogram (power, power=2.0)

S_power_mel = librosa.feature.melspectrogram(y=loudness, sr=sr, power=2.0, n_mels=128)
S_power_mel = librosa.power_to_db(S_power_mel)
fig, ax = new_fig(SECOND_FIGSIZE)
img = librosa.display.specshow(
    S_power_mel, sr=sr, x_axis="time", y_axis="mel", ax=ax, cmap="gray"
)
cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB", aspect=20, pad=0.01)
cbar.set_label("Power (dB ref: max)")
ax.set_title("Loudness Mel Spectrogram (Power)")
ax.set_ylabel("Frequency (Hz)")
ax.set_xlabel("Time (s)")
fig.savefig(
    "output/pipeline/stage_2_loudness.svg",
)
plt.close(fig)

S_power_mel = librosa.feature.melspectrogram(y=gaussian, sr=sr, power=2.0, n_mels=128)
S_power_mel = librosa.power_to_db(S_power_mel, ref=np.max)

fig, ax = new_fig(SECOND_FIGSIZE)
img = librosa.display.specshow(
    S_power_mel, sr=sr, x_axis="time", y_axis="mel", ax=ax, cmap="gray_r", vmin=-80, vmax=0
)
cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB", aspect=20, pad=0.01
)
cbar.set_label("Power (dB ref: max)")
ax.set_title("Gaussian Mel Spectrogram (Power)")
ax.set_ylabel("Frequency (Hz)")
ax.set_xlabel("Time (s)")
fig.savefig(
    "output/pipeline/stage_2_gaussian.svg",
)
plt.close(fig)

S_power_mel = librosa.feature.melspectrogram(y=combined, sr=sr, power=2.0, n_mels=128)
S_power_mel = librosa.power_to_db(S_power_mel, ref=np.max)
fig, ax = new_fig(SECOND_FIGSIZE)
img = librosa.display.specshow(
    S_power_mel, sr=sr, x_axis="time", y_axis="mel", ax=ax, cmap="gray_r", vmin=-80, vmax=0
)
cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB", aspect=20, pad=0.01
)
cbar.set_label("Power (dB ref: max)")
ax.set_title("Combined Mel Spectrogram (Power)")
ax.set_ylabel("Frequency (Hz)")
ax.set_xlabel("Time (s)")
fig.savefig(
    "output/pipeline/stage_2_combined.svg",
)
plt.close(fig)

S_power_mel = librosa.feature.melspectrogram(y=window_1, sr=sr, power=2.0, n_mels=128)
fig, ax = new_fig(SECOND_FIGSIZE)
img = librosa.display.specshow(
    S_power_mel, sr=sr, x_axis="time", y_axis="mel", ax=ax, cmap="gray_r", vmin=np.percentile(S_power_mel, 2), vmax=np.percentile(S_power_mel, 98)
)
cbar = fig.colorbar(img, ax=ax, aspect=20, pad=0.01)
cbar.set_label("Power")
ax.set_title("Mel Spectrogram (Power)")
ax.set_ylabel("Frequency (Hz)")
ax.set_xlabel("Time (s)")
fig.savefig(
    "output/pipeline/stage_2_mel_power.svg",
)
plt.close(fig)

# 6. Log-Mel Spectrogram (power, log10)
S_log_power_mel = np.log10(S_power_mel + 1e-10)
fig, ax = plt.subplots(figsize=SECOND_FIGSIZE)
img = librosa.display.specshow(
    S_log_power_mel, sr=sr, x_axis="time", y_axis="mel", ax=ax, cmap="gray_r"
)
cbar = fig.colorbar(img, ax=ax, aspect=20, pad=0.01)
cbar.set_label("log₁₀(Power)")
ax.set_title("Log-Mel Spectrogram (Power)")
fig.savefig(
    "output/pipeline/stage_2_logmel_power.svg",
)
plt.close(fig)

# 7. Log-Mel Spectrogram (power in dB)
S_db = librosa.power_to_db(
    S_power_mel, ref=np.max
)  # dB relative to max ≈ [−80, 0]

fig, ax = plt.subplots(figsize=SECOND_FIGSIZE)
img = librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="mel", ax=ax, cmap="gray_r")

cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB", aspect=20, pad=0.01)
cbar.set_label("Power (dB ref: max)")
ax.set_title("Log-Mel Spectrogram (Power, dB)")

fig.savefig(
    "output/pipeline/stage_2_logmel_power_db.svg",
    bbox_inches="tight",
    pad_inches=0,
)
plt.close(fig)

# ── Step 4: Features (Mel & PCEN) ─────────────────────────────────────────────
spec = librosa.feature.melspectrogram(
    y=combined, sr=sr, n_fft=2048, hop_length=512, n_mels=128, power=2.0
)

S_log_power_mel = np.log10(spec + 1e-10)

fig, ax = new_fig()
im = librosa.display.specshow(
    S_log_power_mel, sr=sr, hop_length=512, x_axis="time", y_axis="mel", cmap="gray_r", ax=ax
)
cbar = fig.colorbar(im, ax=ax, pad=0.01, format="%+2.0f")
cbar.set_label("Power (dB ref: max)")
ax.set_title("Log-Mel Spectrogram (Power, log10)")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Frequency (Hz)")
save_fig(fig, "output/pipeline/stage4_logmel.svg")

spec_db = librosa.power_to_db(spec, ref=np.max)

fig, ax = new_fig()
im = librosa.display.specshow(
    S_db, sr=sr, hop_length=512, x_axis="time", y_axis="mel", cmap="gray_r", ax=ax
)
cbar = fig.colorbar(im, ax=ax, pad=0.01, format="%+2.0f dB")
cbar.set_label("Power (dB ref: max)")
ax.set_title("Log-Mel Spectrogram (Power, dB)")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Frequency (Hz)")
save_fig(fig, "output/pipeline/stage4_logmel_db.svg")

# PCEN (from mel magnitude)
spec_mag = librosa.feature.melspectrogram(
    y=combined, sr=sr, n_fft=2048, hop_length=512, n_mels=80, power=1.0
)
pcen = librosa.pcen(S=spec_mag, sr=sr, hop_length=512)
fig, ax = new_fig()
im = librosa.display.specshow(
    pcen, sr=sr, hop_length=512, x_axis="time", y_axis="mel",
    cmap="gray_r", vmin=0, ax=ax
)
ax.set_title("PCEN Spectrogram")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Frequency (Hz)")
save_fig(fig, "output/pipeline/stage4_pcen.svg")

# ── Step 5: Additional augmentations (SpecAugment & MixUp) ────────────────────

# MixUp
window_2 = y[int(4 * sr * args.window) : int(5 * sr * args.window)]
loudness_2 = aug_loudness_norm_np(window_2)
combined_2 = aug_gaussian_noise_np(loudness_2, 6)
spec_2 = librosa.feature.melspectrogram(
    y=window_2, sr=sr, n_fft=2048, hop_length=512, n_mels=128, power=2.0
)
spec_2_db = librosa.power_to_db(spec_2, ref=np.max)

fig, ax = new_fig()
im = librosa.display.specshow(
    S_db, sr=sr, hop_length=512, x_axis="time", y_axis="mel",cmap="gray_r", ax=ax
)
cbar = fig.colorbar(im, ax=ax, pad=0.01, format="%+2.0f dB")
cbar.set_label("Power (dB ref: max)")
ax.set_title("Mixup (Power, dB)")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Frequency (Hz)")
save_fig(fig, "output/pipeline/stage5_mixa.svg")

fig, ax = new_fig()
im = librosa.display.specshow(
    spec_2_db, sr=sr, hop_length=512, x_axis="time", y_axis="mel", cmap="gray_r", ax=ax
)
cbar = fig.colorbar(im, ax=ax, pad=0.01, format="%+2.0f dB")
cbar.set_label("Power (dB ref: max)")
ax.set_title("Mixup (Power, dB)")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Frequency (Hz)")
save_fig(fig, "output/pipeline/stage5_mixb.svg")


mixup_db = mixup_spec_power(S_db, spec_2_db, 0.4)

fig, ax = new_fig()
im = librosa.display.specshow(
    mixup_db, sr=sr, hop_length=512, x_axis="time", y_axis="mel",
    vmin=-80, vmax=0, cmap="gray_r", ax=ax
)
cbar = fig.colorbar(im, ax=ax, pad=0.01, format="%+2.0f dB")
cbar.set_label("Power (dB ref: max)")
ax.set_title("Mixup (Power, dB)")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Frequency (Hz)")
save_fig(fig, "output/pipeline/stage5_mixup.svg")

# SpecAugment on mixup (visual-only)
mixup_specaug = aug_specaugment_np(S_db, max_freq_width=2, max_time_width=2)
fig, ax = new_fig()
im = librosa.display.specshow(
    mixup_specaug, sr=sr, hop_length=512, x_axis="time", y_axis="mel",cmap="gray_r", ax=ax
)
cbar = fig.colorbar(im, ax=ax, pad=0.01, format="%+2.0f dB")
cbar.set_label("Power (dB ref: max)")
ax.set_title("SpecAugment (Power, dB)")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Frequency (Hz)")
save_fig(fig, "output/pipeline/stage5_specaug.svg")

# ── Step 6 / 7 placeholders ───────────────────────────────────────────────────
# (Add your training/testing plots here using the same new_fig()/save_fig() pattern)
