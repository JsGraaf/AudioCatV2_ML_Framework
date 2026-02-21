import os

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
INPUT_DIR = "input"
OUTPUT_DIR = "output"
FONT_SIZE = 18
os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.rcParams.update({"font.size": FONT_SIZE})

# Get first .mp3 file
mp3_files = [
    f for f in os.listdir(INPUT_DIR) if (f.endswith(".wav") or f.endswith("mp3"))
]
print(mp3_files)
if not mp3_files:
    raise FileNotFoundError("No .mp3 files found in the input directory.")

for f in mp3_files:
    filename = f
    filepath = os.path.join(INPUT_DIR, filename)
    base_name = os.path.splitext(filename)[0]

    # Load first 3 seconds
    y, sr = librosa.load(filepath, sr=None, duration=5)

    # 1. Raw waveform
    fig, ax = plt.subplots(figsize=(20, 2))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title("Raw Waveform")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    fig.savefig(
        os.path.join(OUTPUT_DIR, f"{base_name}_waveform.svg"),
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close(fig)

    # 2. STFT Magnitude
    D = librosa.stft(y)
    S_mag = np.abs(D)
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(S_mag, sr=sr, x_axis="time", y_axis="linear", ax=ax)
    cbar = fig.colorbar(img, ax=ax, aspect=20, pad=0.01)
    cbar.set_label("Amplitude")
    ax.set_title("STFT Magnitude")
    fig.savefig(
        os.path.join(OUTPUT_DIR, f"{base_name}_stft_magnitude.svg"),
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close(fig)
    
    # 3. STFT Power
    D = librosa.stft(y)
    S_power = np.abs(D) ** 2
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(S_power, sr=sr, x_axis="time", y_axis="linear", ax=ax)
    cbar = fig.colorbar(img, ax=ax, aspect=20, pad=0.01)
    cbar.set_label("Power")
    ax.set_title("STFT Power")
    fig.savefig(
        os.path.join(OUTPUT_DIR, f"{base_name}_stft_power.svg"),
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close(fig)

    # 4. Mel Spectrogram (Magnitude, power=1.0)
    S_magnitude_mel = librosa.feature.melspectrogram(y=y, sr=sr, power=1.0, n_mels=128)
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(
        S_magnitude_mel, sr=sr, x_axis="time", y_axis="mel", ax=ax
    )
    cbar = fig.colorbar(img, ax=ax, aspect=20, pad=0.01)
    cbar.set_label("Magnitude")
    ax.set_title("Mel Spectrogram (Magnitude)")
    fig.savefig(
        os.path.join(OUTPUT_DIR, f"{base_name}_mel_magnitude.svg"),
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close(fig)

    # 5. Mel Spectrogram (power, power=2.0)
    S_power_mel = librosa.feature.melspectrogram(y=y, sr=sr, power=2.0, n_mels=128)
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(
        S_power_mel, sr=sr, x_axis="time", y_axis="mel", ax=ax, cmap="magma"
    )
    cbar = fig.colorbar(img, ax=ax, aspect=20, pad=0.01)
    cbar.set_label("Power")
    ax.set_title("Mel Spectrogram (Power)")
    fig.savefig(
        os.path.join(OUTPUT_DIR, f"{base_name}_mel_power.svg"),
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close(fig)

    # 6. Log-Mel Spectrogram (power, log10)
    S_log_power_mel = np.log10(S_power_mel + 1e-10)
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(
        S_log_power_mel, sr=sr, x_axis="time", y_axis="mel", ax=ax, cmap="magma"
    )
    cbar = fig.colorbar(img, ax=ax, aspect=20, pad=0.01)
    cbar.set_label("log₁₀(Power)")
    ax.set_title("Log-Mel Spectrogram (Power)")
    fig.savefig(
        os.path.join(OUTPUT_DIR, f"{base_name}_logmel_power.svg"),
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close(fig)

    # 7. Log-Mel Spectrogram (power in dB)
    S_db = librosa.power_to_db(
        S_power_mel, ref=np.max
    )  # dB relative to max ≈ [−80, 0]

    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="mel", ax=ax)
    cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB", aspect=20, pad=0.01)
    cbar.set_label("Power (dB ref: max)")
    ax.set_title("Log-Mel Spectrogram (Power, dB)")

    fig.savefig(
        os.path.join(OUTPUT_DIR, f"{base_name}_logmel_power_db.svg"),
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close(fig)

    # 8. PCEN Spectrogram (Power in dB)
    S_pcen = librosa.pcen(S_magnitude_mel, sr=sr).astype(np.float32)
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(S_pcen, sr=sr, x_axis="time", y_axis="mel", ax=ax)
    cbar = fig.colorbar(img, ax=ax, aspect=20, pad=0.01)
    cbar.set_label("PCEN")

    ax.set_title("PCEN Mel Spectrogram")

    fig.savefig(
        os.path.join(OUTPUT_DIR, f"{base_name}_pcen.svg"),
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close(fig)
    print(
        f"✅ 8-stage audio processing visualizations saved in '{OUTPUT_DIR}' for file: {filename}"
    )
