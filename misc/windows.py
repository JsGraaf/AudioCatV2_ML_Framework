import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows
import os

# === CONFIGURABLE PARAMETERS ===
N = 64
FFT_LEN = 1024
FONT_SIZE = 12
OUTPUT_DIR = "./output/"
EPSILON = 1e-10  # To avoid log(0)

WINDOWS = {
    "Rectangular": np.ones(N),
    "Hanning": windows.hann(N, sym=False),
    "Hamming": windows.hamming(N, sym=False),
}

COLORS = {
    "Rectangular": "black",
    "Hanning": "blue",
    "Hamming": "green",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Shared plot ===
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Time domain plot
for name, win in WINDOWS.items():
    x = np.arange(len(win))
    axes[0].plot(x, win, label=name, color=COLORS[name])
    axes[0].fill_between(x, 0, win, color=COLORS[name], alpha=0.3)
axes[0].set_title("Window Functions", fontsize=FONT_SIZE)
axes[0].set_xlabel("Sample index", fontsize=FONT_SIZE)
axes[0].set_ylabel("Amplitude", fontsize=FONT_SIZE)
axes[0].tick_params(labelsize=FONT_SIZE)
axes[0].set_ylim(0, 1.1)
axes[0].legend(fontsize=FONT_SIZE)
for spine in axes[0].spines.values():
    spine.set_visible(False)

# Frequency domain plot
for name, win in WINDOWS.items():
    freqs = np.fft.rfftfreq(FFT_LEN, d=1.0)
    fft_mag = np.abs(np.fft.rfft(win, n=FFT_LEN))
    fft_mag /= np.max(fft_mag)
    fft_db = 20 * np.log10(fft_mag + EPSILON)
    axes[1].plot(freqs, fft_db, label=name, color=COLORS[name])
    axes[1].fill_between(freqs, -100, fft_db, color=COLORS[name], alpha=0.3)
axes[1].set_title("FFT Magnitude (dB)", fontsize=FONT_SIZE)
axes[1].set_xlabel("Normalized Frequency", fontsize=FONT_SIZE)
axes[1].set_ylabel("Magnitude [dB]", fontsize=FONT_SIZE)
axes[1].tick_params(labelsize=FONT_SIZE)
axes[1].set_xlim(0, 0.5)
axes[1].set_ylim(-100, 5)
axes[1].legend(fontsize=FONT_SIZE, loc="upper right")
for spine in axes[1].spines.values():
    spine.set_visible(False)

plt.tight_layout()
svg_path = os.path.join(OUTPUT_DIR, "window_comparison.svg")
plt.savefig(svg_path, format="svg", bbox_inches="tight")
plt.show()

