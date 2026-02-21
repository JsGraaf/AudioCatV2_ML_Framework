#!/usr/bin/env python3
"""
Plot an 80-mel triangular filterbank and (optionally) force each triangle's
peak to 1.0 to avoid the "short first triangles" look.

Defaults: sr=16000, n_fft=1024, n_mels=80, fmin=0, fmax=Nyquist.
Uses Slaney-style mel filters by default (norm='slaney', htk=False).
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.filters

# plt.rcParams.update({"font.size":16})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sr", type=int, default=16000, help="sample rate (Hz)")
    ap.add_argument("--n_fft", type=int, default=1024, help="FFT size")
    ap.add_argument("--n_mels", type=int, default=80, help="number of mel bands")
    ap.add_argument("--fmin", type=float, default=0.0, help="min frequency (Hz)")
    ap.add_argument("--fmax", type=float, default=None, help="max frequency (Hz); default=sr/2")
    ap.add_argument("--htk", action="store_true", help="use HTK mel formula (default: Slaney)")
    ap.add_argument("--no_slaney_norm", action="store_true", help="disable Slaney area normalization")
    ap.add_argument("--peak_norm", action="store_true",
                    help="renormalize each filter so its maximum equals 1.0")
    ap.add_argument("--outfile", type=str, default="mel_filterbank.svg", help="output figure file")
    args = ap.parse_args()

    sr = args.sr
    n_fft = args.n_fft
    n_mels = args.n_mels
    fmin = args.fmin
    fmax = args.fmax if args.fmax is not None else sr / 2.0

    norm = None if args.no_slaney_norm else "slaney"

    # Build mel filterbank: shape (n_mels, 1 + n_fft//2)
    mel = librosa.filters.mel(
        sr=sr,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        htk=bool(args.htk),
        norm=norm,
    )

    # Optional per-filter peak normalization (force all peaks to 1.0)
    if args.peak_norm:
        peaks = mel.max(axis=1, keepdims=True) + 1e-12
        mel = mel / peaks

    # Frequency axis for FFT bins (Hz)
    freqs = np.linspace(0, sr / 2.0, 1 + n_fft // 2, endpoint=True)
    # Mel band centers (Hz) for reference ticks
    centers_hz = librosa.mel_frequencies(
        n_mels=n_mels, fmin=fmin, fmax=fmax, htk=bool(args.htk)
    )

    # Plot
    plt.figure(figsize=(10, 3))
    for i in range(n_mels):
        plt.plot(freqs, mel[i], linewidth=0.9, alpha=0.8)

    ymin, ymax = 0.0, np.max(mel) * 1.05
    # plt.vlines(centers_hz, ymin, ymin + 0.1 * (ymax - ymin),
    #            colors="k", linestyles="dotted", linewidth=0.6, alpha=0.6)

    title_bits = [
        f"{n_mels}-Mel Filterbank",
        f"sr={sr}",
        f"n_fft={n_fft}",
        f"fmin={fmin:g}, fmax={fmax:g}",
    ]
    plt.title(" | ".join(title_bits))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (weight)")
    plt.xlim([fmin, fmax])
    plt.ylim([ymin, ymax])
    plt.grid(True, which="both", axis="x", alpha=0.2)
    plt.tight_layout()
    plt.savefig(args.outfile)
    print(f"Saved: {args.outfile}")

if __name__ == "__main__":
    main()
