#!/usr/bin/env python3
"""
Scan a directory for .ogg audio, create Mel spectrograms (power), and
aggregate them to visualize the most occurring frequencies.

Outputs (in output/):
- spectrograms/<file>.png           : per-file Mel spectrogram previews
- aggregated_mel_profile.png        : overall frequency-occurrence profile (Hz vs score)
- aggregated_mel_profile.csv        : table with [mel_hz, score]
"""

import argparse
import glob
import os
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf  # only for duration fallback (optional)
from scipy import signal

# ---------- DSP helpers ----------


def butter_bandpass(y, sr, fmin, fmax, order=6):
    if fmin is None and fmax is None:
        return y.astype(np.float32, copy=False)
    nyq = sr / 2.0
    lo = 0.0 if fmin in (None, 0) else (float(fmin) / nyq)
    hi = 1.0 if fmax in (None, 0) else (float(fmax) / nyq)
    if fmin and fmax:
        b, a = signal.butter(order, [lo, hi], btype="band")
    elif fmin:
        b, a = signal.butter(order, lo, btype="highpass")
    else:
        b, a = signal.butter(order, hi, btype="lowpass")
    return signal.lfilter(b, a, y).astype(np.float32, copy=False)


def notch_lines(y, sr, freqs_hz=None, bw_hz=60.0):
    """Zero-phase IIR notches; freqs_hz is a list like [4910.0]."""
    if not freqs_hz:
        return y
    y_f = y
    for f0 in freqs_hz:
        if f0 <= 0 or f0 >= sr / 2:
            continue
        Q = float(f0) / float(bw_hz)
        b, a = signal.iirnotch(f0 / (sr / 2), Q)
        y_f = signal.filtfilt(b, a, y_f).astype(np.float32, copy=False)
    return y_f


# ---------- Core processing ----------


def compute_mel_power(y, sr, n_fft, hop, n_mels, fmin, fmax):
    # Power Mel (power=2) to match your thesis convention
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0,
    )
    return S  # shape (n_mels, T), power scale


def normalize_per_file(S):
    # Normalize per file so loud files don't dominate the aggregate
    m = S.max()
    return S / (m + 1e-12)


def aggregate_mel_profiles(mel_list):
    """
    mel_list: list of per-file Mel power arrays (n_mels, T) already normalized.
    We collapse time per file (mean over time) -> (n_mels,), then average across files.
    """
    if not mel_list:
        raise ValueError("No Mel spectrograms to aggregate.")
    profiles = [m.mean(axis=1) for m in mel_list]  # (n_mels,)
    stacked = np.stack(profiles, axis=0)  # (N, n_mels)
    return stacked.mean(axis=0)  # (n_mels,)


def save_spectrogram_preview(S, sr, hop, fmin, fmax, out_png, title=None):
    S_db = librosa.power_to_db(S, ref=np.max)
    plt.figure(figsize=(8, 4))
    librosa.display.specshow(
        S_db,
        sr=sr,
        hop_length=hop,
        x_axis="time",
        y_axis="mel",
        fmin=fmin,
        fmax=fmax,
        cmap="magma",
    )
    plt.colorbar(label="dB")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def process_file(path, args):
    y, sr = librosa.load(path, sr=args.sr)  # sr=None uses native; else resamples
    # Optional cleanup (cheap):
    if args.hp or args.lp:
        y = butter_bandpass(y, sr, args.hp, args.lp, order=args.bp_order)
    if args.notch:
        y = notch_lines(y, sr, freqs_hz=args.notch, bw_hz=args.notch_bw)

    S = compute_mel_power(
        y=y,
        sr=sr,
        n_fft=args.n_fft,
        hop=args.hop,
        n_mels=args.n_mels,
        fmin=args.fmin,
        fmax=args.fmax,
    )
    Sn = normalize_per_file(S)
    return S, Sn, sr


def main():
    p = argparse.ArgumentParser(
        description="Overlap Mel spectrograms to find most occurring frequencies."
    )
    p.add_argument(
        "--input",
        default="input",
        help="Input directory (searched recursively) for .ogg",
    )
    p.add_argument("--output", default="output", help="Output directory")
    p.add_argument(
        "--sr", type=int, default=None, help="Target sample rate (None keeps native)"
    )
    p.add_argument("--n_fft", type=int, default=1024)
    p.add_argument("--hop", type=int, default=256)
    p.add_argument("--n_mels", type=int, default=96)
    p.add_argument("--fmin", type=float, default=200.0)
    p.add_argument("--fmax", type=float, default=7500.0)
    p.add_argument(
        "--hp",
        type=float,
        default=350.0,
        help="High-pass cutoff (Hz); 0/None to disable",
    )
    p.add_argument(
        "--lp",
        type=float,
        default=6500.0,
        help="Low-pass cutoff (Hz); 0/None to disable",
    )
    p.add_argument("--bp_order", type=int, default=6, help="Butterworth order")
    p.add_argument(
        "--notch",
        type=float,
        nargs="*",
        default=None,
        help="Frequencies to notch (Hz), e.g. --notch 4910",
    )
    p.add_argument(
        "--notch_bw", type=float, default=60.0, help="Notch bandwidth ~ -3 dB (Hz)"
    )
    args = p.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    spec_dir = out_dir / "spectrograms"
    out_dir.mkdir(parents=True, exist_ok=True)
    spec_dir.mkdir(parents=True, exist_ok=True)

    files = [Path(p) for p in glob.glob(str(in_dir / "**" / "*.ogg"), recursive=True)]
    if not files:
        print(f"No .ogg files found under {in_dir.resolve()}")
        return

    mel_list_norm = []
    first_sr = None

    print(f"Found {len(files)} .ogg files. Processing…")
    for i, fp in enumerate(sorted(files)):
        try:
            S, Sn, sr = process_file(str(fp), args)
            if first_sr is None:
                first_sr = sr
            # Save preview for this file
            out_png = spec_dir / (fp.stem + ".png")
            save_spectrogram_preview(
                S, sr, args.hop, args.fmin, args.fmax, out_png, title=fp.stem
            )
            mel_list_norm.append(Sn)
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(files)} done")
        except Exception as e:
            print(f"  Skipping {fp} due to error: {e}")

    if not mel_list_norm:
        print("No valid spectrograms to aggregate.")
        return

    # Aggregate to get overall frequency-occurrence profile
    agg = aggregate_mel_profiles(mel_list_norm)  # shape (n_mels,)
    mel_hz = librosa.mel_frequencies(n_mels=args.n_mels, fmin=args.fmin, fmax=args.fmax)

    # Save CSV
    import csv

    csv_path = Path(args.output) / "aggregated_mel_profile.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mel_hz", "score"])
        for hz, val in zip(mel_hz, agg):
            w.writerow([f"{hz:.3f}", f"{val:.8f}"])

    # Plot frequency-occurrence profile
    plt.figure(figsize=(9, 4))
    plt.plot(mel_hz, agg)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Occurrence score (avg normalized Mel power)")
    plt.title("Aggregated frequency occurrence (power Mel, per-file normalized)")
    plt.xlim([args.fmin, args.fmax])
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(Path(args.output) / "aggregated_mel_profile.png", dpi=150)
    plt.close()

    print(f"Saved: {csv_path}")
    print(f"Saved: {Path(args.output) / 'aggregated_mel_profile.png'}")
    print(f"Per-file previews in: {spec_dir}")


if __name__ == "__main__":
    main()
