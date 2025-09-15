#!/usr/bin/env python3
"""
Average Mel spectrogram over all audio files in a directory.

- Scans input/ recursively for: .ogg, .wav, .flac, .mp3
- Computes power Mel spectrogram for each file
- Normalizes per file (max=1) so loud files don't dominate
- Resamples the time axis to a fixed number of frames
- Averages in POWER domain across files
- Displays and saves the averaged Mel spectrogram

Usage:
    python avg_mel.py --input input --output output --target_frames 128
"""

import argparse
import glob
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# ---------- helpers ----------


def resize_time_axis(S: np.ndarray, target_frames: int) -> np.ndarray:
    """
    Linearly resample the time axis of a (n_mels, T) array to (n_mels, target_frames).
    """
    n_mels, T = S.shape
    if T == target_frames:
        return S
    # old time positions (0..1), new positions (0..1)
    old_t = np.linspace(0.0, 1.0, num=T, endpoint=True)
    new_t = np.linspace(0.0, 1.0, num=target_frames, endpoint=True)
    out = np.empty((n_mels, target_frames), dtype=S.dtype)
    for i in range(n_mels):
        out[i] = np.interp(new_t, old_t, S[i])
    return out


def mel_power(y, sr, n_fft, hop, n_mels, fmin, fmax):
    """
    Power Mel spectrogram (power=2.0). Returns (n_mels, T) power (not dB).
    """
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
    return S


# ---------- main ----------


def main():
    ap = argparse.ArgumentParser(
        description="Average Mel spectrogram across a directory of audio files."
    )
    ap.add_argument(
        "--input", default="input", help="Input directory (searched recursively)"
    )
    ap.add_argument("--output", default="output", help="Output directory")
    ap.add_argument(
        "--sr", type=int, default=22050, help="Target sample rate (None keeps native)"
    )
    ap.add_argument("--n_fft", type=int, default=1024)
    ap.add_argument("--hop", type=int, default=256)
    ap.add_argument("--n_mels", type=int, default=96)
    ap.add_argument("--fmin", type=float, default=200.0)
    ap.add_argument("--fmax", type=float, default=7500.0)
    ap.add_argument(
        "--target_frames",
        type=int,
        default=128,
        help="Time-normalized frame count for averaging",
    )
    ap.add_argument(
        "--save_png",
        default="avg_mel.png",
        help="Filename to save the averaged spectrogram (placed in output/)",
    )
    args = ap.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect files
    patterns = ("**/*.ogg", "**/*.wav", "**/*.flac", "**/*.mp3")
    files = []
    for pat in patterns:
        files.extend(glob.glob(str(in_dir / pat), recursive=True))
    files = sorted(files)

    if not files:
        print(f"No audio files found under {in_dir.resolve()}")
        return

    print(f"Found {len(files)} files. Processing…")

    accum = None
    count = 0

    for i, fp in enumerate(files, 1):
        try:
            y, sr = librosa.load(fp, sr=args.sr)  # resample to common sr
            if y.size == 0:
                continue

            S = mel_power(
                y, sr, args.n_fft, args.hop, args.n_mels, args.fmin, args.fmax
            )
            # Per-file normalize in power (max=1) to avoid loudness bias
            S = S / (S.max() + 1e-12)
            # Time-normalize to fixed frames
            Sn = resize_time_axis(S, args.target_frames)

            if accum is None:
                accum = np.zeros_like(Sn, dtype=np.float64)
            accum += Sn
            count += 1

            if i % 10 == 0:
                print(f"  {i}/{len(files)} done")
        except Exception as e:
            print(f"  Skipping {fp} due to error: {e}")

    if count == 0:
        print("No valid spectrograms to average.")
        return

    avg_power = (accum / count).astype(np.float32)  # (n_mels, target_frames)
    avg_db = librosa.power_to_db(avg_power, ref=np.max)

    # Plot
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        avg_db,
        x_axis="frames",  # time is normalized frames
        y_axis="mel",
        sr=args.sr,
        hop_length=args.hop,
        fmin=args.fmin,
        fmax=args.fmax,
        cmap="magma",
    )
    cbar = plt.colorbar(format="%+2.0f dB")
    cbar.set_label("dB")
    plt.title(
        f"Averaged Mel spectrogram (N={count}, power domain avg, per-file normalized)"
    )
    plt.xlabel("Normalized time (frames)")
    plt.ylabel("Frequency (Hz)")
    plt.tight_layout()

    out_png = out_dir / args.save_png
    plt.savefig(out_png, dpi=150)
    plt.show()

    # Also save a CSV with mel bin center frequencies and average (time-mean) profile
    mel_hz = librosa.mel_frequencies(n_mels=args.n_mels, fmin=args.fmin, fmax=args.fmax)
    avg_profile = avg_power.mean(axis=1)  # (n_mels,)
    np.savetxt(
        out_dir / "avg_mel_profile.csv",
        np.column_stack([mel_hz, avg_profile]),
        delimiter=",",
        header="mel_hz,avg_power",
        comments="",
        fmt="%.6f",
    )
    print(f"Saved figure to: {out_png}")
    print(f"Saved profile CSV to: {out_dir / 'avg_mel_profile.csv'}")


if __name__ == "__main__":
    main()
