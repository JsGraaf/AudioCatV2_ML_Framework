#!/usr/bin/env python3
import os
from pathlib import Path
import numpy as np

try:
    import soundfile as sf  # preferred
except Exception as e:
    raise SystemExit("Please `pip install soundfile` to read WAV files.") from e


def segment_power(x: np.ndarray) -> float:
    # Mean-square (power) of a mono float segment
    x = np.asarray(x, dtype=np.float32)
    return float(np.mean(x * x)) if x.size else 0.0


def baseline_decision(x: np.ndarray, t_low: float, t_high: float) -> str:
    P = segment_power(x)
    if P < t_low:
        return "discard"      # stay idle
    if P >= t_high:
        return "classify"     # run TinyChirp
    return "store"            # optional: log/skip-ML


def windows(sig: np.ndarray, sr: int, win_sec=3.032, hop_sec=None):
    """Yield (start_idx, end_idx) for fixed windows."""
    if hop_sec is None:
        hop_sec = win_sec  # no overlap by default
    win = int(round(win_sec * sr))
    hop = int(round(hop_sec * sr))
    if win <= 0:
        return
    n = len(sig)
    if n == 0:
        return
    i = 0
    while i < n:
        j = min(i + win, n)
        yield i, j
        if j == n:
            break
        i += hop


def load_mono(path: Path):
    y, sr = sf.read(path, always_2d=False)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    return y.astype(np.float32, copy=False), int(sr)


def main():
    root = Path("output")
    t_low = 9.9e-01
    t_high = 9.9e-01
    win_sec = 3.032
    hop_sec = 3.032  # set to 1.516 for 50% overlap

    wavs = sorted(p for p in root.glob("*.wav"))
    if not wavs:
        print("[warn] no .wav files in output/")
        return
    
    file_decisions = []
    for p in wavs:
        y, sr = load_mono(p)

        for s, e in windows(y, sr, win_sec=win_sec, hop_sec=hop_sec):
            seg = y[s:e]
            # pad last short window to full size for consistent power if you want:
            if len(seg) < int(round(win_sec * sr)):
                seg = np.pad(seg, (0, int(round(win_sec * sr)) - len(seg)))
            decision = baseline_decision(seg, t_low=t_low, t_high=t_high)
            P = segment_power(seg)
            start_t = s / sr
            end_t = e / sr
            print(f"{p.name} [{start_t:7.3f}-{end_t:7.3f}s]  P={P:.6e}  -> {decision}")
            file_decisions.append(decision)

    # quick summary
    n_cls = sum(d == "classify" for d in file_decisions)
    n_store = sum(d == "store" for d in file_decisions)
    n_disc = sum(d == "discard" for d in file_decisions)
    print(n_cls, n_store, n_disc)
    # print(f"-> {p.name}: classify={n_cls}, store={n_store}, discard={n_disc}\n")


if __name__ == "__main__":
    main()
