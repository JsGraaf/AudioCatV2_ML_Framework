#!/usr/bin/env python3
import os
import re
import argparse
from math import ceil
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import librosa
from matplotlib.backends.backend_pdf import PdfPages


# =======================
# Defaults (CLI-overridable)
# =======================
DEF_SR          = 32000
DEF_NUM_MELS    = 80
DEF_NUM_FRAMES  = 283
DEF_HOP_LENGTH  = 340
DEF_N_FFT       = 1024
DEF_FMIN        = 400
DEF_FMAX        = 14999
DEF_WINDOW      = "hamming"
DEF_CENTER      = False
DEF_POWER       = 2.0
DEF_NORM        = "slaney"

# PDF layout
DEF_ROWS_PER_PAGE = 5
DEF_ROW_HEIGHT    = 2.3
DEF_FIG_WIDTH     = 12.0

# =======================
# Timestamp extraction
# Examples:
#   rec_Y2000M01D01_00h00m28s718ms.wav
#   spec_Y2000M01D01_00h00m28s718ms.txt
# =======================
STAMP_RE = re.compile(
    r"Y(?P<y>\d{4})M(?P<m>\d{2})D(?P<d>\d{2})_"
    r"(?P<H>\d{2})h(?P<M>\d{2})m(?P<S>\d{2})s(?P<ms>\d{3})m?s?",
    re.IGNORECASE
)

def extract_stamp(name: str) -> str | None:
    m = STAMP_RE.search(name)
    if not m:
        return None
    return f"Y{m['y']}M{m['m']}D{m['d']}_{m['H']}h{m['M']}m{m['S']}s{m['ms']}ms"


# =======================
# Helpers
# =======================
def sanitize_floats(x: np.ndarray) -> np.ndarray:
    if not np.isfinite(x).all():
        x = x.copy()
        x[~np.isfinite(x)] = 0.0
    return x

def enforce_exact_length(audio: np.ndarray, needed: int) -> np.ndarray:
    n = audio.size
    if n == needed:
        return audio
    out = np.zeros((needed,), dtype=audio.dtype)
    out[:min(n, needed)] = audio[:needed]
    return out

def librosa_mel_01(
    audio: np.ndarray, *, sr: int, n_fft: int, n_mels: int, hop_length: int,
    fmin: int, fmax: int, window: str, center: bool, power: float, norm: str | None
) -> np.ndarray:
    """Return mel spectrogram normalized to ~[0,1] using median-relative dB window."""
    spec = librosa.feature.melspectrogram(
        y=audio.astype(np.float32, copy=False),
        sr=int(sr),
        n_fft=int(n_fft),
        n_mels=int(n_mels),
        hop_length=int(hop_length),
        win_length=None,
        window=window,
        center=center,
        pad_mode="reflect",
        power=float(power),
        fmin=int(fmin),
        fmax=int(fmax),
        norm=norm,
    )
    spec_db = librosa.power_to_db(spec, ref=np.max)
    bg_db = np.median(spec_db, axis=1, keepdims=True)
    spec_rel = np.clip(spec_db - bg_db, -30.0, +15.0)
    return ((spec_rel + 30.0) / 45.0).astype(np.float32)


# =======================
# I/O: WAVs and device specs
# =======================
def load_wavs_by_stamp(wavs_dir: Path, *, sr: int, needed_samples: int) -> dict[str, tuple[str, np.ndarray]]:
    """Return {stamp: (display_name, mono_audio)}."""
    by_key: dict[str, tuple[str, np.ndarray]] = {}
    paths: list[str] = []
    for dp, _, fns in os.walk(wavs_dir):
        for fn in fns:
            if fn.lower().endswith(".wav"):
                paths.append(os.path.join(dp, fn))
    paths.sort()
    for p in paths:
        fn = os.path.basename(p)
        key = extract_stamp(fn)
        if not key:
            continue
        try:
            rel = os.path.relpath(p, start=wavs_dir)
        except Exception:
            rel = fn
        try:
            y, _ = librosa.load(p, sr=sr, mono=True)
        except Exception:
            y = np.zeros((needed_samples,), dtype=np.float32)
        y = sanitize_floats(y.astype(np.float32, copy=False))
        y = enforce_exact_length(y, needed_samples)
        by_key[key] = (rel, y)
    return by_key


def read_device_spec_matrix(
    txt_path: str,
    *,
    num_mels: int,
    num_frames: int,
    layout: str = "frame-major",  # "frame-major" or "mel-major"
    flip_frequency: bool = False,
) -> np.ndarray:
    """Load one spectrogram vector (float32) and reshape to (num_mels, num_frames)."""
    vec = np.fromfile(txt_path, dtype=np.float32) if os.path.getsize(txt_path) else np.zeros(0, np.float32)

    need = num_mels * num_frames
    if vec.size < need:
        vec = np.pad(vec, (0, need - vec.size))
    elif vec.size > need:
        vec = vec[:need]

    if layout == "frame-major":
        mat = vec.reshape(num_frames, num_mels).T  # (mels, frames)
    else:
        mat = vec.reshape(num_mels, num_frames)

    if flip_frequency:
        mat = np.flipud(mat)

    mat[~np.isfinite(mat)] = 0.0
    return mat.astype(np.float32, copy=False)


def load_specs_by_stamp(
    specs_dir: Path,
    *,
    num_mels: int,
    num_frames: int,
    layout: str,
    flip_frequency: bool
) -> dict[str, tuple[str, np.ndarray]]:
    """Return {stamp: (display_name, mel_matrix)}."""
    by_key: dict[str, tuple[str, np.ndarray]] = {}
    paths: list[str] = []
    for dp, _, fns in os.walk(specs_dir):
        for fn in fns:
            if fn.lower().endswith(".txt"):
                paths.append(os.path.join(dp, fn))
    paths.sort()
    for p in paths:
        fn = os.path.basename(p)
        key = extract_stamp(fn)
        if not key:
            continue
        try:
            rel = os.path.relpath(p, start=specs_dir)
        except Exception:
            rel = fn
        mat = read_device_spec_matrix(
            p,
            num_mels=num_mels,
            num_frames=num_frames,
            layout=layout,
            flip_frequency=flip_frequency,
        )
        by_key[key] = (rel, mat)
    return by_key


# =======================
# Plotting helpers
# =======================
def _axes_extent(num_frames: int, hop_length: int, sr: int, fmin: int, fmax: int):
    """extent for imshow to map x→seconds and y→Hz (linear)."""
    duration = (num_frames - 1) * hop_length / float(sr)
    return (0.0, duration, float(fmin), float(fmax))

def make_pdf(
    rows: list[tuple[str, np.ndarray, str, np.ndarray]],
    *,
    out_pdf: Path,
    rows_per_page: int,
    row_height: float,
    fig_width: float,
    num_frames: int,
    hop_length: int,
    sr: int,
    fmin: int,
    fmax: int,
) -> None:
    # shared color scale
    vals = []
    for _, lmel, _, rmel in rows:
        vals.append(lmel.ravel()); vals.append(rmel.ravel())
    all_vals = np.concatenate(vals) if vals else np.array([0.0], dtype=np.float32)
    vmin, vmax = float(np.min(all_vals)), float(np.max(all_vals))

    total_pages = ceil(len(rows) / rows_per_page)
    extent = _axes_extent(num_frames, hop_length, sr, fmin, fmax)

    with PdfPages(out_pdf.as_posix()) as pdf:
        for page in range(total_pages):
            start = page * rows_per_page
            end = min(start + rows_per_page, len(rows))
            page_rows = rows[start:end]
            n_rows = len(page_rows)

            fig_h = max(row_height * n_rows + 0.8, 6.0)
            fig, axes = plt.subplots(n_rows, 2, figsize=(fig_width, fig_h), squeeze=False)

            for r, (lname, lmel, rname, rmel) in enumerate(page_rows):
                axL = axes[r, 0]
                imL = axL.imshow(
                    lmel, aspect="auto", origin="lower",
                    cmap="magma", vmin=vmin, vmax=vmax, extent=extent
                )
                # PDF mode: include filenames in subplot titles
                axL.set_title(f"Librosa (WAV): {os.path.basename(lname)}", fontsize=9)
                axL.set_xlabel("Time (s)"); axL.set_ylabel("Frequency (Hz)")
                axL.tick_params(labelsize=8)

                axR = axes[r, 1]
                axR.imshow(
                    rmel, aspect="auto", origin="lower",
                    cmap="magma", vmin=vmin, vmax=vmax, extent=extent
                )
                axR.set_title(f"Device (SD): {os.path.basename(rname)}", fontsize=9)
                axR.set_xlabel("Time (s)"); axR.set_ylabel("Frequency (Hz)")
                axR.tick_params(labelsize=8)

            # Colorbar on the right for PDF pages
            cbar = fig.colorbar(imL, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
            cbar.set_label("Normalized dB (0..1)")

            plt.suptitle(f"Matched WAV vs Device Spectrograms — Page {page+1}/{total_pages}",
                         fontsize=12, y=0.995)
            fig.tight_layout(rect=[0, 0, 0.98, 0.98])
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

from mpl_toolkits.axes_grid1 import make_axes_locatable

def make_single_svg(
    *,
    rec_title: str,
    wav_mel: np.ndarray,
    dev_title: str,
    dev_mel: np.ndarray,
    out_svg: Path,
    num_frames: int,
    hop_length: int,
    sr: int,
    fmin: int,
    fmax: int,
) -> None:
    """Two panels with a colorbar placed OUTSIDE to the right, exported as SVG."""
    vmin = float(min(np.min(wav_mel), np.min(dev_mel)))
    vmax = float(max(np.max(wav_mel), np.max(dev_mel)))
    extent = _axes_extent(num_frames, hop_length, sr, fmin, fmax)

    # regular 1×2 layout (no cax column)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 3.6))
    fig.subplots_adjust(wspace=0.25, right=0.98, top=0.88)  # leave room for suptitle

    imL = axL.imshow(
        wav_mel, aspect="auto", origin="lower",
        cmap="magma", vmin=vmin, vmax=vmax, extent=extent
    )
    axL.set_title("Librosa (WAV)", fontsize=10)
    axL.set_xlabel("Time (s)"); axL.set_ylabel("Frequency (Hz)")

    imR = axR.imshow(
        dev_mel, aspect="auto", origin="lower",
        cmap="magma", vmin=vmin, vmax=vmax, extent=extent
    )
    axR.set_title("Device (SD)", fontsize=10)
    axR.set_xlabel("Time (s)"); axR.set_ylabel("Frequency (Hz)")

    # Create a new axes to the RIGHT of axR and put the colorbar there
    divider = make_axes_locatable(axR)
    cax = divider.append_axes("right", size="3%", pad=0.06)  # width and gap
    cbar = fig.colorbar(imR, cax=cax)
    cbar.set_label("Normalized dB (0..1)")
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle(rec_title, fontsize=12, y=0.98)
    fig.savefig(out_svg.as_posix(), format="svg", bbox_inches="tight")
    plt.close(fig)




# =======================
# CLI
# =======================
def build_argparser():
    ap = argparse.ArgumentParser(
        description="Compare WAV mel-spectrograms (librosa) vs device spectrograms."
    )
    ap.add_argument("-i", "--input", type=Path, default=Path("input"),
                    help="Root containing Recordings/ and Spectrograms/.")
    ap.add_argument("-o", "--output", type=Path, default=Path("output"),
                    help="Output directory.")
    ap.add_argument("--pdf-name", type=str, default=None,
                    help="PDF filename (for batch mode).")
    ap.add_argument("--svg-name", type=str, default=None,
                    help="SVG filename (for single-file mode).")
    ap.add_argument("--single", type=str, default=None,
                    help="Single recording filename (e.g., rec_Y...wav or spec_Y...txt). If set, exports SVG.")

    # Device serialization
    ap.add_argument("--device-layout", choices=["frame-major", "mel-major"], default="frame-major",
                    help="How device serialized the spectrogram.")
    ap.add_argument("--flip-frequency", action="store_true",
                    help="Flip frequency axis (device stored high→low).")

    # Plot layout for PDF
    ap.add_argument("--rows-per-page", type=int, default=DEF_ROWS_PER_PAGE)
    ap.add_argument("--row-height", type=float, default=DEF_ROW_HEIGHT)
    ap.add_argument("--fig-width", type=float, default=DEF_FIG_WIDTH)

    # Front-end parameters
    ap.add_argument("--sr", type=int, default=DEF_SR)
    ap.add_argument("--num-mels", type=int, default=DEF_NUM_MELS)
    ap.add_argument("--num-frames", type=int, default=DEF_NUM_FRAMES)
    ap.add_argument("--hop-length", type=int, default=DEF_HOP_LENGTH)
    ap.add_argument("--n-fft", type=int, default=DEF_N_FFT)
    ap.add_argument("--fmin", type=int, default=DEF_FMIN)
    ap.add_argument("--fmax", type=int, default=DEF_FMAX)
    ap.add_argument("--window", type=str, default=DEF_WINDOW)
    ap.add_argument("--center", action="store_true", default=DEF_CENTER)
    ap.add_argument("--power", type=float, default=DEF_POWER)
    ap.add_argument("--norm", type=str, default=DEF_NORM)

    return ap


# =======================
# Main
# =======================
def main():
    args = build_argparser().parse_args()

    sd_root  = args.input
    wavs_dir = sd_root / "Recordings"
    specs_dir= sd_root / "Spectrograms"
    if not wavs_dir.exists() or not specs_dir.exists():
        raise SystemExit(f"Expected '{wavs_dir}' and '{specs_dir}' under --input '{sd_root}'")

    needed_samples = (args.num_frames - 1) * args.hop_length + args.n_fft

    # Load index
    rec_by_key = load_wavs_by_stamp(wavs_dir, sr=args.sr, needed_samples=needed_samples)
    spec_by_key= load_specs_by_stamp(
        specs_dir,
        num_mels=args.num_mels,
        num_frames=args.num_frames,
        layout=args.device_layout,
        flip_frequency=args.flip_frequency
    )

    args.output.mkdir(parents=True, exist_ok=True)

    # ---------- Single-file mode ----------
    if args.single:
        stamp = extract_stamp(args.single)
        if not stamp:
            raise SystemExit(f"Could not extract stamp from --single '{args.single}'")

        if stamp not in rec_by_key:
            raise SystemExit(f"No WAV found for stamp '{stamp}'")
        if stamp not in spec_by_key:
            raise SystemExit(f"No device spectrogram found for stamp '{stamp}'")

        wav_name, wav_audio = rec_by_key[stamp]
        spec_name, spec_mat = spec_by_key[stamp]

        wav_mel = librosa_mel_01(
            wav_audio,
            sr=args.sr, n_fft=args.n_fft, n_mels=args.num_mels,
            hop_length=args.hop_length, fmin=args.fmin, fmax=args.fmax,
            window=args.window, center=args.center, power=args.power, norm=args.norm
        )

        svg_name = args.svg_name or f"{stamp}.svg"
        out_svg = args.output / svg_name

        # Title = recording filename (relative path)
        make_single_svg(
            rec_title=wav_name,
            wav_mel=wav_mel,
            dev_title=spec_name,
            dev_mel=spec_mat,
            out_svg=out_svg,
            num_frames=args.num_frames,
            hop_length=args.hop_length,
            sr=args.sr,
            fmin=args.fmin,
            fmax=args.fmax,
        )
        print(f"Exported SVG: {out_svg}")
        return

    # ---------- Batch PDF mode ----------
    common = sorted(set(rec_by_key) & set(spec_by_key))
    if not common:
        print(f"No matches. WAV keys: {len(rec_by_key)}, SPEC keys: {len(spec_by_key)}")
        return

    rows: list[tuple[str, np.ndarray, str, np.ndarray]] = []
    for k in common:
        wav_name, wav_audio = rec_by_key[k]
        spec_name, spec_mat = spec_by_key[k]
        wav_mel = librosa_mel_01(
            wav_audio,
            sr=args.sr, n_fft=args.n_fft, n_mels=args.num_mels,
            hop_length=args.hop_length, fmin=args.fmin, fmax=args.fmax,
            window=args.window, center=args.center, power=args.power, norm=args.norm
        )
        rows.append((wav_name, wav_mel, spec_name, spec_mat))

    pdf_name = args.pdf_name or f"compare_matched_wav_vs_device_{sd_root.name}.pdf"
    out_pdf = args.output / pdf_name

    make_pdf(
        rows,
        out_pdf=out_pdf,
        rows_per_page=args.rows_per_page,
        row_height=args.row_height,
        fig_width=args.fig_width,
        num_frames=args.num_frames,
        hop_length=args.hop_length,
        sr=args.sr,
        fmin=args.fmin,
        fmax=args.fmax,
    )
    print(f"Matched pairs: {len(rows)}  →  PDF: {out_pdf}")


if __name__ == "__main__":
    main()
