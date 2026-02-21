#!/usr/bin/env python3
"""
Convert raw binary float files ('.txt' extension) into WAVs, splitting on 3.032s,
then read MCU predictions stored as a binary file with struct '<96sf' from
input/Predictions/ and link them to the generated chunks.

Outputs:
  - WAV chunks in output/
  - output/chunks_with_predictions.csv
"""

import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Iterable

import numpy as np
import pandas as pd
import struct
import os
import wave

# Preferred writer
try:
    import soundfile as sf
    _HAVE_SF = True
except Exception:
    _HAVE_SF = False

# Fallbacks
try:
    from scipy.io import wavfile
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

# ─────────────────────────────────────────────────────────────────────────

TYPICAL_SRS = [8000, 12000, 16000, 22050, 24000, 32000, 44100, 48000]
REC = struct.Struct("<96sf")  # little-endian: 96-byte path + float32 decision


def parse_args():
    p = argparse.ArgumentParser(description="Convert binary float .txt files to WAV and link with binary predictions.")
    p.add_argument("--input", default="input", help="Input directory (default: input)")
    p.add_argument("--output", default="output", help="Output directory (default: output)")
    p.add_argument("--pattern", default="*.txt", help="Glob pattern to match (default: *.txt)")
    p.add_argument("--dtype", default="float32", choices=["float32", "float64"], help="Sample dtype inside files")
    p.add_argument("--little_endian", action="store_true", default=True, help="Treat data as little-endian")
    p.add_argument("--big_endian", action="store_true", help="Treat data as big-endian (overrides little_endian)")
    p.add_argument("--channels", type=int, default=1, help="Number of channels (default: 1)")
    p.add_argument("--sr", default="auto", help='Sample rate (e.g., "16000") or "auto" (default)')
    p.add_argument("--durations", default="3.032,6.064,9.096,12.128", help="Allowed durations when --sr auto")
    p.add_argument("--split_on", type=float, default=3.032, help="Chunk length in seconds (default: 3.032)")
    p.add_argument("--subtype", default="PCM_16", choices=["PCM_16", "PCM_24", "PCM_32", "FLOAT"], help="WAV encoding")
    p.add_argument("--normalize", action="store_true", help="Normalize each chunk to |max|=1.0")
    p.add_argument("--gain_db", type=float, default=0.0, help="Gain in dB after normalization")
    p.add_argument("--strict", action="store_true", help="Error on mismatches; otherwise warn & continue")
    p.add_argument("--rec_dir", default="Recordings", help="Folder with input binary float files")
    p.add_argument("--pred_dir", default="Predictions", help="Folder with binary prediction files")
    p.add_argument("--pred_glob", default="*.bin,*.dat,*.txt", help="Globs (comma-separated) for prediction files")
    return p.parse_args()


def _np_dtype(dtype: str, big_endian: bool) -> np.dtype:
    kind = {"float32": "f4", "float64": "f8"}[dtype]
    endian = ">" if big_endian else "<"
    return np.dtype(endian + kind)


def _infer_sr_from_len(n_samples: int, channels: int, allowed_durs: List[float]) -> Optional[int]:
    if channels <= 0:
        return None
    frames = n_samples // channels
    if frames * channels != n_samples:
        return None
    candidates = []
    for dur in allowed_durs:
        if dur <= 0:
            continue
        sr_f = frames / dur
        sr_candidate = int(round(sr_f))
        if abs(sr_candidate * dur - frames) < 1e-6:
            candidates.append(sr_candidate)
    if not candidates:
        return None
    for c in candidates:
        if c in TYPICAL_SRS:
            return c
    best = min(TYPICAL_SRS, key=lambda t: min(abs(c - t) for c in candidates))
    return best


def _apply_gain_and_normalize(x: np.ndarray, normalize: bool, gain_db: float) -> np.ndarray:
    y = x.astype(np.float32, copy=True)
    if normalize and y.size:
        peak = np.max(np.abs(y))
        if peak > 0:
            y /= peak
    if gain_db:
        y *= 10.0 ** (gain_db / 20.0)
    return y


def _write_wav(out_path: Path, sr: int, data_f32: np.ndarray, subtype: str):
    sf.write(out_path.as_posix(), data_f32, sr, subtype=subtype)
    return


# ── Predictions reader (binary) ─────────────────────────────────────────

REC = struct.Struct("<96sf")  # unchanged

def iter_predictions_with_idx(bin_path: Path):
    """
    Yield (orig_base, chunk_idx, score) from the binary predictions file.
    Duplicates per base are assigned chunk_idx = 1, 2, ... in read order.
    Examples in file:
      000914.txt 0.30
      000914.txt 0.55
    -> yields ('000914', 1, 0.30) and ('000914', 2, 0.55)
    """
    counts = {}  # base -> next index to assign
    with open(bin_path, "rb") as f:
        while True:
            chunk = f.read(REC.size)
            if not chunk:
                break
            if len(chunk) < REC.size:
                break  # ignore partial tail
            raw_path, decision = REC.unpack(chunk)
            path_str = raw_path.split(b"\x00", 1)[0].decode("utf-8", errors="ignore")
            base = Path(path_str).stem  # "000914.txt" -> "000914"
            counts[base] = counts.get(base, 0) + 1
            yield base, counts[base], float(decision)


def load_all_binary_predictions(pred_dir: Path, patterns: str) -> pd.DataFrame:
    """
    Return DF with columns: ['orig_base','chunk_idx','score','__pred_source'].
    """
    if not pred_dir.exists():
        return pd.DataFrame(columns=["orig_base", "chunk_idx", "score", "__pred_source"])

    files: List[Path] = []
    for pat in [p.strip() for p in patterns.split(",") if p.strip()]:
        files.extend(sorted(pred_dir.rglob(pat)))
    if not files:
        return pd.DataFrame(columns=["orig_base", "chunk_idx", "score", "__pred_source"])

    rows = []
    for p in files:
        try:
            for base, idx, score in iter_predictions_with_idx(p):
                rows.append({
                    "orig_base": base,
                    "chunk_idx": int(idx),
                    "score": float(score),
                    "__pred_source": p.name,
                })
        except Exception as e:
            print(f"[warn] Failed reading predictions from {p.name}: {e}")
    return pd.DataFrame(rows)


# ── Main ────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    in_dir = Path(os.path.join(args.input, args.rec_dir))
    out_dir = Path(args.output)
    pred_dir = Path(os.path.join(args.input, args.pred_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    big_endian = True if args.big_endian else False
    dtype = _np_dtype(args.dtype, big_endian)
    allowed_durs = [float(x.strip()) for x in args.durations.split(",") if x.strip()]

    files = sorted(in_dir.rglob(args.pattern))
    sr_auto = (str(args.sr).lower() == "auto")
    fixed_sr: Optional[int] = None if sr_auto else int(args.sr)

    print(f"[info] Writer: {'soundfile' if _HAVE_SF else ('scipy' if _HAVE_SCIPY else 'wave stdlib')}")
    print(f"[info] dtype={dtype}, channels={args.channels}, {'big' if big_endian else 'little'}-endian")
    print(f"[info] Split on {args.split_on}s ({'disabled' if args.split_on <= 0 else 'enabled'})")

    # Collect metadata of produced chunks
    chunk_rows: List[dict] = []
    n_ok = 0
    n_skip = 0

    for f in files:
        try:
            raw = np.fromfile(f, dtype=dtype)
            if raw.size == 0:
                raise ValueError("Empty file or dtype mismatch")

            data = raw

            # sample rate
            if sr_auto:
                sr = _infer_sr_from_len(raw.size, args.channels, allowed_durs)
                if sr is None:
                    raise ValueError(f"Could not infer sample rate from length {raw.size} and durations {allowed_durs}.")
            else:
                sr = fixed_sr

            frames_total = data.shape[0] if data.ndim > 1 else data.size
            dur = frames_total / float(sr)
            if allowed_durs and not any(abs(dur - d) < 1e-3 for d in allowed_durs):
                msg = (f"Duration {dur:.6f}s not in allowed {allowed_durs} for {f.name} (sr={sr}, frames={frames_total})")
                if args.strict:
                    raise ValueError(msg)
                else:
                    print(f"[warn] {msg}")

            # split
            if args.split_on and args.split_on > 0:
                frames_per_chunk = int(round(args.split_on * sr))
                n_chunks = max(1, int(np.ceil(frames_total / frames_per_chunk)))
                for k in range(n_chunks):
                    s = k * frames_per_chunk
                    e = min(s + frames_per_chunk, frames_total)
                    if e <= s:
                        continue
                    chunk = data[s:e] if data.ndim == 1 else data[s:e, :]
                    if chunk.shape[0] < frames_per_chunk:
                        print(f"[warn] {f.name}: last chunk truncated ({chunk.shape[0]}/{frames_per_chunk} frames)")
                        if args.strict:
                            raise ValueError("Truncated last chunk under --strict.")

                    # chunk = _apply_gain_and_normalize(chunk, normalize=args.normalize, gain_db=args.gain_db)
                    chunk = chunk - np.mean(chunk)  # DC removal
                    chunk = _apply_gain_and_normalize(chunk, normalize=args.normalize, gain_db=args.gain_db)
                    
                    out_wav = out_dir / f"{f.stem}_p{k+1:02d}.wav"
                    _write_wav(out_wav, sr, chunk, args.subtype)

                    chunk_rows.append({
                        "orig_txt": f.name,
                        "orig_base": Path(f.name).stem,
                        "chunk_idx": k + 1,
                        "chunk_file": out_wav.name,
                        "sr": sr,
                        "frames_total": frames_total,
                        "frames_per_chunk": frames_per_chunk,
                        "chunk_start_frame": s,
                        "chunk_end_frame": e,
                        "chunk_start_sec": s / sr,
                        "chunk_end_sec": e / sr,
                        "chunk_duration_sec": (e - s) / sr,
                    })
                print(f"[ok] {f.name} -> {n_chunks} chunk(s) @ sr={sr} | dur≈{dur:.3f}s")
                n_ok += 1
            else:
                data = _apply_gain_and_normalize(data, normalize=args.normalize, gain_db=args.gain_db)
                out_path = out_dir / (f.stem + ".wav")
                _write_wav(out_path, sr, data, args.subtype)
                chunk_rows.append({
                    "orig_txt": f.name,
                    "orig_base": Path(f.name).stem,
                    "chunk_idx": 1,
                    "chunk_file": out_path.name,
                    "sr": sr,
                    "frames_total": frames_total,
                    "frames_per_chunk": frames_total,
                    "chunk_start_frame": 0,
                    "chunk_end_frame": frames_total,
                    "chunk_start_sec": 0.0,
                    "chunk_end_sec": dur,
                    "chunk_duration_sec": dur,
                })
                print(f"[ok] {f.name} -> {out_path.name} | sr={sr} | dur≈{dur:.3f}s")
                n_ok += 1

        except Exception as e:
            print(f"[skip] {f.name}: {e}")
            n_skip += 1

    print(f"\n[convert] Wrote {n_ok} file(s); skipped {n_skip}. Output WAVs in: {out_dir}")

    # Build chunk dataframe
    chunks_df = pd.DataFrame(chunk_rows)
    if chunks_df.empty:
        print("[info] No chunks indexed; nothing to merge.")
        return

    # Load binary predictions and merge
    preds_df = load_all_binary_predictions(pred_dir, args.pred_glob)
    if preds_df.empty:
        print(f"[merge] No binary prediction files found in {pred_dir}. Writing chunk index only.")
        merged = chunks_df.copy()
    else:
        print(f"[merge] Loaded {len(preds_df)} prediction rows from {pred_dir}")
        merged = chunks_df.merge(
            preds_df, how="left",
            left_on=["orig_base", "chunk_idx"],
            right_on=["orig_base", "chunk_idx"]
        )

    # Save dataframe
    out_csv = out_dir / "chunks_with_predictions.csv"
    merged.to_csv(out_csv, index=False)
    print(f"[done] Saved dataframe:\n  - {out_csv}")


if __name__ == "__main__":
    main()
