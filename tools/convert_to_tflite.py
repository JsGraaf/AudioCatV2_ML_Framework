#!/usr/bin/env python3
"""
Export a model to INT8 TFLite using a representative dataset sampled
BALANCED across species folders:

data_root/
  Species_A/
    a.ogg b.wav ...
  Species_B/
    c.flac ...

Usage example:
  python3 export_int8_from_species.py \
    --model input/model.keras \
    --data_root datasets/custom_set/xc_dataset \
    --per_species 40 \
    --output_dir tflite_out \
    --sr 16000 \
    --emit_header
"""

import argparse
import os
import pathlib
import sys
from typing import Iterable, List, Optional, Tuple

import numpy as np
import tensorflow as tf

AUDIO_EXTS = {".wav", ".ogg", ".flac", ".mp3", ".m4a", ".aac", ".wma"}

# ───────────── Utils ─────────────


def human_size(nbytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if nbytes < 1024.0:
            return f"{nbytes:.1f}{unit}"
        nbytes /= 1024.0
    return f"{nbytes:.1f}PB"


def ensure_dir(p: pathlib.Path):
    p.mkdir(parents=True, exist_ok=True)


def save_bytes(p: pathlib.Path, data: bytes):
    with open(p, "wb") as f:
        f.write(data)


def bytes_to_c_header(
    data: bytes, var_name: str, out_path: pathlib.Path, align: int = 8
):
    lines = []
    lines.append("// Auto-generated. Embed this in your STM32 firmware.\n")
    lines.append("#pragma once\n#include <stdint.h>\n")
    lines.append(f"#define {var_name.upper()}_LEN ({len(data)})\n")
    lines.append(f"#ifdef __GNUC__\n__attribute__((aligned({align})))\n#endif\n")
    lines.append(f"const unsigned char {var_name}[] = {{\n")
    for i in range(0, len(data), 12):
        chunk = data[i : i + 12]
        lines.append("  " + ", ".join(f"0x{b:02x}" for b in chunk) + ",\n")
    lines.append("};\n")
    out_path.write_text("".join(lines))
    return out_path


def list_tflite_ops(tflite_model: bytes) -> List[str]:
    try:
        interp = tf.lite.Interpreter(model_content=tflite_model)
        interp.allocate_tensors()
        ops = {d.get("op_name", "?") for d in interp._get_ops_details()}  # type: ignore
        return sorted(ops)
    except Exception:
        return []


def is_savedmodel_dir(p: pathlib.Path) -> bool:
    return p.is_dir() and (p / "saved_model.pb").exists()


def guess_input_signature_keras(
    model: tf.keras.Model,
) -> Tuple[Tuple[int, ...], tf.dtypes.DType]:
    x = model.inputs[0] if isinstance(model.inputs, (list, tuple)) else model.inputs
    shape = tuple([None] + list(x.shape.as_list()[1:]))
    dtype = x.dtype
    return shape, dtype


# ───────────── Species data helpers ─────────────


def find_species_dirs(root: pathlib.Path) -> List[pathlib.Path]:
    return [p for p in sorted(root.iterdir()) if p.is_dir()]


def list_audio_files(d: pathlib.Path) -> List[pathlib.Path]:
    return [
        p
        for p in sorted(d.rglob("*"))
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS
    ]


def balanced_sample_across_species(
    data_root: pathlib.Path, per_species: int, seed: int = 42
) -> List[pathlib.Path]:
    import random

    rng = random.Random(seed)
    selected: List[pathlib.Path] = []
    for sp_dir in find_species_dirs(data_root):
        files = list_audio_files(sp_dir)
        if not files:
            continue
        rng.shuffle(files)
        selected.extend(files[:per_species])
    if not selected:
        raise SystemExit(f"No audio found under {data_root}")
    return selected


# ───────────── Representative dataset ─────────────


def make_rep_from_species_dirs(
    data_root: pathlib.Path,
    input_shape: Tuple[int, ...],
    sample_rate: int,
    per_species: int,
    seed: int = 42,
):
    """
    Build a representative dataset generator using balanced sampling from species folders.
    For 1-D audio inputs ([T] or [T,1]), crops/pads waveform to T.
    For feature-shaped inputs (e.g., [H,W,C]) without preprocessing layers inside the model,
    calibration still proceeds but may be less accurate (random normal fallback).
    """
    try:
        import librosa
    except Exception as e:
        print(f"[rep] librosa not available ({e}); using synthetic reps.")
        return None

    per_sample = input_shape[1:]
    # detect 1-D audio shapes
    target_len = None
    is_1d = False
    if len(per_sample) == 1:
        target_len = int(per_sample[0])
        is_1d = True
    elif len(per_sample) == 2 and per_sample[1] == 1:
        target_len = int(per_sample[0])
        is_1d = True

    files = balanced_sample_across_species(data_root, per_species, seed=seed)

    def _gen():
        import random

        rng = random.Random(seed)
        rng.shuffle(files)
        for p in files:
            try:
                if is_1d and target_len is not None:
                    y, sr = librosa.load(str(p), sr=sample_rate, mono=True)
                    if len(y) >= target_len:
                        off = (
                            0
                            if len(y) == target_len
                            else rng.randint(0, len(y) - target_len)
                        )
                        y = y[off : off + target_len]
                    else:
                        y = np.pad(y, (0, target_len - len(y)), mode="constant")
                    if len(per_sample) == 1:
                        sample = y.astype(np.float32)[None, :]  # [1, T]
                    else:
                        sample = y.astype(np.float32)[None, :, None]  # [1, T, 1]
                else:
                    # Feature-shaped fallback (encourage adding preproc layers in Keras for best results)
                    sample = (
                        np.random.normal(size=(1,) + per_sample).astype(np.float32)
                        * 0.2
                    )
                yield [sample]
            except Exception as e:
                print(f"[rep] skip {p.name}: {e}")
                continue

    return _gen


def make_rep_synthetic(input_shape: Tuple[int, ...], n_examples: int = 512):
    per_sample = input_shape[1:]
    rng = np.random.default_rng(123)

    def _gen():
        for _ in range(n_examples):
            yield [rng.normal(size=(1,) + per_sample).astype(np.float32) * 0.2]

    return _gen


# ───────────── INT8 conversion ─────────────


def convert_to_int8(
    keras_or_savedmodel: pathlib.Path,
    rep_gen,
    out_dir: pathlib.Path,
    name_prefix: str,
) -> bytes:
    if is_savedmodel_dir(keras_or_savedmodel):
        conv = tf.lite.TFLiteConverter.from_saved_model(str(keras_or_savedmodel))
    else:
        model = tf.keras.models.load_model(str(keras_or_savedmodel), compile=False)
        conv = tf.lite.TFLiteConverter.from_keras_model(model)

    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    if rep_gen is not None:
        conv.representative_dataset = rep_gen
        print("[rep] Using representative dataset.")
    else:
        print("[rep][warn] No representative dataset; calibration may be poor.")

    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.float32

    tfl = conv.convert()
    out_path = out_dir / f"{name_prefix}_int8.tflite"
    save_bytes(out_path, tfl)
    print(
        f"[ok] INT8 saved: {out_path}  size={human_size(len(tfl))}  ops={list_tflite_ops(tfl)}"
    )
    return tfl


def quick_smoketest_int8(tflite_bytes: bytes):
    try:
        interp = tf.lite.Interpreter(model_content=tflite_bytes)
        interp.allocate_tensors()
        iinfo = interp.get_input_details()[0]
        oinfo = interp.get_output_details()[0]
        sc, zp = iinfo["quantization"]
        q0 = int(np.clip(np.round(0.0 / (sc or 1.0) + zp), -128, 127))
        x = np.ones(iinfo["shape"], dtype=np.int8) * q0
        interp.set_tensor(iinfo["index"], x)
        interp.invoke()
        y = interp.get_tensor(oinfo["index"])
        print(f"[smoke] ok. out shape={y.shape}, dtype={y.dtype}")
    except Exception as e:
        print(f"[smoke][warn] failed: {e}")


# ───────────── Main ─────────────


def main():
    ap = argparse.ArgumentParser(
        description="INT8 TFLite converter with species/audio representative dataset."
    )
    ap.add_argument(
        "--model",
        required=True,
        help="Path to .keras/.h5 model OR a SavedModel directory (contains saved_model.pb)",
    )
    ap.add_argument(
        "--data_root",
        required=True,
        help="Root folder containing species subfolders with audio (e.g., datasets/custom_set/xc_dataset)",
    )
    ap.add_argument(
        "--per_species",
        type=int,
        default=100,
        help="Number of audio files to sample per species (balanced).",
    )
    ap.add_argument(
        "--sr",
        type=int,
        default=32000,
        help="Sample rate for loading audio during calibration.",
    )
    ap.add_argument(
        "--seed", type=int, default=42, help="Random seed for sampling/crops."
    )
    ap.add_argument(
        "--output_dir", type=str, default="tflite_out", help="Output directory."
    )
    ap.add_argument(
        "--emit_header",
        action="store_true",
        help="Emit a C header with INT8 model bytes.",
    )
    ap.add_argument(
        "--header_var",
        type=str,
        default="g_model_int8",
        help="C variable name in header.",
    )
    args = ap.parse_args()

    model_path = pathlib.Path(args.model)
    data_root = pathlib.Path(args.data_root)
    out_dir = pathlib.Path(args.output_dir)
    ensure_dir(out_dir)

    if not model_path.exists():
        print(f"ERROR: model not found: {model_path}")
        sys.exit(1)
    if not data_root.exists():
        print(f"ERROR: data_root not found: {data_root}")
        sys.exit(1)

    # Try to load Keras to infer input shape (best case). If SavedModel only, try Keras load; else fallback.
    input_shape = None
    if is_savedmodel_dir(model_path):
        try:
            km = tf.keras.models.load_model(str(model_path), compile=False)
            input_shape, _ = guess_input_signature_keras(km)
        except Exception as e:
            print(f"[info] SavedModel loaded without Keras inference: {e}")
    else:
        km = tf.keras.models.load_model(str(model_path), compile=False)
        print(km.summary())
        print(km.layers[-1].name, km.layers[-1].activation)  # should be 'linear'
        input_shape, _ = guess_input_signature_keras(km)

    # Build rep dataset generator from species folders
    if input_shape is not None:
        rep_gen = make_rep_from_species_dirs(
            data_root=data_root,
            input_shape=input_shape,
            sample_rate=args.sr,
            per_species=args.per_species,
            seed=args.seed,
        )
        if rep_gen is None:
            print("[rep] Falling back to synthetic representative dataset.")
            rep_gen = make_rep_synthetic(
                input_shape, n_examples=max(256, args.per_species)
            )
    else:
        # Conservative fallback if we couldn't infer shape
        print(
            "[rep][warn] Could not infer input shape; falling back to (None,16000,1)."
        )
        input_shape = (None, 16000, 1)
        rep_gen = make_rep_synthetic(input_shape, n_examples=max(256, args.per_species))

    # Convert
    name_prefix = model_path.stem if model_path.is_file() else model_path.name
    tfl = convert_to_int8(model_path, rep_gen, out_dir, name_prefix)

    # Smoketest
    quick_smoketest_int8(tfl)

    # Optional header
    if args.emit_header:
        hpath = out_dir / f"{name_prefix}_int8_model.h"
        bytes_to_c_header(tfl, args.header_var, hpath)
        print(f"[ok] Header: {hpath}")

    print("\n== Summary ==")
    print(
        f"INT8 TFLite: {out_dir / (name_prefix + '_int8.tflite')} ({human_size(len(tfl))})"
    )
    if args.emit_header:
        print(f"C Header:   {out_dir / (name_prefix + '_int8_model.h')}")
    print("Done.")


if __name__ == "__main__":
    main()
