#!/usr/bin/env python3
"""
Convert a .keras model into STM32-friendly artifacts:
- SavedModel (clean)
- TFLite FP32
- TFLite dynamic-range quant
- TFLite full-integer (int8) quant (with representative dataset)
- Optional C header with int8 model bytes

Default dirs: input/ (model, optional audio) and output/ (artifacts).
"""

import argparse
import json
import os
import pathlib
import struct
import sys
from typing import Iterable, List, Optional, Tuple

import numpy as np
import tensorflow as tf

# Works with TF 2.12–2.16+. If you use a much older TF, adjust converter flags accordingly.

# ---------------------------
# Utilities
# ---------------------------


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
    """Emit a C header containing the model as a const unsigned char array."""
    lines = []
    lines.append("// Auto-generated. Embed this in your STM32 firmware.\n")
    lines.append("#pragma once\n#include <stdint.h>\n")
    lines.append(f"#define {var_name.upper()}_LEN ({len(data)})\n")
    lines.append(f"#ifdef __GNUC__\n__attribute__((aligned({align})))\n#endif\n")
    lines.append(f"const unsigned char {var_name}[] = {{\n")
    # format as 12 bytes per line
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
        ops = set()
        for d in interp._get_ops_details():  # type: ignore (private, but stable)
            ops.add(d.get("op_name", "?"))
        return sorted(ops)
    except Exception:
        return []


def guess_input_signature(
    keras_model: tf.keras.Model,
) -> Tuple[Tuple[int, ...], tf.dtypes.DType]:
    """Return (shape, dtype) for the first input (batch dimension left as None)."""
    if isinstance(keras_model.inputs, (list, tuple)):
        x = keras_model.inputs[0]
    else:
        x = keras_model.inputs
    shape = tuple([None] + list(x.shape.as_list()[1:]))
    dtype = x.dtype
    return shape, dtype


# ---------------------------
# Representative dataset
# ---------------------------


def audio_file_iter(
    audio_dir: pathlib.Path, exts=(".wav", ".ogg", ".flac", ".mp3")
) -> Iterable[pathlib.Path]:
    if not audio_dir.exists():
        return []
    for p in sorted(audio_dir.rglob("*")):
        if p.suffix.lower() in exts:
            yield p


def make_rep_generator_from_audio(
    audio_dir: pathlib.Path,
    input_shape: Tuple[int, ...],
    sample_rate: int = 16000,
    n_examples: int = 256,
):
    """
    Create a TFLite representative dataset generator using real audio.
    Assumes model expects either raw audio [batch, time] or features [batch, H, W, C].
    If features are expected, we just pass the raw-resized waveform; your model's
    own preprocessing (e.g., a tf layer) should handle transforms.
    """
    try:
        import librosa
    except Exception as e:
        print(
            f"[rep] librosa not available ({e}); falling back to synthetic calibration."
        )
        return None

    # Deduce per-sample data shape excluding batch
    per_sample = input_shape[1:]
    # Flatten time length if 1D audio
    target_len = None
    if len(per_sample) == 1:
        target_len = per_sample[0]
    elif len(per_sample) == 2:
        # e.g., [time, 1]
        target_len = per_sample[0]
    else:
        # feature-like input (e.g., [H, W, C]); we'll just feed random if no preprocessing layer
        pass

    files = list(audio_file_iter(audio_dir))
    if not files:
        print(f"[rep] No audio files found in {audio_dir}.")
        return None

    def _gen():
        cnt = 0
        for p in files:
            if cnt >= n_examples:
                break
            try:
                y, sr = librosa.load(str(p), sr=sample_rate, mono=True)
                if target_len is not None:
                    if len(y) < target_len:
                        y = np.pad(y, (0, target_len - len(y)), mode="constant")
                    else:
                        y = y[:target_len]
                    # Shape to model input (add channel dim if needed)
                    if len(per_sample) == 1:
                        sample = y.astype(np.float32)[None, :]  # [1, T]
                    elif len(per_sample) == 2 and per_sample[1] == 1:
                        sample = y.astype(np.float32)[None, :, None]  # [1, T, 1]
                    else:
                        sample = y.astype(np.float32)[None, :]  # best effort
                else:
                    # For feature-shaped inputs, hope the model includes preprocessing.
                    sample = np.random.normal(size=(1,) + per_sample).astype(np.float32)

                yield [sample]
                cnt += 1
            except Exception as e:
                print(f"[rep] skip {p.name}: {e}")
                continue
        # If we provided fewer than requested, still OK.

    return _gen


def make_rep_generator_synthetic(input_shape: Tuple[int, ...], n_examples: int = 256):
    per_sample = input_shape[1:]
    rng = np.random.default_rng(123)

    def _gen():
        for _ in range(n_examples):
            sample = rng.normal(size=(1,) + per_sample).astype(np.float32) * 0.2
            yield [sample]

    return _gen


# ---------------------------
# Converters
# ---------------------------


def convert_to_tflite(
    model: tf.keras.Model,
    rep_gen: Optional[callable],
    out_dir: pathlib.Path,
    int8_only: bool = True,
    name_prefix: str = "model",
):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Save a clean SavedModel (good for STM32 tools that accept TF models)
    savedmodel_dir = out_dir / f"{name_prefix}_savedmodel"
    if savedmodel_dir.exists():
        # overwrite cleanly
        import shutil

        shutil.rmtree(savedmodel_dir)
    tf.saved_model.save(model, str(savedmodel_dir))
    print(f"[ok] SavedModel → {savedmodel_dir}")

    # 2) TFLite float32
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = []  # no PTQ
    fp32 = conv.convert()
    save_bytes(out_dir / f"{name_prefix}_fp32.tflite", fp32)
    print(f"[ok] TFLite FP32: {human_size(len(fp32))}  ops={list_tflite_ops(fp32)}")

    # 3) TFLite dynamic-range quant (weights quantized; activations float)
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    # Keep default supported_ops set; no rep dataset needed
    drq = conv.convert()
    save_bytes(out_dir / f"{name_prefix}_drq.tflite", drq)
    print(
        f"[ok] TFLite Dynamic-Range: {human_size(len(drq))}  ops={list_tflite_ops(drq)}"
    )

    # 4) TFLite full integer (int8) with rep dataset
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    if rep_gen is not None:
        conv.representative_dataset = rep_gen
    else:
        print(
            "[warn] No representative dataset provided; full INT8 calibration may be poor."
        )

    if int8_only:
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.int8
        conv.inference_output_type = tf.int8
    else:
        # Allow fallback to float if an op lacks int8 kernel (larger model).
        conv.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS,
        ]
        conv.inference_input_type = tf.int8
        conv.inference_output_type = tf.int8

    int8 = conv.convert()
    save_bytes(out_dir / f"{name_prefix}_int8.tflite", int8)
    print(f"[ok] TFLite INT8: {human_size(len(int8))}  ops={list_tflite_ops(int8)}")

    return fp32, drq, int8, savedmodel_dir


def quick_int8_smoketest(tflite_bytes: bytes, input_shape: Tuple[int, ...]):
    try:
        interp = tf.lite.Interpreter(model_content=tflite_bytes)
        interp.allocate_tensors()
        iinfo = interp.get_input_details()[0]
        ominfo = interp.get_output_details()[0]
        # Create a zero-ish sample with proper quantization scaling
        scale, zp = iinfo["quantization"]
        if scale == 0:
            x = np.zeros(iinfo["shape"], dtype=np.int8)
        else:
            x = np.clip(np.round(0.0 / scale + zp), -128, 127).astype(
                np.int8
            ) * np.ones(iinfo["shape"], dtype=np.int8)
        interp.set_tensor(iinfo["index"], x)
        interp.invoke()
        y = interp.get_tensor(ominfo["index"])
        print(f"[ok] INT8 smoketest ran. Output shape: {y.shape}, dtype: {y.dtype}")
    except Exception as e:
        print(f"[warn] INT8 smoketest failed: {e}")


# ---------------------------
# Main
# ---------------------------


def main():
    ap = argparse.ArgumentParser(
        description="Quantize & export a .keras model for STM32"
    )
    ap.add_argument(
        "--model", type=str, default="input/model.keras", help=".keras model path"
    )
    ap.add_argument(
        "--output_dir", type=str, default="tflite_output", help="Output directory"
    )
    ap.add_argument(
        "--rep_data_dir",
        type=str,
        default="input/rep_audio",
        help="Directory with audio for calibration (optional)",
    )
    ap.add_argument(
        "--n_calib", type=int, default=256, help="Number of calibration examples"
    )
    ap.add_argument(
        "--sr", type=int, default=16000, help="Sample rate for audio calibration"
    )
    ap.add_argument(
        "--int8_only",
        action="store_true",
        help="Force all ops to int8 (best for MCUs).",
    )
    ap.add_argument(
        "--emit_header",
        action="store_true",
        help="Emit a C header with the INT8 model bytes",
    )
    ap.add_argument(
        "--header_var",
        type=str,
        default="g_model_int8",
        help="Variable name in the C header",
    )
    args = ap.parse_args()

    model_path = pathlib.Path(args.model)
    out_dir = pathlib.Path(args.output_dir)
    ensure_dir(out_dir)

    if not model_path.exists():
        print(f"ERROR: model not found: {model_path}")
        sys.exit(1)

    print(f"[load] {model_path}")
    model = tf.keras.models.load_model(str(model_path), compile=False)
    in_shape, in_dtype = guess_input_signature(model)
    print(f"[model] input_shape={in_shape}, dtype={in_dtype.name}")

    # Representative dataset
    rep_dir = pathlib.Path(args.rep_data_dir)
    rep_gen = make_rep_generator_from_audio(rep_dir, in_shape, args.sr, args.n_calib)
    if rep_gen is None:
        rep_gen = make_rep_generator_synthetic(in_shape, args.n_calib)

    # Convert
    fp32, drq, int8, savedmodel_dir = convert_to_tflite(
        model=model,
        rep_gen=rep_gen,
        out_dir=out_dir,
        int8_only=args.int8_only,
        name_prefix=model_path.stem,
    )

    # Smoketest INT8
    quick_int8_smoketest(int8, in_shape)

    # Emit header
    if args.emit_header:
        hpath = out_dir / f"{model_path.stem}_int8_model.h"
        bytes_to_c_header(int8, args.header_var, hpath)
        print(f"[ok] C header emitted: {hpath}")

    # Summary
    print("\n== Summary ==")
    print(f"SavedModel: {savedmodel_dir}")
    print(
        f"FP32 TFLite: {out_dir / (model_path.stem + '_fp32.tflite')}  ({human_size(len(fp32))})"
    )
    print(
        f"DRQ  TFLite: {out_dir / (model_path.stem + '_drq.tflite')}   ({human_size(len(drq))})"
    )
    print(
        f"INT8 TFLite: {out_dir / (model_path.stem + '_int8.tflite')}  ({human_size(len(int8))})"
    )
    if args.emit_header:
        print(f"C Header:   {out_dir / (model_path.stem + '_int8_model.h')}")
    print("Done.")


if __name__ == "__main__":
    main()
