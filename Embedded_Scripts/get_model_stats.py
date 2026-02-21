#!/usr/bin/env python3
"""
analyze_model.py

Usage:
  python analyze_model.py -t input/YourRunDir

The target directory should contain:
  - a .keras model file (preferred), OR
  - a weights file (e.g., weights.h5) plus a YAML config that can build the model

YAML expectations (flexible):
  - model.name: "MiniResNet" (optional; only needed if you want to (re)build)
  - model.num_blocks: int
  - model.base_channels: int
  - model.num_classes: int
  - model.input_shape: [H, W, C]  # preferred; else inferred from model file
  - features.n_mels / features.num_frames can be used to compose input_shape if needed
"""

import os, sys, argparse, glob, math, io, time, json
from pathlib import Path
import numpy as np

sys.path.append("../models")


# Keras / TF
import tensorflow as tf
from tensorflow import keras

# YAML
try:
    import yaml
except ImportError:
    print("Please `pip install pyyaml`")
    sys.exit(1)


# --------------------------
# Utilities
# --------------------------
def human_bytes(n: int) -> str:
    u = ["B", "KB", "MB", "GB", "TB", "PB"]
    i = 0
    x = float(n)
    while x >= 1024 and i < len(u) - 1:
        x /= 1024.0
        i += 1
    return f"{x:.2f} {u[i]}"


def bytes_per_dtype(dtype_str: str) -> int:
    return np.dtype(dtype_str).itemsize


def find_first(patterns, base: Path):
    for pat in patterns:
        hit = list(base.glob(pat))
        if hit:
            return hit[0]
    return None


def read_yaml_config(tdir: Path):
    cfg_path = find_first(["config.yaml", "config.yml", "*.yaml", "*.yml"], tdir)
    if not cfg_path:
        return {}
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f) or {}


def estimate_activation_bytes(model: keras.Model, batch_size=1, dtype="float32") -> int:
    """Rough upper bound: sum layer outputs (batch=1)."""
    total = 0
    b = bytes_per_dtype(dtype)
    for layer in model.layers:
        out_shape = getattr(layer, "output_shape", None)
        if out_shape is None:
            continue
        shapes = (
            out_shape
            if isinstance(out_shape, (list, tuple))
            and out_shape
            and isinstance(out_shape[0], (list, tuple))
            else [out_shape]
        )
        for s in shapes:
            if s is None:
                continue
            if isinstance(s, tuple):
                dims = [
                    batch_size if d is None else int(d)
                    for d in s
                    if d is None or (isinstance(d, (int, np.integer)) and d > 0)
                ]
                if not dims:
                    continue
                numel = 1
                for d in dims:
                    numel *= d
                total += numel * b
    return int(total)


def tflite_sizes(model: keras.Model, make_int8=True):
    sizes = {}
    # FP32
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    tfl = conv.convert()
    sizes["tflite_fp32_bytes"] = len(tfl)

    if make_int8:
        conv = tf.lite.TFLiteConverter.from_keras_model(model)
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        # NOTE: without representative dataset, this is weight-quantized; activations may remain float.
        tfl_i8 = conv.convert()
        sizes["tflite_int8_wo_calib_bytes"] = len(tfl_i8)
    return sizes


# --------------------------
# Minimal MiniResNet builder (only if you *need* to rebuild)
# --------------------------
def conv3x3(ch, stride=1):
    return keras.Sequential(
        [
            keras.layers.Conv2D(ch, 3, strides=stride, padding="same", use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
        ]
    )


class BasicBlock(keras.layers.Layer):
    def __init__(self, out_ch, stride=1):
        super().__init__()
        self.stride = stride
        self.conv1 = keras.layers.Conv2D(
            out_ch, 3, strides=stride, padding="same", use_bias=False
        )
        self.bn1 = keras.layers.BatchNormalization()
        self.relu = keras.layers.ReLU()
        self.conv2 = keras.layers.Conv2D(
            out_ch, 3, strides=1, padding="same", use_bias=False
        )
        self.bn2 = keras.layers.BatchNormalization()
        self.proj = None

    def build(self, input_shape):
        in_ch = int(input_shape[-1])
        if self.stride != 1 or in_ch != int(self.conv1.filters):
            self.proj = keras.Sequential(
                [
                    keras.layers.Conv2D(
                        self.conv1.filters, 1, strides=self.stride, use_bias=False
                    ),
                    keras.layers.BatchNormalization(),
                ]
            )

    def call(self, x, training=False):
        identity = x
        out = self.conv1(x, training=training)
        out = self.bn1(out, training=training)
        out = self.relu(out)
        out = self.conv2(out, training=training)
        out = self.bn2(out, training=training)
        if self.proj is not None:
            identity = self.proj(identity, training=training)
        out = keras.layers.add([out, identity])
        return self.relu(out)


# --- NEW: manual MACs fallback ----------------------------------------------
def _prod(shape):
    p = 1
    for d in shape:
        p *= int(d)
    return p


def _layer_macs(layer):
    """
    Returns MACs for common layers. Ignores BN/Activations/Pooling (negligible or non-MAC).
    Works on built model with known output shapes (batch dim may be None).
    """
    import tensorflow as tf
    from tensorflow import keras

    name = layer.__class__.__name__
    cfg = layer.get_config()

    # Helper to get HW(C) without batch
    def out_hw_c():
        s = layer.output_shape
        if isinstance(s, (list, tuple)) and s and isinstance(s[0], (list, tuple)):
            s = s[0]
        if s is None:
            return None
        # Expect (B,H,W,C) for 2D convs
        if len(s) == 4:
            _, H, W, C = s
            return int(H), int(W), int(C)
        if len(s) == 3:  # 1D conv: (B,T,C)
            _, T, C = s
            return int(T), 1, int(C)
        if len(s) == 5:  # 3D conv: (B,D,H,W,C) -> treat spatial = D*H*W
            _, D, H, W, C = s
            return int(D) * int(H) * int(W), 1, int(C)
        return None

    def in_c():
        s = layer.input_shape
        if isinstance(s, (list, tuple)) and s and isinstance(s[0], (list, tuple)):
            s = s[0]
        if s is None:
            return None
        return int(s[-1])

    # Conv2D / Conv1D / Conv3D (handled through shapes)
    if name in ("Conv2D", "Conv1D", "Conv3D"):
        hwc = out_hw_c()
        cin = in_c()
        if not hwc or cin is None:
            return 0
        H, W, cout = hwc
        k = cfg.get("kernel_size")
        if isinstance(k, int):
            kh, kw = k, (k if name != "Conv1D" else 1)
        elif isinstance(k, (list, tuple)):
            if name == "Conv1D":
                kh, kw = k[0], 1
            elif name == "Conv3D":
                # kernel=(kd,kh,kw): fold kd into H
                kd, kh, kw = k
                H *= int(kd)
            else:
                kh, kw = k
        else:
            kh, kw = 1, 1
        groups = int(cfg.get("groups", 1))
        macs_per_out = kh * kw * (cin // groups)
        return int(H) * int(W) * int(cout) * int(macs_per_out)

    # DepthwiseConv2D
    if name == "DepthwiseConv2D":
        hwc = out_hw_c()
        if not hwc:
            return 0
        H, W, cin = hwc
        k = cfg.get("kernel_size")
        kh, kw = k if isinstance(k, (list, tuple)) else (k, k)
        depth_mult = int(cfg.get("depth_multiplier", 1))
        # Each input channel has its own kh*kw kernel; output channels = cin*depth_mult
        return int(H) * int(W) * int(cin) * int(depth_mult) * int(kh) * int(kw)

    # Dense
    if name == "Dense":
        s_in = layer.input_shape
        if (
            isinstance(s_in, (list, tuple))
            and s_in
            and isinstance(s_in[0], (list, tuple))
        ):
            s_in = s_in[0]
        if s_in is None:
            return 0
        in_features = int(s_in[-1])
        out_features = int(cfg.get("units"))
        return in_features * out_features

    # Others: treat as 0 (BN/Relu/Pooling/Add/Concat/etc)
    return 0


def compute_macs_manual(model):
    """Sum MACs across layers we know how to count."""
    total = 0
    for layer in model.layers:
        try:
            total += _layer_macs(layer)
        except Exception:
            pass
    return int(total)


# --- REPLACE your try_get_flops with this -----------------------------------
def try_get_flops(model, batch_size=1):
    """
    Try keras-flops; on any error (incl. duplicate op registration), fall back to manual MACs.
    Returns (flops, macs, used_fallback)
    """
    # Attempt keras-flops
    try:
        import sys

        if "keras_flops" in sys.modules:
            from keras_flops import get_flops  # already imported; should be safe
        else:
            from keras_flops import get_flops
        flops = int(get_flops(model, batch_size=batch_size))
        macs = flops // 2
        return flops, macs, False
    except Exception:
        # Fallback: manual MACs
        macs = compute_macs_manual(model)
        flops = macs * 2
        return flops, macs, True


# --------------------------
# Inference / reporting
# --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--target", required=True, help="Target directory containing model/config"
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float16", "bfloat16", "int8"],
        help="Param/activation dtype for memory estimations",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="For FLOPs reporting (if available)"
    )
    args = parser.parse_args()

    tdir = Path(args.target).resolve()
    if not tdir.exists() or not tdir.is_dir():
        print(f"Target not found: {tdir}")
        sys.exit(1)

    # Ensure output/<target_name> exists for report
    out_root = Path("output") / tdir.name
    out_root.mkdir(parents=True, exist_ok=True)
    report_path = out_root / "model_report.txt"

    # Read config (best-effort)
    cfg = read_yaml_config(tdir)
    model_cfg = cfg.get("model", {})
    features = cfg.get("features", {})

    # Find model artifacts
    keras_file = find_first(["*.keras"], tdir)
    weights_file = find_first(["*weights*.h5", "*.h5", "*.ckpt", "*.weights"], tdir)

    # Try to get input shape from config first
    input_shape = model_cfg.get("input_shape")
    if not input_shape:
        # Compose from features (best effort)
        n_mels = features.get("n_mels")
        num_frames = features.get("num_frames") or features.get("frames")
        if n_mels and num_frames:
            input_shape = [int(n_mels), int(num_frames), 1]

    # Load or build model
    model = None
    loaded_via = None
    if keras_file:
        try:
            model = keras.models.load_model(keras_file, compile=False)
            loaded_via = f"loaded from {keras_file.name}"
            # If input_shape is unknown, infer from loaded model
            if input_shape is None and model.inputs:
                ish = tuple(model.inputs[0].shape[1:])
                input_shape = [int(d) for d in ish]
        except Exception as e:
            print(f"Failed to load .keras model: {e}")

    if model is None:
        # Attempt to build a minimal MiniResNet if requested
        name = (model_cfg.get("name") or "").lower()
        num_classes = int(model_cfg.get("num_classes", 2))
        base_channels = int(model_cfg.get("base_channels", 32))
        num_blocks = int(model_cfg.get("num_blocks", 4))

        if not input_shape:
            print(
                "Cannot infer input_shape. Please put `model.input_shape: [H,W,C]` in your config."
            )
            sys.exit(1)

        if name in ("miniresnet", "mini_resnet", "resnet_mini", "tinyresnet"):
            model = build_miniresnet(
                tuple(input_shape), num_classes, base_channels, num_blocks
            )
            loaded_via = f"built MiniResNet from config (blocks={num_blocks}, base={base_channels})"
            # Load weights if present
            if weights_file and weights_file.suffix in [".h5", ".weights", ".ckpt"]:
                try:
                    model.load_weights(str(weights_file))
                    loaded_via += f" + loaded weights ({weights_file.name})"
                except Exception as e:
                    print(
                        f"Warning: could not load weights file {weights_file.name}: {e}"
                    )
        else:
            print(
                "No .keras model found and model builder is unknown.\n"
                "Provide a .keras file or set model.name: MiniResNet in your YAML."
            )
            sys.exit(1)

    # Compute metrics
    params = model.count_params()
    param_bytes = params * bytes_per_dtype(args.dtype)
    act_bytes = estimate_activation_bytes(model, batch_size=1, dtype=args.dtype)
    flops, macs, used_fallback = try_get_flops(model, batch_size=args.batch_size)

    macs = (flops // 2) if flops is not None else None

    # Storage sizes
    storage = {}
    if keras_file and keras_file.exists():
        storage["keras_file"] = keras_file.name
        storage["keras_bytes"] = keras_file.stat().st_size

    # TFLite sizes
    try:
        sizes = tflite_sizes(model, make_int8=True)
        storage.update(sizes)
    except Exception as e:
        storage["tflite_error"] = f"{e}"

    # Prepare report
    lines = []
    lines.append(f"=== Model Source ===")
    lines.append(f"{loaded_via}")
    lines.append("")
    lines.append(f"=== Input Shape ===")
    lines.append(f"{tuple(input_shape)}")
    lines.append("")
    lines.append(f"=== Parameters ===")
    lines.append(f"Total params: {params:,}")
    lines.append(f"Param memory (@{args.dtype}): {human_bytes(param_bytes)}")
    lines.append("")
    lines.append(f"=== Activations (estimate, batch=1) ===")
    lines.append(f"Activation memory (@{args.dtype}): {human_bytes(act_bytes)}")
    lines.append("")
    lines.append("=== Compute ===")
    if used_fallback:
        lines.append(f"MACs (manual, approx): {macs:,}")
        lines.append(f"FLOPs ≈ 2×MACs: {flops:,}")
        lines.append("Note: Manual estimator covers Conv/DWConv/Dense; others ignored.")
    else:
        lines.append(f"FLOPs (keras-flops, per batch={args.batch_size}): {flops:,}")
        lines.append(f"MACs  (approx): {macs:,}")
    lines.append("")
    lines.append(f"=== Storage ===")
    if "keras_bytes" in storage:
        lines.append(f"{storage['keras_file']}: {human_bytes(storage['keras_bytes'])}")
    if "tflite_fp32_bytes" in storage:
        lines.append(f"TFLite FP32: {human_bytes(storage['tflite_fp32_bytes'])}")
    if "tflite_int8_wo_calib_bytes" in storage:
        lines.append(
            f"TFLite INT8 (no rep. dataset): {human_bytes(storage['tflite_int8_wo_calib_bytes'])}"
        )
        lines.append(
            "Note: For *full* int8 (weights+activations), provide a representative dataset during conversion."
        )
    if "tflite_error" in storage:
        lines.append(f"TFLite convert error: {storage['tflite_error']}")
    lines.append("")

    report_txt = "\n".join(lines)
    print(report_txt)

    # Save report
    with open(report_path, "w") as f:
        f.write(report_txt)
    print(f"[Saved] {report_path}")

    # Also save a JSON summary
    summary = {
        "source": loaded_via,
        "input_shape": input_shape,
        "params": params,
        "param_bytes": param_bytes,
        "activation_bytes_estimate": act_bytes,
        "flops_per_batch": flops,
        "macs_per_batch": macs,
        "storage": storage,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
    }
    with open(out_root / "model_report.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Saved] {out_root / 'model_report.json'}")


if __name__ == "__main__":
    main()
