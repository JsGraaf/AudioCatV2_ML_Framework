import logging
from argparse import ArgumentParser

import numpy as np
import tensorflow as tf

from dataset_loaders import (
    get_birdclef_datasets,
    get_birdset_dataset,
    get_custom_dataset,
)
from init import init
from misc import load_config
from metric_utils import confusion_at_threshold
from models.binary_cnn import build_binary_cnn
from models.dual_class_cnn import build_dual_class_cnn

from models.miniresnet import build_miniresnet
from models.tinychirp import build_cnn_mel
import os


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _tflite_get_io(interpreter):
    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]
    return in_det, out_det


def _tflite_quantize(x_f32, scale, zp, dtype):
    q = np.round(x_f32 / scale + zp)
    q = np.clip(q, np.iinfo(dtype).min, np.iinfo(dtype).max)
    return q.astype(dtype)


def _tflite_dequantize(y_int, scale, zp):
    return scale * (y_int.astype(np.float32) - zp)


def _fix_to_expected_hw(x3, expected_hw_c):
    """x3 rank-3 -> (H,W,C) shaped to expected (H,W,C)."""
    exp_H, exp_W, exp_C = expected_hw_c
    h, w = x3.shape[0], x3.shape[1]
    c = x3.shape[2] if x3.shape.rank == 3 else None

    if x3.shape.rank == 2:
        x3 = tf.expand_dims(x3, -1)  # (H,W,1)
        h, w, c = x3.shape

    # If channels-first (C,H,W) -> (H,W,C)
    if (
        x3.shape.rank == 3
        and x3.shape[0] == exp_C
        and x3.shape[1] == exp_H
        and x3.shape[2] == exp_W
    ):
        x3 = tf.transpose(x3, [1, 2, 0])
        h, w, c = x3.shape

    # If transposed (W,H,C) -> (H,W,C)
    if (h == exp_W) and (w == exp_H) and (c == exp_C):
        x3 = tf.transpose(x3, [1, 0, 2])
        h, w, c = x3.shape

    if c is None or c != exp_C:
        if c == 1 and exp_C == 1:
            pass
        else:
            x3 = x3[..., :1]
            c = 1

    if h != exp_H or w != exp_W:
        raise ValueError(
            f"Rep sample has shape (H,W,C)=({h},{w},{c}), expected ({exp_H},{exp_W},{exp_C})."
        )

    return x3


def rep_ds_gen_from_tfdata(ds, model, max_samples=-1):
    """
    Representative dataset generator for TFLite calibration.
    Yields [input] with shape (1,H,W,C) float32.
    """
    _, exp_H, exp_W, exp_C = model.input_shape  # e.g. (None,80,241,1)
    n = 0
    for elem in ds:
        if isinstance(elem, (tuple, list)):
            x = elem[0]
        elif isinstance(elem, dict):
            x = next(v for v in elem.values() if tf.is_tensor(v))
        else:
            x = elem

        x = tf.convert_to_tensor(x)

        # If batched, iterate items in batch
        if x.shape.rank >= 4 and x.shape[0] is not None and x.shape[0] > 1:
            bs = int(x.shape[0])
            for i in range(bs):
                xi = tf.cast(x[i], tf.float32)
                if xi.shape.rank == 2:
                    xi = tf.expand_dims(xi, -1)
                xi = _fix_to_expected_hw(xi, (exp_H, exp_W, exp_C))
                xi = tf.expand_dims(xi, 0)
                yield [xi]
                n += 1
                if max_samples > 0 and n >= max_samples:
                    return
        else:
            # Unbatched or bs==1
            if x.shape.rank == 4:
                xi = tf.cast(x[:1], tf.float32)
                if xi.shape[1:] != (exp_H, exp_W, exp_C):
                    xi_fixed = _fix_to_expected_hw(
                        tf.squeeze(xi, 0), (exp_H, exp_W, exp_C)
                    )
                    xi = tf.expand_dims(xi_fixed, 0)
            else:
                xi = tf.cast(x, tf.float32)
                if xi.shape.rank == 2:
                    xi = tf.expand_dims(xi, -1)
                xi = _fix_to_expected_hw(xi, (exp_H, exp_W, exp_C))
                xi = tf.expand_dims(xi, 0)
            yield [xi]
            n += 1
            if max_samples > 0 and n >= max_samples:
                return


def dataset_as_single_samples(ds, model):
    """
    Yields (x_np, y_scalar) with x_np shape (1,H,W,1) float32,
    matching model.input_shape = (None,H,W,1).
    """
    _, exp_H, exp_W, exp_C = model.input_shape
    for elem in ds:
        if isinstance(elem, (tuple, list)):
            x, y = elem[0], elem[1]
        elif isinstance(elem, dict):
            keys = list(elem.keys())
            x, y = elem[keys[0]], elem[keys[1]]
        else:
            raise ValueError("val_ds must yield (x,y) or dict with features+label")

        x = tf.convert_to_tensor(x)
        if x.shape.rank == 4:
            x = x[0]
        x = tf.cast(x, tf.float32)
        x = _fix_to_expected_hw(x, (exp_H, exp_W, exp_C))
        x = tf.expand_dims(x, 0)

        y_scalar = float(tf.reshape(tf.cast(y, tf.float32), [-1])[0].numpy())
        yield x.numpy(), y_scalar


# --------- NEW: Collect predictions once, then scan thresholds efficiently ---------


def collect_probs_keras(model, val_ds):
    """Return arrays y_true, y_prob from a single pass."""
    y_true, y_prob = [], []
    for x, y in dataset_as_single_samples(val_ds, model):
        p = model.predict(x, verbose=0)  # (1,1)
        y_true.append(y)
        y_prob.append(float(np.squeeze(p)))
    return np.asarray(y_true, dtype=np.float32), np.asarray(y_prob, dtype=np.float32)


def collect_probs_tflite(interpreter, val_ds, model):
    """Return arrays y_true, y_prob from a single pass of the TFLite model."""
    in_det, out_det = _tflite_get_io(interpreter)
    in_index = in_det["index"]
    out_index = out_det["index"]

    in_is_float = in_det["dtype"] == np.float32
    out_is_float = out_det["dtype"] == np.float32
    in_scale, in_zp = in_det.get("quantization", (0.0, 0))
    out_scale, out_zp = out_det.get("quantization", (0.0, 0))

    try:
        has_logistic = any(
            "LOGISTIC" in (d.get("op_name") or "")
            for d in interpreter._get_ops_details()
        )
    except Exception:
        has_logistic = False

    y_true, y_prob = [], []

    for x_np, y_scalar in dataset_as_single_samples(val_ds, model):
        xin = x_np  # (1,H,W,C)

        if tuple(xin.shape) != tuple(in_det["shape"]):
            try:
                interpreter.resize_tensor_input(in_index, xin.shape, strict=False)
                interpreter.allocate_tensors()
                in_det, out_det = _tflite_get_io(interpreter)
                in_index = in_det["index"]
                out_index = out_det["index"]
                in_is_float = in_det["dtype"] == np.float32
                out_is_float = out_det["dtype"] == np.float32
                in_scale, in_zp = in_det.get("quantization", (0.0, 0))
                out_scale, out_zp = out_det.get("quantization", (0.0, 0))
            except Exception:
                if xin.shape[0] != 1:
                    xin = xin[:1, ...]

        if in_is_float:
            x_feed = xin.astype(np.float32)
        else:
            if not in_scale or in_scale <= 0:
                raise RuntimeError("Quantized TFLite input has invalid scale.")
            x_feed = _tflite_quantize(xin, in_scale, in_zp, in_det["dtype"])

        interpreter.set_tensor(in_index, x_feed)
        interpreter.invoke()
        y_raw = interpreter.get_tensor(out_index)

        if out_is_float:
            y_f = y_raw.astype(np.float32)
        else:
            if out_scale and out_scale > 0:
                y_f = _tflite_dequantize(y_raw, out_scale, out_zp)
            else:
                y_f = y_raw.astype(np.float32)

        y_val = float(np.squeeze(y_f))
        y_p = y_val if has_logistic else _sigmoid(y_val)

        y_true.append(float(y_scalar))
        y_prob.append(y_p)

    return np.asarray(y_true, dtype=np.float32), np.asarray(y_prob, dtype=np.float32)


def best_thresh_at_precision(y_true, y_prob, min_precision=0.9):
    """
    Efficiently scan thresholds to maximize recall with precision >= min_precision.
    If multiple thresholds have the same recall, prefer higher precision.

    Returns:
      t_best, recall_best, precision_best
    """
    y_true = y_true.astype(np.int32)
    pos_total = int(np.sum(y_true))
    if pos_total == 0:
        return 0.5, 0.0, 0.0  # degenerate

    # Sort by predicted probability desc
    order = np.argsort(-y_prob)
    y_prob_sorted = y_prob[order]
    y_true_sorted = y_true[order]

    # Cumulative TP as we lower threshold from +inf to -inf
    tp_cum = np.cumsum(y_true_sorted)  # tp at each cut
    pred_cum = np.arange(1, len(y_true_sorted) + 1)  # total predicted positives
    precision = tp_cum / np.maximum(pred_cum, 1)
    recall = tp_cum / pos_total

    # Candidate thresholds are the score at each position
    thresh = y_prob_sorted

    # Filter candidates meeting precision
    ok = precision >= min_precision
    if not np.any(ok):
        # No threshold achieves desired precision
        # pick the one with highest precision anyway (with largest recall tie-break)
        i = int(np.argmax(precision))
        return float(thresh[i]), float(recall[i]), float(precision[i])

    # Among ok, choose max recall; if tie, max precision
    idx_ok = np.where(ok)[0]
    recalls_ok = recall[idx_ok]
    best_recall = np.max(recalls_ok)
    idx_best_recall = idx_ok[recalls_ok == best_recall]
    # tie-break on precision
    precisions_ties = precision[idx_best_recall]
    i_rel = int(np.argmax(precisions_ties))
    i_best = int(idx_best_recall[i_rel])

    return float(thresh[i_best]), float(recall[i_best]), float(precision[i_best])


if __name__ == "__main__":
    argparse = ArgumentParser()
    argparse.add_argument("--input_dir", required=True)
    args = argparse.parse_args()

    logging.getLogger().setLevel(logging.INFO)
    CONFIG_PATH = os.path.join(args.input_dir, "config.yaml")
    logging.info(f"Loading config from {CONFIG_PATH}")
    config = load_config(CONFIG_PATH)
    if config is None:
        exit(1)

    # Ensure output dir
    output_dir_path = os.path.join(args.input_dir, "tflite_model")
    os.makedirs(output_dir_path, exist_ok=True)
    output_model_name = os.path.join(
        output_dir_path, str(os.path.basename(args.input_dir)).lower() + ".tflite"
    )

    logging.info(
        f"Running Experiment {config['exp']['name']} for {config['exp']['target']}"
    )

    # Initialize
    init(config["exp"]["random_state"])

    # Load dataset
    if config["data"]["use_dataset"] == "birdset":
        logging.info("[Info] Loading birdset!")
        datasets = get_birdset_dataset(config)
    elif config["data"]["use_dataset"] == "birdclef":
        logging.info("[Info] Loading Birdclef")
        datasets = get_birdclef_datasets(config)
    else:
        logging.info("[Info] Loading custom Dataset")
        datasets = get_custom_dataset(config)

    # Prepare representative ds (features only)
    ds = (
        datasets["val_ds"]
        .map(lambda x, _: x, num_parallel_calls=tf.data.AUTOTUNE)
        .unbatch()
        .batch(1)
    )

    # Build & load Keras model
    model = build_miniresnet(
        input_shape=(
            config["data"]["audio"]["n_mels"],
            config["data"]["audio"]["n_frames"],
            1,
        ),
        n_classes=1,
        loss=config["ml"]["loss"],
        n_stacks=config["ml"]["stacks"],
        gamma=config["ml"]["gamma"],
        alpha=config["ml"]["alpha"],
    )
    model.load_weights(os.path.join(args.input_dir, "full_training_model.keras"))

    # Convert to INT8 TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: rep_ds_gen_from_tfdata(ds, model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    tflite_model = converter.convert()

    with open(output_model_name, "wb") as f:
        f.write(tflite_model)

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    # Inputs / outputs (debug)
    print("== Inputs ==")
    for d in interpreter.get_input_details():
        print(
            dict(
                name=d["name"],
                index=d["index"],
                shape=d["shape"],
                dtype=str(d["dtype"]),
                quant=d.get("quantization", None),
            )
        )

    print("\n== Outputs ==")
    for d in interpreter.get_output_details():
        print(
            dict(
                name=d["name"],
                index=d["index"],
                shape=d["shape"],
                dtype=str(d["dtype"]),
                quant=d.get("quantization", None),
            )
        )

    ops = {d.get("op_name", "?") for d in interpreter._get_ops_details()}
    print("\n== Ops in graph ==")
    print(sorted(ops))

    # ===== Evaluate once each, then optimize threshold =====
    val_ds = datasets["test_ds"].unbatch().batch(1)

    # Keras pass
    y_true_k, y_prob_k = collect_probs_keras(model, val_ds)
    t_k, r_k, p_k = best_thresh_at_precision(y_true_k, y_prob_k, min_precision=0.90)
    cm_k = confusion_at_threshold(y_true_k, y_prob_k, t_k)
    print(f"[Keras]  recall={r_k:.4f} precision={p_k:.4f}  cm={cm_k} t={t_k:.4f}")

    # TFLite pass
    y_true_t, y_prob_t = collect_probs_tflite(interpreter, val_ds, model)
    t_t, r_t, p_t = best_thresh_at_precision(y_true_t, y_prob_t, min_precision=0.90)
    cm_t = confusion_at_threshold(y_true_t, y_prob_t, t_t)
    print(f"[TFLite] recall={r_t:.4f} precision={p_t:.4f}  cm={cm_t} t={t_t:.4f}")

    # Write both to metrics.txt (append in one go)
    with open(os.path.join(output_dir_path, "metrics.txt"), "w") as f:
        f.write(
            f"[Keras]  recall={r_k:.4f} precision={p_k:.4f}  cm={cm_k} t={t_k:.4f}\n\n"
        )
        f.write(
            f"[TFLite] recall={r_t:.4f} precision={p_t:.4f}  cm={cm_t} t={t_t:.4f}\n"
        )
