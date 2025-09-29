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
from models.binary_cnn import build_binary_cnn
from models.dual_class_cnn import build_dual_class_cnn

# from models.miniresnet import build_miniresnet
from models.miniresnet_logits import build_miniresnet
from models.tinychirp import build_cnn_mel


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _tflite_get_io(interpreter):
    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]
    return in_det, out_det


def _tflite_quantize(x_f32, scale, zp, dtype):
    # x_q = round(x/scale + zp)
    q = np.round(x_f32 / scale + zp)
    q = np.clip(q, np.iinfo(dtype).min, np.iinfo(dtype).max)
    return q.astype(dtype)


def _tflite_dequantize(y_int, scale, zp):
    # y_f = scale * (y_int - zp)
    return scale * (y_int.astype(np.float32) - zp)


def _fix_to_expected_hw(x3, expected_hw_c):
    """x3 rank-3 -> (H,W,C) shaped to expected (H,W,C)."""
    exp_H, exp_W, exp_C = expected_hw_c
    h, w = x3.shape[0], x3.shape[1]
    c = x3.shape[2] if x3.shape.rank == 3 else None

    # Add channel dim if missing
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

    # Final sanity: enforce expected channel dim
    if c is None or c != exp_C:
        if c == 1 and exp_C == 1:
            pass  # OK
        else:
            # force to single channel if needed
            x3 = x3[..., :1]
            c = 1

    # Now ensure H,W match exactly (no resizing here—better to fix dataset)
    if h != exp_H or w != exp_W:
        raise ValueError(
            f"Rep sample has shape (H,W,C)=({h},{w},{c}), expected ({exp_H},{exp_W},{exp_C})."
        )

    return x3


def rep_ds_gen_from_tfdata(ds, model, max_samples=100):
    """
    Yields [input] where input has shape (1, H, W, C) and dtype float32,
    matching model.input_shape = (None, H, W, C).
    """
    _, exp_H, exp_W, exp_C = model.input_shape  # e.g. (None,80,241,1)
    n = 0
    for elem in ds:
        # peel (x,y) or dict
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
                xi = tf.cast(x[i], tf.float32)  # rank-3 or rank-2
                if xi.shape.rank == 2:  # (H,W) -> (H,W,1)
                    xi = tf.expand_dims(xi, -1)
                xi = _fix_to_expected_hw(xi, (exp_H, exp_W, exp_C))
                xi = tf.expand_dims(xi, 0)  # add batch -> (1,H,W,C)
                yield [xi]
                n += 1
                if n >= max_samples:
                    return
        else:
            # Unbatched or bs==1
            if x.shape.rank == 4:  # (1,H,W,C) or (B,H,W,C with B==1)
                xi = tf.cast(x[:1], tf.float32)
                # verify
                if xi.shape[1:] != (exp_H, exp_W, exp_C):
                    xi_fixed = _fix_to_expected_hw(
                        tf.squeeze(xi, 0), (exp_H, exp_W, exp_C)
                    )
                    xi = tf.expand_dims(xi_fixed, 0)
            else:
                xi = tf.cast(x, tf.float32)
                if xi.shape.rank == 2:
                    xi = tf.expand_dims(xi, -1)  # (H,W,1)
                xi = _fix_to_expected_hw(xi, (exp_H, exp_W, exp_C))
                xi = tf.expand_dims(xi, 0)
            yield [xi]
            n += 1
            if n >= max_samples:
                return


def dataset_as_single_samples(ds, model):
    """
    Yields (x_np, y_scalar) where x_np has shape (1,H,W,1) float32
    matching model.input_shape = (None,H,W,1).
    """
    _, exp_H, exp_W, exp_C = model.input_shape
    for elem in ds:
        if isinstance(elem, (tuple, list)):
            x, y = elem[0], elem[1]
        elif isinstance(elem, dict):
            # pick first tensor as x, second as y (adjust if your dict is different)
            keys = list(elem.keys())
            x, y = elem[keys[0]], elem[keys[1]]
        else:
            raise ValueError("val_ds must yield (x,y) or dict with features+label")

        x = tf.convert_to_tensor(x)
        # If batched, take first item (we assume ds is batched(1) already)
        if x.shape.rank == 4:
            x = x[0]  # (H,W,C) or (C,H,W)
        x = tf.cast(x, tf.float32)
        x = _fix_to_expected_hw(x, (exp_H, exp_W, exp_C))
        x = tf.expand_dims(x, 0)  # (1,H,W,C)

        # squeeze y to scalar float
        y_scalar = float(tf.reshape(tf.cast(y, tf.float32), [-1])[0].numpy())
        yield x.numpy(), y_scalar


def eval_keras_binary(model, val_ds, threshold=0.5):
    """Return accuracy and confusion counts for Keras model on val_ds."""
    y_true, y_prob = [], []
    for x, y in dataset_as_single_samples(val_ds, model):
        p = model.predict(x, verbose=0)  # (1,1)
        y_true.append(y)
        y_prob.append(float(np.squeeze(p)))
    y_true = np.asarray(y_true, dtype=np.float32)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    y_pred = (y_prob >= threshold).astype(np.float32)
    acc = float(np.mean((y_pred == y_true)))
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    return acc, dict(tp=tp, tn=tn, fp=fp, fn=fn), y_prob


def eval_tflite_binary(interpreter, val_ds, model, threshold=0.5):
    """
    Run TFLite on val_ds, return accuracy and confusion counts.
    Handles float or int8 I/O, and applies sigmoid if output is a logit.
    """
    in_det, out_det = _tflite_get_io(interpreter)
    in_index = in_det["index"]
    out_index = out_det["index"]

    in_is_float = in_det["dtype"] == np.float32
    out_is_float = out_det["dtype"] == np.float32
    in_scale, in_zp = in_det.get("quantization", (0.0, 0))
    out_scale, out_zp = out_det.get("quantization", (0.0, 0))

    # Detect if graph already has sigmoid/logistic
    try:
        has_logistic = any(
            "LOGISTIC" in (d.get("op_name") or "")
            for d in interpreter._get_ops_details()
        )
    except Exception:
        has_logistic = False

    y_true, y_prob = [], []

    fixed_shape = tuple(in_det["shape"])
    expect_batch1 = len(fixed_shape) > 0 and fixed_shape[0] == 1

    # IMPORTANT: iterator already yields (1,H,W,C)
    for x_np, y_scalar in dataset_as_single_samples(val_ds, model):
        xin = x_np  # already batched: (1,H,W,C)

        # If shapes differ, try to resize once (dynamic shape). Else, trim to batch=1.
        if xin.shape != tuple(in_det["shape"]):
            try:
                interpreter.resize_tensor_input(in_index, xin.shape, strict=False)
                interpreter.allocate_tensors()
                # refresh details after resize
                in_det, out_det = _tflite_get_io(interpreter)
                in_index = in_det["index"]
                out_index = out_det["index"]
                in_is_float = in_det["dtype"] == np.float32
                out_is_float = out_det["dtype"] == np.float32
                in_scale, in_zp = in_det.get("quantization", (0.0, 0))
                out_scale, out_zp = out_det.get("quantization", (0.0, 0))
            except Exception:
                if xin.shape[0] != 1 and expect_batch1:
                    xin = xin[:1, ...]

        # Quantize if needed
        if in_is_float:
            x_feed = xin.astype(np.float32)
        else:
            if not in_scale or in_scale <= 0:
                raise RuntimeError("Quantized TFLite input has invalid scale.")
            x_feed = _tflite_quantize(xin, in_scale, in_zp, in_det["dtype"])

        interpreter.set_tensor(in_index, x_feed)
        interpreter.invoke()
        y_raw = interpreter.get_tensor(out_index)  # e.g. (1,1) or (1,)

        # Dequantize / to float
        if out_is_float:
            y_f = y_raw.astype(np.float32)
        else:
            if out_scale and out_scale > 0:
                y_f = _tflite_dequantize(y_raw, out_scale, out_zp)
            else:
                # Treat as logits in int domain (rare)
                y_f = y_raw.astype(np.float32)

        y_val = float(np.squeeze(y_f))
        y_p = y_val if has_logistic else _sigmoid(y_val)

        y_true.append(float(y_scalar))
        y_prob.append(y_p)

    y_true = np.asarray(y_true, dtype=np.float32)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    y_pred = (y_prob >= threshold).astype(np.float32)

    acc = float(np.mean((y_pred == y_true)))
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    return acc, dict(tp=tp, tn=tn, fp=fp, fn=fn), y_prob


if __name__ == "__main__":
    argparse = ArgumentParser()
    argparse.add_argument("--input_model", required=True)
    argparse.add_argument("--output_model", required=True)
    args = argparse.parse_args()

    logging.getLogger().setLevel(logging.INFO)
    # Load the config
    CONFIG_PATH = "config.yaml"
    logging.info(f"Loading config from {CONFIG_PATH}")
    config = load_config(CONFIG_PATH)

    if config is None:
        exit(1)

    logging.info(
        f"Running Experiment {config['exp']['name']} for {config['exp']['target']}"
    )

    # Initialize the framework
    init(config["exp"]["random_state"])

    if config["data"]["use_dataset"] == "birdset":
        logging.info("[Info] Loading birdset!")
        datasets = get_birdset_dataset(config)
    elif config["data"]["use_dataset"] == "birdclef":
        logging.info("[Info] Loading Birdclef")
        datasets = get_birdclef_datasets(config)
    else:
        logging.info("[Info] Loading custom Dataset")
        datasets = get_custom_dataset(config)

    # Prepare testing dataset
    ds = (
        datasets["test_ds"]
        .map(lambda x, _: x, num_parallel_calls=tf.data.AUTOTUNE)
        .unbatch()
        .batch(1)
    )

    # Load the model
    model = build_miniresnet(
        input_shape=(
            config["data"]["audio"]["n_mels"],
            config["data"]["audio"]["n_frames"],
            1,
        ),
        n_classes=1,
        loss=config["ml"]["loss"],
        logits=False,
    )

    model.load_weights(args.input_model)

    # Load the model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: rep_ds_gen_from_tfdata(ds, model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # converter.inference_input_type = tf.int8
    # converter.inference_output_type = tf.int8
    tflite_model = converter.convert()

    # Output the model
    with open(args.output_model, "wb") as f:
        f.write(tflite_model)

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    # Inputs / outputs
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

    # Ops present (unordered set)
    ops = {
        d.get("op_name", "?") for d in interpreter._get_ops_details()
    }  # _get_ops_details is semi-private
    print("\n== Ops in graph ==")
    print(sorted(ops))

    # ===== Run both evaluations on your val_ds =====
    val_ds = (
        datasets["val_ds"].unbatch().batch(1).shuffle(1000).take(1000)
    )  # (x,y) pairs

    keras_acc, keras_cm, keras_probs = eval_keras_binary(model, val_ds, threshold=0.5)
    print("\n[Keras]  acc=%.4f  cm=%s" % (keras_acc, keras_cm))

    tfl_acc, tfl_cm, tfl_probs = eval_tflite_binary(
        interpreter, val_ds, model, threshold=0.5
    )
    print("[TFLite] acc=%.4f  cm=%s" % (tfl_acc, tfl_cm))
