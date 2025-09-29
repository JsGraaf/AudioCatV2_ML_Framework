import logging
from pathlib import Path
import numpy as np
import tensorflow as tf

from init import init
from misc import load_config


def load_tflite_model(path: Path):
    interpreter = tf.lite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    return interpreter


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    # Load config & init
    CONFIG_PATH = Path("config_tflite.yaml")
    config = load_config(CONFIG_PATH)
    assert config is not None, "Failed to load config.yaml"
    init(config["exp"]["random_state"])

    # Load TFLite model
    interpreter = load_tflite_model(Path(config["ml"]["model_path"]))
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Use first input/output tensor
    i0 = input_details[0]
    o0 = output_details[0]

    in_shape = tuple(i0["shape"])
    in_dtype = i0["dtype"]

    scale, zp = i0.get("quantization", (0.0, 0))

    # We want an input that represents float 1.0 everywhere
    x_float = 0.0
    x_val = 15

    if scale and scale > 0 and np.issubdtype(in_dtype, np.integer):
        # Quantized path: q = round(x/scale + zero_point), then clip to dtype range
        q_val = int(np.round(x_float / scale + zp))
        q_val = int(np.clip(q_val, np.iinfo(in_dtype).min, np.iinfo(in_dtype).max))
        x = np.full(in_shape, x_val, dtype=in_dtype)
    else:
        # Float (or no quantization): just fill with 1.0 in the right dtype
        x = np.full(in_shape, x_float, dtype=in_dtype)

    print("Input:", x)

    # Set input & run
    interpreter.set_tensor(i0["index"], x)
    interpreter.invoke()

    # Get raw output
    y_raw = interpreter.get_tensor(o0["index"])

    print("Raw output:", y_raw)

    # If output is quantized, also print de-quantized values
    out_scale, out_zp = o0.get("quantization", (0.0, 0))
    if out_scale and out_scale > 0:
        y_float = out_scale * (y_raw.astype(np.float32) - out_zp)
        print("De-quantized output:", y_float)
