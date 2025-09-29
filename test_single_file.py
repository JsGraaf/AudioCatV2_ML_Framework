import argparse
import os
import pprint
from pathlib import Path
from typing import Dict

import librosa
import numpy as np
import tensorflow as tf
from birdnet.audio_based_prediction import predict_species_within_audio_file

from audio_processing import generate_mel_spectrogram
from misc import load_config

# Use the logits version of your model builder
from models.miniresnet import build_miniresnet

pp = pprint.PrettyPrinter(indent=4)


def split_audio(y, sr, chunk_sec):
    chunk_len = int(chunk_sec * sr)
    n_chunks = int(np.ceil(len(y) / chunk_len))
    chunks = []
    for i in range(n_chunks):
        start = i * chunk_len
        end = min((i + 1) * chunk_len, len(y))
        y_chunk = y[start:end]
        if len(y_chunk) < chunk_len:
            y_chunk = np.pad(y_chunk, (0, chunk_len - len(y_chunk)))
        chunks.append((y_chunk, start / sr, end / sr))
    return chunks


def npdtype_to_ctype(dt: np.dtype) -> str:
    if dt == np.float32:
        return "float"
    if dt == np.int8:
        return "int8_t"
    if dt == np.uint8:
        return "uint8_t"
    if dt == np.int16:
        return "int16_t"
    if dt == np.int32:
        return "int32_t"
    return "uint8_t"


def dump_c_array(name: str, arr: np.ndarray, filename: str = "c_array_dump.h"):
    """Dump a numpy array as a C array into a file."""
    flat = arr.flatten()
    ctype = npdtype_to_ctype(arr.dtype)

    with open(filename, "w") as f:
        f.write(
            f"// C array dump of input tensor ({ctype}), shape={list(arr.shape)}, len={flat.size}\n"
        )
        f.write(f"const {ctype} {name}[{flat.size}] = {{\n")
        per_line = 16
        for j, v in enumerate(flat):
            if np.issubdtype(arr.dtype, np.floating):
                f.write(f"  {float(v):.6f}f,")
            else:
                f.write(f"  {int(v)},")
            if (j + 1) % per_line == 0:
                f.write("\n")
        if flat.size % per_line != 0:
            f.write("\n")
        f.write("};\n")

    print(f"[INFO] Dumped C array to {filename}")


def load_tflite_model(path: Path):
    print("Loading TFLite model...")
    interpreter = tf.lite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()

    o = interpreter.get_output_details()[0]
    print("OUT quant:", o.get("quantization", (0.0, 0)))  # (scale, zp)
    try:
        ops = {d.get("op_name", "?") for d in interpreter._get_ops_details()}  # type: ignore
        print("[ops]", sorted(ops))
    except Exception:
        pass

    return interpreter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("--model_path", required=True)
    args = parser.parse_args()

    CONFIG_PATH = os.path.join("config.yaml")
    config = load_config(CONFIG_PATH)
    if config is None:
        raise SystemExit("Failed to load config.yaml")

    audio_cfg: Dict = config["data"]["audio"]
    win_sec = float(audio_cfg["seconds"])
    sr = int(audio_cfg["sample_rate"])
    n_fft = int(audio_cfg["n_fft"])
    hop_length = int(audio_cfg["hop_length"])
    n_mels = int(audio_cfg["n_mels"])
    fmin = int(audio_cfg["fmin"])
    fmax = int(audio_cfg["fmax"])

    use_pcen = bool(audio_cfg.get("use_pcen", False))
    pcen_time_constant = float(audio_cfg.get("pcen_time_constant", 0.4))
    pcen_gain = float(audio_cfg.get("pcen_gain", 0.98))
    pcen_bias = float(audio_cfg.get("pcen_bias", 2.0))
    pcen_power = float(audio_cfg.get("pcen_power", 0.5))
    pcen_eps = float(audio_cfg.get("pcen_eps", 1e-6))

    # Build Keras model with LOGITS (no sigmoid in final layer)
    model = build_miniresnet(
        input_shape=(
            config["data"]["audio"]["n_mels"],
            config["data"]["audio"]["n_frames"],
            1,
        ),
        n_classes=1,
    )
    model.load_weights(config["ml"]["model_path"])

    # Load TFLite model (may be float or quantized)
    tfl_interpreter = None
    try:
        tfl_path = args.model_path
        print(tfl_path)
        tfl_interpreter = load_tflite_model(Path(tfl_path))
        tfl_in = tfl_interpreter.get_input_details()[0]
        tfl_out = tfl_interpreter.get_output_details()[0]
        tfl_in_dtype = tfl_in["dtype"]
        tfl_scale, tfl_zp = tfl_in.get("quantization", (0.0, 0))
    except Exception as e:
        print(f"[TFLite] Disabled: {e}")
        tfl_interpreter = None
        tfl_in = tfl_out = None

    # Load audio
    y, sr = librosa.load(args.input_file, sr=config["data"]["audio"]["sample_rate"])

    # BirdNET for comparison (best-effort: ignore errors)
    try:
        birdnet_results = list(predict_species_within_audio_file(Path(args.input_file)))
    except Exception as e:
        print(f"[BirdNET] Warning: {e}")
        birdnet_results = []

    # Chunks
    chunks = split_audio(y, sr, config["data"]["audio"]["seconds"])

    results = []
    for i_chunk, (y_chunk, t0, t1) in enumerate(chunks):
        # Make mel
        spec = generate_mel_spectrogram(
            audio=y_chunk,
            sr=sr,
            n_fft=n_fft,
            n_mels=n_mels,
            hop_length=hop_length,
            win_length=None,
            window="hann",
            center=True,
            pad_mode="reflect",
            power=2.0,
            fmin=fmin,
            fmax=fmax,
            norm=None,
            use_pcen=use_pcen,
            pcen_time_constant=pcen_time_constant,
            pcen_gain=pcen_gain,
            pcen_bias=pcen_bias,
            pcen_power=pcen_power,
            pcen_eps=pcen_eps,
        )

        H_expected = int(config["data"]["audio"]["n_mels"])
        W_expected = int(config["data"]["audio"]["n_frames"])

        if spec.ndim != 2:
            raise ValueError(f"Expected 2-D [n_mels, n_frames], got {spec.shape}")

        H, W = spec.shape
        if H == W_expected and W == H_expected:
            spec = spec.T
            H, W = spec.shape

        if H != H_expected:
            raise ValueError(f"n_mels mismatch: got {H}, expected {H_expected}.")

        if W < W_expected:
            pad = np.zeros((H, W_expected - W), dtype=spec.dtype)
            spec = np.concatenate([spec, pad], axis=1)
        elif W > W_expected:
            off = (W - W_expected) // 2
            spec = spec[:, off : off + W_expected]

        # Keras logits & prob
        x_keras = spec.astype(np.float32)[..., None][None, ...]
        prob_keras = model.predict(x_keras, verbose=0)  # logits
        prob_keras = float(np.squeeze(prob_keras))

        # TFLite inference (always produce logits and sigmoid(prob) explicitly)
        tflite_prob = None

        if tfl_interpreter is not None:
            # Prepare input for TFLite

            # Run
            tfl_interpreter.set_tensor(tfl_in["index"], x_keras)
            tfl_interpreter.invoke()
            tflite_prob = float(
                np.squeeze(tfl_interpreter.get_tensor(tfl_out["index"]))
            )

            # (Optional) Dump first chunk input as a C array
            if i_chunk == 1:
                dump_c_array("tflite_input_data", x_keras[0], "input_tensor_dump.h")

        # Assemble result row
        results.append(
            {
                "chunk": i_chunk,
                "t0": t0,
                "t1": t1,
                "keras_prob": prob_keras,
                "tflite_prob": tflite_prob,
                "birdnet": (
                    birdnet_results[i_chunk] if i_chunk < len(birdnet_results) else None
                ),
            }
        )

    for r in results:
        print(f"BirdNET: {r['birdnet']}")
        print(f"Keras: {r['keras_prob']}")
        print(f"TFLite: {r['tflite_prob']}")
        print("\n\n")


if __name__ == "__main__":
    main()
