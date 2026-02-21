import argparse
import glob
import os
import pprint
from pathlib import Path
from typing import Dict

import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
from birdnet.audio_based_prediction import predict_species_within_audio_file
from birdnet.audio_based_prediction_mp import predict_species_within_audio_files_mp
from tqdm import tqdm

from audio_processing import generate_mel_spectrogram
from misc import load_config

# Use the logits version of your model builder
from models.miniresnet import build_miniresnet

pp = pprint.PrettyPrinter(indent=4)


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
    parser.add_argument("input_dir")
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
    print(config["ml"]["model_path"])
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
    model.load_weights(config["ml"]["model_path"])

    # Load TFLite model (may be float or quantized)
    tfl_interpreter = None
    try:
        tfl_path = args.model_path
        print(tfl_path)
        tfl_interpreter = load_tflite_model(Path(tfl_path))
        tfl_in = tfl_interpreter.get_input_details()[0]
        tfl_out = tfl_interpreter.get_output_details()[0]
    except Exception as e:
        print(f"[TFLite] Disabled: {e}")
        tfl_interpreter = None
        tfl_in = tfl_out = None

    # BirdNET for comparison (best-effort: ignore errors)
    try:
        birdnet_results = list(
            predict_species_within_audio_files_mp(
                [
                    Path(x)
                    for x in glob.glob(
                        args.input_dir + "*.wav",
                    )
                ]
            )
        )
    except Exception as e:
        print(f"[BirdNET] Warning: {e}")
        birdnet_results = []

    # Convert into dict
    birdnet_dict = {}
    for x in birdnet_results:
        birdnet_dict[str(x[0]).rsplit("/", maxsplit=1)[-1]] = list(x[1].items())[0]

    def read_files(path):
        chunks = []
        for f in glob.glob(path + "*.wav"):
            y, _ = librosa.load(f)
            if y.size:
                peak = np.max(np.abs(y))
                if peak > 0 and np.isfinite(peak):
                    y = (y / peak).astype(np.float32, copy=False)

            chunks.append((os.path.basename(f), y))
        return chunks

    # Chunks
    chunks = read_files(args.input_dir)

    results = []
    for name, y_chunk in tqdm(
        chunks, total=len(chunks), desc="Processing", unit="chunk"
    ):
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
            # Run
            tfl_interpreter.set_tensor(tfl_in["index"], x_keras)
            tfl_interpreter.invoke()
            tflite_prob = float(
                np.squeeze(tfl_interpreter.get_tensor(tfl_out["index"]))
            )

        # Assemble result row
        results.append(
            {
                "name": name,
                "keras_prob": prob_keras,
                "tflite_prob": tflite_prob,
                "birdnet": (birdnet_dict[name]),
            }
        )

    # Load the on-edge prediction csv
    pred_csv = pd.read_csv(os.path.join(args.input_dir, "preds.csv"))

    merged = pd.merge(pred_csv, pd.DataFrame(results), left_on="rec", right_on="name")

    merged = merged[
        [
            "score",
            "name",
            "keras_prob",
            "tflite_prob",
            "birdnet",
        ]
    ]
    merged.to_csv("single_dir_output.csv", index=False)


if __name__ == "__main__":
    main()
