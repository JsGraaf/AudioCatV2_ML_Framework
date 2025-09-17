from dataclasses import dataclass
from typing import Callable, Literal, Optional, Tuple

import numpy as np
from scipy.signal import butter, sosfilt

Decision = Literal[
    "idle_discard",
    "run_inference",
    "store_direct",
    "discard_after_inference",
    "store_after_inference",
]


@dataclass
class BaselineConfig:
    fs: int = 16_000  # Hz (paper downsampled to 16 kHz)
    hp_order: int = 9  # 9th-order Butterworth
    hp_cut_hz: float = 7000.0  # 7 kHz high-pass cutoff
    t_low: float = 1.0e-7  # tuned on train set in paper
    t_high: float = 1.29e-5  # tuned "power-saving" threshold in paper
    power_saving: bool = True  # paper’s power-saving flag


def minmax_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x_min = np.min(x)
    x_max = np.max(x)
    rng = max(x_max - x_min, eps)
    return (x - x_min) / rng * 2.0 - 1.0  # normalize ~[-1, 1]


def highpass_filter(x: np.ndarray, fs: int, order: int, cutoff_hz: float) -> np.ndarray:
    # Use SOS for numerical stability on higher-order filters
    sos = butter(order, cutoff_hz, btype="highpass", fs=fs, output="sos")
    return sosfilt(sos, x).astype(np.float32)


def segment_power(x: np.ndarray) -> float:
    # Average power of the filtered segment
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return 0.0
    return float(np.mean(x * x))


def baseline_decide(
    audio: np.ndarray,
    cfg: BaselineConfig,
    inference_fn: Optional[Callable[[np.ndarray], bool]] = None,
) -> Tuple[Decision, float]:
    """
    Returns (decision, P). If inference is requested but `inference_fn` is None,
    the decision will be 'run_inference' to signal the caller to run it.

    `inference_fn` must return True for 'target' and False otherwise.
    """
    # Step 0: preconditions
    if audio.ndim > 1:
        # mixdown to mono if needed
        audio = np.mean(audio, axis=-1)
    audio = audio.astype(np.float32, copy=False)

    # Step 1: normalize and high-pass
    x = minmax_normalize(audio)
    x_hp = highpass_filter(x, cfg.fs, cfg.hp_order, cfg.hp_cut_hz)

    # Step 2: compute average power
    P = segment_power(x_hp)

    # Step 3: threshold logic
    if P < cfg.t_low:
        # Below noise floor → idle/low-power & discard
        return "idle_discard", P

    if cfg.power_saving:
        # If below t_high, run inference; otherwise store directly
        if P < cfg.t_high:
            if inference_fn is None:
                return "run_inference", P
            is_target = bool(inference_fn(audio))
            return (
                "store_after_inference" if is_target else "discard_after_inference",
                P,
            )
        else:
            # Loud enough → store without inference
            return "store_direct", P
    else:
        # Always run inference after passing t_low
        if inference_fn is None:
            return "run_inference", P
        is_target = bool(inference_fn(audio))
        return ("store_after_inference" if is_target else "discard_after_inference", P)


# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    # Fake audio: 3 s at 16 kHz
    cfg = BaselineConfig()
    t = np.arange(0, 3 * cfg.fs) / cfg.fs
    # Example: quiet background + a 6 kHz tone burst (slightly below hp cutoff)
    sig = 0.01 * np.random.randn(t.size).astype(np.float32)
    sig += (np.sin(2 * np.pi * 6000 * t) * (t > 1.0) * (t < 2.0)).astype(
        np.float32
    ) * 0.1

    # Dummy inference function
    def dummy_infer(wave: np.ndarray) -> bool:
        # Here you’d run your Tiny model (or Python proxy).
        # Return True if 'target', False otherwise.
        return np.max(np.abs(wave)) > 0.2  # placeholder rule

    decision, P = baseline_decide(sig, cfg, inference_fn=dummy_infer)
    print(f"Decision: {decision}, P={P:.3e}")
