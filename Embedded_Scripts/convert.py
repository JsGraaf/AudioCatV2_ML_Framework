import os, re
import numpy as np
import soundfile as sf

input_dir = r"E:\Recordings"
output_dir = "output"
output_filename = "merged_output.wav"
sr = 32000

os.makedirs(output_dir, exist_ok=True)

# --- robust sort for both numbered and datetime file names ---
_numfile_re = re.compile(r"^(\d+)\.txt$", re.IGNORECASE)
_datetime_re = re.compile(
    r"^(?P<hh>\d{2})_(?P<mm>\d{2})_(?P<ss>\d{2})_(?P<ms>\d{3})__"
    r"(?P<DD>\d{2})-(?P<MM>\d{2})-(?P<YYYY>\d{4})\.txt$",
    re.IGNORECASE,
)

def _natural_chunks(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.findall(r"\d+|\D+", s)]

def file_sort_key(name: str):
    m = _datetime_re.match(name)
    if m:
        y, mo, d = int(m["YYYY"]), int(m["MM"]), int(m["DD"])
        hh, mm, ss, ms = int(m["hh"]), int(m["mm"]), int(m["ss"]), int(m["ms"])
        return (0, y, mo, d, hh, mm, ss, ms)
    m = _numfile_re.match(name)
    if m:
        return (1, int(m.group(1)))
    return (2, _natural_chunks(name))

def sanitize_floats(x: np.ndarray) -> np.ndarray:
    bad = ~np.isfinite(x)
    if bad.any():
        x = x.copy()
        x[bad] = 0.0
    return x

# --- merge: concatenate EXACTLY what’s in each file (no 4096 slicing, no amplitude filter) ---
files = [f for f in os.listdir(input_dir) if f.lower().endswith(".txt")]
files.sort(key=file_sort_key)

chunks = []
for fname in files:
    path = os.path.join(input_dir, fname)
    with open(path, "rb") as f:
        raw = f.read()
    if len(raw) % 4 != 0:
        print(f"Skipping {fname}: size {len(raw)} not multiple of 4 bytes for float32")
        continue

    arr = np.frombuffer(raw, dtype=np.float32)
    arr = sanitize_floats(arr)
    chunks.append(arr)

if not chunks:
    raise SystemExit("No valid audio found.")

audio = np.concatenate(chunks, dtype=np.float32)

# optional cleanup
audio -= np.mean(audio)                    # remove DC
peak = np.max(np.abs(audio))
if peak > 0:
    target_linear = 10 ** (-1.0 / 20.0)    # -1 dBFS
    audio = np.clip(audio * (target_linear / peak), -1.0, 1.0)

out_path = os.path.join(output_dir, output_filename)
sf.write(out_path, audio, sr)
print(f"Wrote {out_path} | samples={len(audio)}")
