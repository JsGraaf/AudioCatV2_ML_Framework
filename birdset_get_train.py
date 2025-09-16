#!/usr/bin/env python3
import argparse
import csv
import math
import os
from pathlib import Path

from datasets import Audio, load_dataset  # pip install datasets<=3.6.0

# BirdSet currently recommends datasets<=3.6.0. See dataset card.


# Optional: eBird taxonomy CSV columns expected:
#   scientific_name,common_name,ebird_code
def load_ebird_map(tax_csv: Path):
    m = {}
    if tax_csv and tax_csv.exists():
        import csv

        with tax_csv.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                code = (
                    r.get("ebird_code")
                    or r.get("SPECIES_CODE")
                    or r.get("species_code")
                )
                sci = (
                    r.get("scientific_name")
                    or r.get("SCI_NAME")
                    or r.get("scientificName")
                )
                com = (
                    r.get("common_name")
                    or r.get("PRIMARY_COM_NAME")
                    or r.get("englishName")
                )
                if code:
                    m[code.strip().lower()] = (
                        (sci or "").strip(),
                        (com or "").strip(),
                    )
    return m


def clamp_window(center, half, total):
    start = max(0.0, center - half)
    end = min(total, center + half)
    # ensure exact 3.0 s when possible
    if end - start < 2 * half:
        if start == 0.0:
            end = min(total, 2 * half)
        elif end == total:
            start = max(0.0, total - 2 * half)
    return float(start), float(end)


def main():
    ap = argparse.ArgumentParser(description="Create BirdSet HSN CSV of 3s positives.")
    ap.add_argument(
        "--config", default="HSN", help="BirdSet split/config (HSN, HSN_xc, HSN_scape)."
    )
    ap.add_argument("--cache_dir", default="cache/huggingface", help="HF cache dir")
    ap.add_argument(
        "--tax_csv",
        type=Path,
        default=None,
        help="Optional eBird taxonomy CSV to fill names.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("output/hsn_train.csv"),
        help="Output CSV path",
    )
    ap.add_argument(
        "--min_event",
        type=float,
        default=0.05,
        help="Drop events shorter than this (s)",
    )
    ap.add_argument("--win", type=float, default=3.0, help="Window length (seconds)")
    args = ap.parse_args()

    eb_map = load_ebird_map(args.tax_csv) if args.tax_csv else {}

    # Load BirdSet (HSN). Trust remote code is required on some setups.
    print("Loading Dataset")
    ds = load_dataset(
        "DBD-research-group/BirdSet",
        args.config,
        trust_remote_code=True,
        cache_dir=args.cache_dir,
    )

    # We’ll use the TRAIN (focal) split; positives come from 'detected_events'
    train = ds["train"]

    # Cast 'audio' to enable access to absolute file paths via sample['audio']['path']
    train = train.cast_column("audio", Audio(sampling_rate=32_000))

    # Map class id -> ebird code string
    code_list = train.features["ebird_code"].names  # index -> 'grycat', etc.

    # Prepare output
    print("Writing Output")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "",
                "path",
                "start",
                "end",
                "scientific_name",
                "common_name",
                "confidence",
                "group_key",
                "primary_label",
            ]
        )

        row_id = 0
        half = args.win / 2.0

        for ex in train:
            # Absolute file path for your pipeline
            # Prefer the actual cached file path (Audio column), fall back to 'filepath'
            audio_path = (
                ex["audio"]["path"]
                if ex.get("audio") and ex["audio"].get("path")
                else ex["filepath"]
            )
            if not audio_path:
                continue

            length_sec = float(ex.get("length") or 0.0)
            code_idx = int(ex["ebird_code"])
            eb_code = code_list[code_idx]  # e.g., 'grycat'

            # Events: list of [start, end] per recording
            events = ex.get("detected_events") or []
            for ev in events:
                if not ev or len(ev) < 2:
                    continue
                s, e = float(ev[0]), float(ev[1])
                if (e - s) < args.min_event:
                    continue
                center = 0.5 * (s + e)
                start, end = clamp_window(
                    center,
                    half,
                    total=length_sec if length_sec > 0 else (center + half),
                )

                sci, com = eb_map.get(eb_code, ("", ""))
                # Confidence is not provided for Train events; set to 1.0 (or tweak if you filter by quality)
                conf = 1.0

                group_key = os.path.basename(audio_path)

                writer.writerow(
                    [
                        row_id,
                        audio_path,
                        f"{start:.3f}",
                        f"{end:.3f}",
                        sci,
                        com,
                        f"{conf:.6f}",
                        group_key,
                        eb_code,
                    ]
                )
                row_id += 1

    print(f"Wrote {row_id} rows to {args.out}")


if __name__ == "__main__":
    main()
