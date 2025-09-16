#!/usr/bin/env python3
import argparse
import csv
import os
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from tqdm import tqdm

from datasets import Audio, load_dataset  # pip install datasets


# ---------- Optional taxonomy mapping ----------
# Accepts a CSV with headers like:
#   ebird_code,scientific_name,common_name
# or eBird official columns:
#   SPECIES_CODE, SCI_NAME, PRIMARY_COM_NAME
def load_taxonomy_map(path: Path) -> Dict[str, Tuple[str, str]]:
    mapping: Dict[str, Tuple[str, str]] = {}
    if not path:
        return mapping
    if not path.exists():
        warnings.warn(f"Taxonomy CSV not found: {path}")
        return mapping
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            code = (
                r.get("ebird_code")
                or r.get("SPECIES_CODE")
                or r.get("species_code")
                or r.get("Species Code")
            )
            sci = (
                r.get("scientific_name")
                or r.get("SCI_NAME")
                or r.get("scientificName")
                or r.get("Scientific Name")
                or ""
            )
            com = (
                r.get("common_name")
                or r.get("PRIMARY_COM_NAME")
                or r.get("englishName")
                or r.get("Common Name")
                or ""
            )
            if code:
                mapping[code.strip().lower()] = (sci.strip(), com.strip())
    return mapping


# ---------- Event iterators for different schemas ----------
def iter_events_full_test(
    ex, code_list: List[str]
) -> Iterable[Tuple[float, float, str, float]]:
    """
    Yields (start, end, ebird_code, confidence) for a 'test' split example.
    Supports either:
      - flattened per-event rows (fields on the example), or
      - 'annotations' list of dicts.
    """
    # Case A: per-example has a single bbox (flattened)
    s = ex.get("start_time")
    e = ex.get("end_time")
    if s is not None and e is not None:
        # ebird_code might be int index (ClassLabel) or str
        label = ex.get("ebird_code")
        if isinstance(label, int) and code_list:
            label = code_list[label]
        label = (label or "").lower()
        conf = float(ex.get("score") or ex.get("confidence") or 1.0)
        yield float(s), float(e), label, conf
        return

    # Case B: annotation list
    anns = ex.get("annotations") or []
    for a in anns:
        s = a.get("start_time", 0.0)
        e = a.get("end_time", 0.0)
        label = (a.get("ebird_code") or a.get("label") or "").lower()
        conf = float(a.get("score") or a.get("confidence") or 1.0)
        yield float(s), float(e), label, conf


def iter_events_test_5s(
    ex, code_list: List[str]
) -> Iterable[Tuple[float, float, str, float]]:
    """
    Yields one row per (segment, label) from 'test_5s'.
    'ebird_code_multilabel' is typically a list of class indices.
    """
    start = float(ex.get("start_time") or 0.0)
    end = float(ex.get("end_time") or (start + 5.0))
    labels = ex.get("ebird_code_multilabel") or []
    # Convert indices → codes, or accept already-strings
    if labels and isinstance(labels[0], int) and code_list:
        codes = [code_list[i] for i in labels]
    else:
        codes = [str(l).lower() for l in labels]
    for code in codes:
        yield start, end, code.lower(), 1.0


def main():
    ap = argparse.ArgumentParser(
        description="Export BirdSet HSN soundscapes to BirdCLEF-style CSV."
    )
    ap.add_argument(
        "--config",
        default="HSN",
        choices=["HSN", "HSN_scape"],
        help="BirdSet configuration. Use HSN_scape for soundscapes-only.",
    )
    ap.add_argument(
        "--split",
        default="test",
        choices=["test", "test_5s"],
        help="Which split to export (full-length test with strong labels, or 5s segments).",
    )
    ap.add_argument(
        "--cache_dir",
        default="cache/huggingface",
        help="HuggingFace datasets cache directory.",
    )
    ap.add_argument(
        "--sr",
        type=int,
        default=32000,
        help="Resample rate for Audio casting (for array access); paths are still valid regardless.",
    )
    ap.add_argument(
        "--tax_csv",
        type=Path,
        default=None,
        help="Optional taxonomy CSV to fill scientific/common names.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("output/hsn_soundscapes.csv"),
        help="Output CSV file path.",
    )
    args = ap.parse_args()

    # Load taxonomy map if provided
    tax_map = load_taxonomy_map(args.tax_csv)

    # Load dataset
    ds = load_dataset(
        "DBD-research-group/BirdSet",
        args.config,
        trust_remote_code=True,
        cache_dir=args.cache_dir,
    )

    if args.split not in ds:
        raise SystemExit(
            f"Split '{args.split}' not available for config '{args.config}'. Available: {list(ds.keys())}"
        )

    split = ds[args.split]
    # Cast audio to get decoded arrays and *filesystem paths*
    split = split.cast_column("audio", Audio(sampling_rate=args.sr))

    # Class label names if present (for index→code)
    code_list = None
    if "ebird_code" in split.features and hasattr(
        split.features["ebird_code"], "names"
    ):
        code_list = split.features["ebird_code"].names
    elif "ebird_code_multilabel" in split.features and hasattr(
        split.features["ebird_code_multilabel"], "feature"
    ):
        # MultiHot labels: ClassLabel as feature
        feat = split.features["ebird_code_multilabel"].feature
        if hasattr(feat, "names"):
            code_list = feat.names

    # Writer
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
        for ex in tqdm(split, desc=f"Exporting {args.split}", unit="example"):
            # Prefer resolved local path from Audio column; fallback to 'filepath'
            audio_path = None
            if isinstance(ex.get("audio"), dict):
                audio_path = ex["audio"].get("path") or None
            if not audio_path:
                audio_path = ex.get("filepath") or ""

            if not audio_path:
                continue

            basename = os.path.basename(audio_path)

            # Choose iterator based on split
            if args.split == "test_5s":
                events_iter = iter_events_test_5s(ex, code_list or [])
            else:
                events_iter = iter_events_full_test(ex, code_list or [])

            for start, end, eb_code, conf in events_iter:
                if not eb_code:
                    continue
                sci, com = tax_map.get(eb_code.lower(), ("", ""))
                writer.writerow(
                    [
                        row_id,
                        audio_path,
                        f"{float(start):.3f}",
                        f"{float(end):.3f}",
                        sci,
                        com,
                        f"{float(conf):.6f}",
                        basename,
                        eb_code.lower(),
                    ]
                )
                row_id += 1

    print(f"Wrote CSV with soundscape events: {args.out}")


if __name__ == "__main__":
    main()
