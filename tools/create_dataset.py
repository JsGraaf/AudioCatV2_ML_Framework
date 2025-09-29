#!/usr/bin/env python3
"""
Build a representative dataset for TFLite INT8 calibration.

Source layout (species folders with audio files):
    datasets/custom_set/xc_dataset/
        Pica_pica/
            XC12345.ogg
            ...
        Turdus_merula/
            ...
        ...

Destination:
    datasets/quant_data/
        Pica_pica/
            ...
        Turdus_merula/
            ...
        manifest.csv

By default: copy up to --per_species files per species (balanced), preserving
subfolders by species. Supports copy/symlink/hardlink and multithreaded I/O.
"""

import argparse
import csv
import os
import random
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Tuple

from tqdm import tqdm

AUDIO_EXTS = {".wav", ".ogg", ".flac", ".mp3", ".m4a", ".aac", ".wma"}


def find_species_dirs(root: Path) -> List[Path]:
    return [p for p in sorted(root.iterdir()) if p.is_dir()]


def find_audio_files(species_dir: Path) -> List[Path]:
    files = []
    for p in species_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            files.append(p)
    return sorted(files)


def plan_selection(
    src_root: Path,
    per_species: int | None,
    total: int | None,
    seed: int,
    strategy: str,
) -> List[Tuple[str, Path]]:
    """
    Returns a list of (species_name, src_file) pairs to include.
    strategy: 'balanced' (default) or 'proportional' for --total.
    """
    rng = random.Random(seed)
    species_dirs = find_species_dirs(src_root)

    species_files = []
    for sd in species_dirs:
        files = find_audio_files(sd)
        if files:
            species_files.append((sd.name, files))

    if not species_files:
        raise SystemExit(f"No audio files found under {src_root}")

    plan: List[Tuple[str, Path]] = []

    if per_species is not None:
        # Balanced: up to N per species
        for sp, files in species_files:
            pick = files[:]  # copy list
            rng.shuffle(pick)
            plan.extend((sp, f) for f in pick[:per_species])
        return plan

    # total requested
    assert total is not None
    if strategy == "balanced":
        # round-robin sample until we hit total
        per_bucket = max(1, total // len(species_files))
        tmp: List[Tuple[str, Path]] = []
        for sp, files in species_files:
            pick = files[:]
            rng.shuffle(pick)
            tmp.extend((sp, f) for f in pick[:per_bucket])
        # if we need more (due to integer division), fill remainder from the largest pools
        remaining = max(0, total - len(tmp))
        if remaining > 0:
            # Flatten leftovers
            leftovers: List[Tuple[str, Path]] = []
            for sp, files in species_files:
                # Take from the remainder of each shuffled list
                take_from = files[per_bucket:]
                rng.shuffle(take_from)
                leftovers.extend((sp, f) for f in take_from)
            rng.shuffle(leftovers)
            tmp.extend(leftovers[:remaining])
        plan = tmp[:total]
        return plan

    elif strategy == "proportional":
        # Sample proportional to each species pool size
        all_files = [(sp, f) for sp, files in species_files for f in files]
        rng.shuffle(all_files)
        plan = all_files[:total]
        return plan

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def copy_one(
    src: Path, dst: Path, mode: str = "copy", overwrite: bool = False
) -> Tuple[bool, str, int]:
    """
    Copy/link one file. Returns (ok, errmsg, bytes_copied_or_0).
    """
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            if not overwrite:
                return True, "exists", 0
            else:
                dst.unlink()

        if mode == "copy":
            shutil.copy2(src, dst)
        elif mode == "symlink":
            os.symlink(src.resolve(), dst)
        elif mode == "hardlink":
            os.link(src.resolve(), dst)
        else:
            return False, f"invalid mode {mode}", 0

        nbytes = src.stat().st_size if dst.exists() else 0
        return True, "", nbytes
    except Exception as e:
        return False, str(e), 0


def build_quant_set(
    src_root: Path,
    dst_root: Path,
    per_species: int | None,
    total: int | None,
    seed: int,
    strategy: str,
    mode: str,
    overwrite: bool,
    workers: int,
) -> Path:
    """
    Build datasets/quant_data with selected audio and a manifest.csv.
    """
    selection = plan_selection(src_root, per_species, total, seed, strategy)

    # Manifest path
    manifest = dst_root / "manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)

    # Multithreaded transfer
    tasks = []
    with ThreadPoolExecutor(max_workers=workers) as ex, open(
        manifest, "w", newline=""
    ) as mf:
        writer = csv.DictWriter(
            mf,
            fieldnames=[
                "species",
                "src_path",
                "dst_path",
                "bytes",
                "status",
                "message",
            ],
        )
        writer.writeheader()

        for species, src_file in selection:
            rel_name = src_file.name
            dst_file = dst_root / species / rel_name
            tasks.append(
                (
                    species,
                    src_file,
                    dst_file,
                    ex.submit(copy_one, src_file, dst_file, mode, overwrite),
                )
            )

        ok_count = 0
        total_bytes = 0

        for species, src_file, dst_file, fut in tqdm(tasks, desc="Transferring files"):
            ok, msg, nbytes = fut.result()
            writer.writerow(
                {
                    "species": species,
                    "src_path": str(src_file),
                    "dst_path": str(dst_file),
                    "bytes": nbytes,
                    "status": "ok" if ok else "fail",
                    "message": msg,
                }
            )
            if ok:
                ok_count += 1
                total_bytes += nbytes

    print(f"[INFO] Wrote manifest: {manifest}")
    print(
        f"[INFO] Completed: {ok_count}/{len(selection)} files. Total size ~{total_bytes/1_048_576:.2f} MB"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Prepare a representative dataset by sampling audio from species folders."
    )
    parser.add_argument(
        "--src",
        type=str,
        default="datasets/custom_set/xc_dataset",
        help="Source root with species subfolders",
    )
    parser.add_argument(
        "--dst",
        type=str,
        default="datasets/quant_data",
        help="Destination representative dataset folder",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--per_species",
        type=int,
        help="Number of files to sample per species (balanced).",
    )
    group.add_argument(
        "--total",
        type=int,
        help="Total number of files to sample overall.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="balanced",
        choices=["balanced", "proportional"],
        help="Sampling strategy when using --total.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="copy",
        choices=["copy", "symlink", "hardlink"],
        help="How to materialize files in the destination.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination files if they already exist.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Max concurrent file operations.",
    )
    args = parser.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)

    if not src_root.exists():
        raise SystemExit(f"Source not found: {src_root}")

    print(f"[INFO] Source:      {src_root}")
    print(f"[INFO] Destination: {dst_root}")
    if args.per_species is not None:
        print(f"[INFO] Sampling:    {args.per_species} per species (balanced)")
    else:
        print(f"[INFO] Sampling:    total={args.total} strategy={args.strategy}")
    print(
        f"[INFO] Mode:        {args.mode}  | overwrite={args.overwrite} | workers={args.workers} | seed={args.seed}"
    )

    build_quant_set(
        src_root=src_root,
        dst_root=dst_root,
        per_species=args.per_species,
        total=args.total,
        seed=args.seed,
        strategy=args.strategy,
        mode=args.mode,
        overwrite=args.overwrite,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
