"""
Practical dataset-management helper for the ML phase.

Subcommands:
- import: copy external images/masks into dataset/images and dataset/masks
- prepare-photosketch: convert the Photo-Sketching dataset into binary masks and manifests
- bootstrap: create draft masks from raw images using classical CV
- audit: inspect pair quality and write a dataset report
- split: create train/val/test manifest files for repeatable experiments
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import random
import re
import shutil

import cv2

from src.ml_dataset_utils import (
    audit_dataset,
    binarize_line_art,
    generate_classical_bootstrap_mask,
    list_supported_files,
    list_supported_files_recursive,
    match_image_mask_pairs,
    save_manifest_pairs,
    write_bootstrap_preview,
)


VALID_SPLIT_NAMES = ("train", "val", "test")


def sanitize_stem(raw: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._")
    return cleaned or "sample"


def normalize_stem(value: str) -> str:
    return Path(str(value).strip()).stem.lower()


def normalize_split_name(raw: str) -> str | None:
    mapping = {
        "train": "train",
        "training": "train",
        "tr": "train",
        "val": "val",
        "valid": "val",
        "validation": "val",
        "dev": "val",
        "test": "test",
        "testing": "test",
        "te": "test",
    }
    return mapping.get(raw.strip().lower())


def tokenize_identifiers(raw: str) -> list[str]:
    return [token for token in re.split(r"[\s,;]+", raw.strip()) if token]


def flatten_identifier_values(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, int, float)):
        return [str(value)]
    if isinstance(value, list):
        flattened: list[str] = []
        for item in value:
            flattened.extend(flatten_identifier_values(item))
        return flattened
    if isinstance(value, dict):
        for key in ("id", "image", "image_id", "stem", "name", "file", "filename"):
            if key in value:
                return flatten_identifier_values(value[key])
        flattened = []
        for item in value.values():
            flattened.extend(flatten_identifier_values(item))
        return flattened
    return [str(value)]


def infer_split_from_filename(path: Path) -> str | None:
    parts = re.split(r"[^A-Za-z0-9]+", path.stem.lower())
    for part in parts:
        split_name = normalize_split_name(part)
        if split_name is not None:
            return split_name
    return None


def resolve_existing_path(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resolve_photosketch_inputs(args: argparse.Namespace) -> tuple[Path, Path, list[Path]]:
    source_root = Path(args.source_root) if args.source_root else None

    if args.source_images_dir:
        images_dir = Path(args.source_images_dir)
    elif source_root is not None:
        images_dir = resolve_existing_path(
            [
                source_root / "image",
                source_root / "images",
            ]
        )
        if images_dir is None:
            raise FileNotFoundError("Could not auto-detect Photo-Sketching images directory under --source-root.")
    else:
        raise ValueError("Either --source-root or --source-images-dir is required.")

    if args.source_sketches_dir:
        sketches_dir = Path(args.source_sketches_dir)
    elif source_root is not None:
        sketches_dir = resolve_existing_path(
            [
                source_root / "png",
                source_root / "rendered_sketch",
                source_root / "rendered_sketch_png",
                source_root / "sketch",
                source_root / "sketches",
            ]
        )
        if sketches_dir is None:
            raise FileNotFoundError("Could not auto-detect Photo-Sketching rendered sketch directory under --source-root.")
    else:
        raise ValueError("Either --source-root or --source-sketches-dir is required.")

    split_files: list[Path] = []
    for value in args.split_file:
        path = Path(value)
        if path.is_dir():
            split_files.extend(sorted(path.glob("*")))
        else:
            split_files.append(path)

    if not split_files and source_root is not None:
        detected_split_path = resolve_existing_path(
            [
                source_root / "split",
                source_root / "splits",
                source_root / "dataset_split",
            ]
        )
        if detected_split_path is not None:
            if detected_split_path.is_dir():
                split_files.extend(sorted(detected_split_path.glob("*")))
            else:
                split_files.append(detected_split_path)

    split_files = [path for path in split_files if path.suffix.lower() in {".txt", ".json", ".csv", ".tsv"}]
    return images_dir, sketches_dir, split_files


def parse_split_files(paths: list[Path]) -> dict[str, str]:
    assignments: dict[str, str] = {}

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Split file not found: {path}")

        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                handled = False
                for key, value in payload.items():
                    split_name = normalize_split_name(str(key))
                    if split_name is None:
                        continue
                    for identifier in flatten_identifier_values(value):
                        assignments[normalize_stem(identifier)] = split_name
                    handled = True
                if handled:
                    continue
            elif isinstance(payload, list):
                handled = False
                for record in payload:
                    if not isinstance(record, dict):
                        continue
                    split_name = normalize_split_name(str(record.get("split", "")))
                    if split_name is None:
                        continue
                    identifier_values = flatten_identifier_values(
                        record.get("id")
                        or record.get("image")
                        or record.get("image_id")
                        or record.get("stem")
                        or record.get("file")
                        or record.get("filename")
                    )
                    for identifier in identifier_values:
                        assignments[normalize_stem(identifier)] = split_name
                    handled = True
                if handled:
                    continue
            raise ValueError(f"Unsupported JSON split format: {path}")

        inferred_split = infer_split_from_filename(path)
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if ":" in line:
                left, right = line.split(":", 1)
                split_name = normalize_split_name(left)
                if split_name is not None:
                    for identifier in tokenize_identifiers(right):
                        assignments[normalize_stem(identifier)] = split_name
                    continue

            parts = tokenize_identifiers(line)
            if not parts:
                continue

            first_split = normalize_split_name(parts[0])
            last_split = normalize_split_name(parts[-1])

            if first_split is not None and len(parts) >= 2:
                for identifier in parts[1:]:
                    assignments[normalize_stem(identifier)] = first_split
                continue

            if last_split is not None and len(parts) >= 2:
                for identifier in parts[:-1]:
                    assignments[normalize_stem(identifier)] = last_split
                continue

            if inferred_split is not None:
                for identifier in parts:
                    assignments[normalize_stem(identifier)] = inferred_split
                continue

            raise ValueError(f"Could not infer split assignment from line in {path}: {raw_line}")

    return assignments


def choose_sketch_group_key(sketch_stem: str, image_stems_sorted: list[str], image_stem_set: set[str]) -> str | None:
    sketch_key = sketch_stem.lower()
    if sketch_key in image_stem_set:
        return sketch_key

    for image_stem in image_stems_sorted:
        if not sketch_key.startswith(image_stem):
            continue
        remainder = sketch_key[len(image_stem) :]
        if not remainder:
            return image_stem
        if remainder[0] in "_-. ":
            return image_stem

    return None


def group_aware_random_split(
    image_stems: list[str],
    seed: int,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, str]:
    if val_ratio < 0.0 or test_ratio < 0.0 or (val_ratio + test_ratio) >= 1.0:
        raise ValueError("val_ratio and test_ratio must be >= 0 and sum to less than 1.")

    shuffled = image_stems[:]
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    test_count = int(total * test_ratio)
    val_count = int(total * val_ratio)
    train_count = total - val_count - test_count

    assignments: dict[str, str] = {}
    for image_stem in shuffled[:train_count]:
        assignments[image_stem] = "train"
    for image_stem in shuffled[train_count : train_count + val_count]:
        assignments[image_stem] = "val"
    for image_stem in shuffled[train_count + val_count :]:
        assignments[image_stem] = "test"
    return assignments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage the robot-drawing ML dataset.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    import_parser = subparsers.add_parser("import", help="Copy external images and optional masks into dataset/")
    import_parser.add_argument("--source-images-dir", type=str, required=True, help="External source image directory")
    import_parser.add_argument("--source-masks-dir", type=str, default="", help="Optional external source mask directory")
    import_parser.add_argument("--images-dir", type=str, default="dataset/images", help="Destination images directory")
    import_parser.add_argument("--masks-dir", type=str, default="dataset/masks", help="Destination masks directory")
    import_parser.add_argument("--prefix", type=str, default="", help="Optional prefix added to imported stems")
    import_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing destination files")
    import_parser.add_argument("--dry-run", action="store_true", help="Preview actions without copying files")
    import_parser.add_argument(
        "--report-path",
        type=str,
        default="output/ml_dataset/import_report.json",
        help="JSON report path for the import summary",
    )

    photosketch_parser = subparsers.add_parser(
        "prepare-photosketch",
        help="Prepare the Photo-Sketching dataset into binary masks and manifests",
    )
    photosketch_parser.add_argument(
        "--source-root",
        type=str,
        default="",
        help="Optional Photo-Sketching root folder; auto-detects image/png/split children",
    )
    photosketch_parser.add_argument(
        "--source-images-dir",
        type=str,
        default="",
        help="Folder containing the Photo-Sketching source images",
    )
    photosketch_parser.add_argument(
        "--source-sketches-dir",
        type=str,
        default="",
        help="Folder containing rendered sketch PNGs",
    )
    photosketch_parser.add_argument(
        "--split-file",
        type=str,
        action="append",
        default=[],
        help="Optional split file or split directory; may be provided multiple times",
    )
    photosketch_parser.add_argument(
        "--output-dir",
        type=str,
        default="output/photosketch_prepared",
        help="Output folder for prepared masks, manifests, and reports",
    )
    photosketch_parser.add_argument(
        "--prefix",
        type=str,
        default="photosketch_",
        help="Prefix used for generated mask stems",
    )
    photosketch_parser.add_argument(
        "--limit-per-image",
        type=int,
        default=0,
        help="Optional maximum number of sketches kept per source image (0 keeps all)",
    )
    photosketch_parser.add_argument(
        "--sketch-threshold",
        type=int,
        default=245,
        help="Grayscale threshold used when binarizing rendered sketches",
    )
    photosketch_parser.add_argument(
        "--sketch-mode",
        type=str,
        choices=["auto", "dark_on_light", "light_on_dark"],
        default="auto",
        help="Whether foreground lines are dark or light in the rendered sketch PNGs",
    )
    photosketch_parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fallback validation ratio when no split file is available",
    )
    photosketch_parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Fallback test ratio when no split file is available",
    )
    photosketch_parser.add_argument("--seed", type=int, default=42, help="Shuffle seed for fallback split creation")
    photosketch_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing prepared masks")
    photosketch_parser.add_argument("--dry-run", action="store_true", help="Preview actions without writing files")
    photosketch_parser.add_argument(
        "--report-path",
        type=str,
        default="",
        help="Optional JSON report path; defaults to <output-dir>/photosketch_prepare_report.json",
    )

    bootstrap_parser = subparsers.add_parser(
        "bootstrap",
        help="Create draft binary masks from raw images using classical CV",
    )
    bootstrap_parser.add_argument(
        "--source-images-dir",
        type=str,
        default="dataset/images",
        help="Image directory used to generate draft masks",
    )
    bootstrap_parser.add_argument(
        "--masks-dir",
        type=str,
        default="dataset/masks",
        help="Destination directory for generated draft masks",
    )
    bootstrap_parser.add_argument(
        "--preview-dir",
        type=str,
        default="output/ml_dataset/bootstrap/previews",
        help="Directory for side-by-side preview images",
    )
    bootstrap_parser.add_argument("--blur-kernel", type=int, default=3, help="Gaussian blur kernel size")
    bootstrap_parser.add_argument("--canny-low", type=int, default=80, help="Low threshold for Canny")
    bootstrap_parser.add_argument("--canny-high", type=int, default=160, help="High threshold for Canny")
    bootstrap_parser.add_argument("--close-iters", type=int, default=1, help="Morphological close iterations")
    bootstrap_parser.add_argument("--dilate-iters", type=int, default=0, help="Mask dilation iterations")
    bootstrap_parser.add_argument(
        "--min-component-area",
        type=int,
        default=16,
        help="Drop connected components smaller than this area",
    )
    bootstrap_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing draft masks")
    bootstrap_parser.add_argument("--dry-run", action="store_true", help="Preview actions without writing files")
    bootstrap_parser.add_argument(
        "--report-path",
        type=str,
        default="output/ml_dataset/bootstrap/bootstrap_report.json",
        help="JSON report path for the bootstrap summary",
    )

    audit_parser = subparsers.add_parser("audit", help="Validate the current dataset")
    audit_parser.add_argument("--images-dir", type=str, default="dataset/images", help="Dataset images directory")
    audit_parser.add_argument("--masks-dir", type=str, default="dataset/masks", help="Dataset masks directory")
    audit_parser.add_argument(
        "--report-path",
        type=str,
        default="output/ml_dataset/dataset_audit.json",
        help="JSON report path for the audit summary",
    )

    split_parser = subparsers.add_parser("split", help="Create train/val/test manifests")
    split_parser.add_argument("--images-dir", type=str, default="dataset/images", help="Dataset images directory")
    split_parser.add_argument("--masks-dir", type=str, default="dataset/masks", help="Dataset masks directory")
    split_parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio")
    split_parser.add_argument("--test-ratio", type=float, default=0.0, help="Test ratio")
    split_parser.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    split_parser.add_argument(
        "--output-dir",
        type=str,
        default="output/ml_dataset/splits",
        help="Folder for generated manifests",
    )

    return parser.parse_args()


def write_csv_report(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def handle_prepare_photosketch(args: argparse.Namespace) -> None:
    images_dir, sketches_dir, split_files = resolve_photosketch_inputs(args)
    output_dir = Path(args.output_dir)
    masks_dir = output_dir / "masks"
    manifests_dir = output_dir / "splits"
    report_path = Path(args.report_path) if args.report_path else output_dir / "photosketch_prepare_report.json"

    image_files = list_supported_files_recursive(images_dir)
    sketch_files = list_supported_files_recursive(sketches_dir)
    if not image_files:
        raise ValueError(f"No supported images found in Photo-Sketching images directory: {images_dir}")
    if not sketch_files:
        raise ValueError(f"No supported sketches found in Photo-Sketching sketches directory: {sketches_dir}")

    image_by_stem = {path.stem.lower(): path for path in image_files}
    image_stems_sorted = sorted(image_by_stem.keys(), key=len, reverse=True)
    image_stem_set = set(image_by_stem)

    split_assignments = parse_split_files(split_files) if split_files else {}
    if not split_assignments:
        split_assignments = group_aware_random_split(
            image_stems=sorted(image_by_stem),
            seed=args.seed,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
        )
        split_source = "fallback_random_split"
    else:
        split_source = "source_split_file"

    kept_by_image: dict[str, int] = defaultdict(int)
    unmatched_sketches: list[dict] = []
    prepared_rows: list[dict] = []
    split_pairs: dict[str, list[tuple[Path, Path]]] = {name: [] for name in VALID_SPLIT_NAMES}
    generated_masks = 0
    skipped_existing = 0

    for sketch_path in sketch_files:
        image_stem = choose_sketch_group_key(sketch_path.stem, image_stems_sorted=image_stems_sorted, image_stem_set=image_stem_set)
        if image_stem is None:
            unmatched_sketches.append({"sketch": str(sketch_path), "reason": "no_matching_image"})
            continue

        if args.limit_per_image > 0 and kept_by_image[image_stem] >= args.limit_per_image:
            continue

        image_path = image_by_stem[image_stem]
        split_name = split_assignments.get(image_stem, "train")
        if split_name not in VALID_SPLIT_NAMES:
            unmatched_sketches.append({"sketch": str(sketch_path), "reason": f"unsupported_split:{split_name}"})
            continue

        dest_stem = sanitize_stem(f"{args.prefix}{image_path.stem}__{sketch_path.stem}")
        dest_mask = masks_dir / f"{dest_stem}.png"
        action = "generated"

        if dest_mask.exists() and not args.overwrite:
            skipped_existing += 1
            action = "skipped_existing"
        elif not args.dry_run:
            sketch_gray = cv2.imread(str(sketch_path), cv2.IMREAD_GRAYSCALE)
            if sketch_gray is None:
                unmatched_sketches.append({"sketch": str(sketch_path), "reason": "unreadable_sketch"})
                continue
            binary_mask = binarize_line_art(
                sketch_gray,
                threshold=args.sketch_threshold,
                foreground_mode=args.sketch_mode,
            )
            dest_mask.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dest_mask), binary_mask)
            generated_masks += 1
        else:
            generated_masks += 1
            action = "dry_run_generated"

        split_pairs[split_name].append((image_path.resolve(), dest_mask.resolve()))
        kept_by_image[image_stem] += 1
        prepared_rows.append(
            {
                "split": split_name,
                "image_stem": image_stem,
                "image_path": str(image_path),
                "sketch_path": str(sketch_path),
                "mask_path": str(dest_mask),
                "action": action,
            }
        )

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        manifests_dir.mkdir(parents=True, exist_ok=True)
        for split_name, pairs in split_pairs.items():
            if not pairs:
                continue
            save_manifest_pairs(
                pairs,
                manifests_dir / f"{split_name}_manifest.json",
                split_name=split_name,
                seed=args.seed,
            )

    counts = {
        "source_images": len(image_files),
        "source_sketches": len(sketch_files),
        "prepared_pairs": len(prepared_rows),
        "generated_masks": generated_masks,
        "skipped_existing": skipped_existing,
        "unmatched_sketches": len(unmatched_sketches),
        "distinct_source_images_used": len(kept_by_image),
        "train_pairs": len(split_pairs["train"]),
        "val_pairs": len(split_pairs["val"]),
        "test_pairs": len(split_pairs["test"]),
    }

    train_command = f"python -m src.train --train-manifest {manifests_dir / 'train_manifest.json'} --output models/line_model.pt"
    if split_pairs["val"]:
        train_command += f" --val-manifest {manifests_dir / 'val_manifest.json'}"

    report = {
        "command": "prepare-photosketch",
        "paths": {
            "source_images_dir": str(images_dir),
            "source_sketches_dir": str(sketches_dir),
            "split_files": [str(path) for path in split_files],
            "output_dir": str(output_dir),
            "masks_dir": str(masks_dir),
            "manifests_dir": str(manifests_dir),
        },
        "params": {
            "prefix": args.prefix,
            "limit_per_image": args.limit_per_image,
            "sketch_threshold": args.sketch_threshold,
            "sketch_mode": args.sketch_mode,
            "seed": args.seed,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "overwrite": args.overwrite,
            "dry_run": args.dry_run,
            "split_source": split_source,
        },
        "counts": counts,
        "next_commands": {
            "train": train_command,
        },
        "rows": prepared_rows,
        "unmatched_sketches": unmatched_sketches[:200],
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv_report(
        report_path.with_suffix(".csv"),
        ["split", "image_stem", "image_path", "sketch_path", "mask_path", "action"],
        prepared_rows,
    )

    print(f"Photo-Sketching source images: {len(image_files)}")
    print(f"Photo-Sketching source sketches: {len(sketch_files)}")
    print(f"Prepared pairs: {len(prepared_rows)}")
    print(f"Train pairs: {len(split_pairs['train'])}")
    print(f"Val pairs: {len(split_pairs['val'])}")
    print(f"Test pairs: {len(split_pairs['test'])}")
    print(f"Unmatched sketches: {len(unmatched_sketches)}")
    print(f"Wrote Photo-Sketching report to: {report_path}")


def handle_import(args: argparse.Namespace) -> None:
    source_images_dir = Path(args.source_images_dir)
    source_masks_dir = Path(args.source_masks_dir) if args.source_masks_dir else None
    images_dir = Path(args.images_dir)
    masks_dir = Path(args.masks_dir)
    report_path = Path(args.report_path)

    source_images = list_supported_files(source_images_dir)
    source_mask_by_stem = {}
    if source_masks_dir is not None:
        source_mask_by_stem = {mask_path.stem: mask_path for mask_path in list_supported_files(source_masks_dir)}
    imported_rows: list[dict] = []
    missing_mask_rows: list[dict] = []
    copied_images = 0
    copied_masks = 0
    skipped_existing = 0

    for image_path in source_images:
        dest_stem = sanitize_stem(f"{args.prefix}{image_path.stem}")
        dest_image = images_dir / f"{dest_stem}{image_path.suffix.lower()}"
        source_mask = source_mask_by_stem.get(image_path.stem)
        dest_mask = None if source_mask is None else masks_dir / f"{dest_stem}{source_mask.suffix.lower()}"

        action = "copied"
        if dest_image.exists() and not args.overwrite:
            action = "skipped_existing"
            skipped_existing += 1
        elif not args.dry_run:
            dest_image.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, dest_image)
            copied_images += 1

        if source_mask is not None:
            if dest_mask is not None and dest_mask.exists() and not args.overwrite:
                pass
            elif not args.dry_run and dest_mask is not None:
                dest_mask.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_mask, dest_mask)
                copied_masks += 1
        else:
            missing_mask_rows.append(
                {
                    "stem": dest_stem,
                    "source_image": str(image_path),
                    "expected_mask_destination": str(masks_dir / f"{dest_stem}.png"),
                }
            )

        imported_rows.append(
            {
                "stem": dest_stem,
                "source_image": str(image_path),
                "dest_image": str(dest_image),
                "source_mask": "" if source_mask is None else str(source_mask),
                "dest_mask": "" if dest_mask is None else str(dest_mask),
                "action": action,
            }
        )

    report = {
        "command": "import",
        "source_images_dir": str(source_images_dir),
        "source_masks_dir": "" if source_masks_dir is None else str(source_masks_dir),
        "images_dir": str(images_dir),
        "masks_dir": str(masks_dir),
        "prefix": args.prefix,
        "dry_run": args.dry_run,
        "counts": {
            "source_images": len(source_images),
            "copied_images": copied_images,
            "copied_masks": copied_masks,
            "skipped_existing": skipped_existing,
            "missing_masks": len(missing_mask_rows),
        },
        "rows": imported_rows,
        "missing_masks": missing_mask_rows,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv_report(
        report_path.with_suffix(".csv"),
        ["stem", "source_image", "dest_image", "source_mask", "dest_mask", "action"],
        imported_rows,
    )
    if missing_mask_rows:
        write_csv_report(
            report_path.with_name(report_path.stem + "_missing_masks.csv"),
            ["stem", "source_image", "expected_mask_destination"],
            missing_mask_rows,
        )

    print(f"Source images found: {len(source_images)}")
    print(f"Copied images: {copied_images}")
    print(f"Copied masks: {copied_masks}")
    print(f"Missing masks: {len(missing_mask_rows)}")
    print(f"Wrote import report to: {report_path}")


def handle_bootstrap(args: argparse.Namespace) -> None:
    source_images_dir = Path(args.source_images_dir)
    masks_dir = Path(args.masks_dir)
    preview_dir = Path(args.preview_dir)
    report_path = Path(args.report_path)

    source_images = list_supported_files(source_images_dir)
    generated_rows: list[dict] = []
    generated_count = 0
    skipped_existing = 0
    failed = 0

    for image_path in source_images:
        dest_mask = masks_dir / f"{image_path.stem}.png"
        preview_path = preview_dir / f"{image_path.stem}.png"

        if dest_mask.exists() and not args.overwrite:
            skipped_existing += 1
            generated_rows.append(
                {
                    "stem": image_path.stem,
                    "source_image": str(image_path),
                    "dest_mask": str(dest_mask),
                    "preview": str(preview_path),
                    "action": "skipped_existing",
                    "white_ratio": None,
                    "error": "",
                }
            )
            continue

        try:
            mask = generate_classical_bootstrap_mask(
                image_path=image_path,
                blur_kernel=args.blur_kernel,
                canny_low=args.canny_low,
                canny_high=args.canny_high,
                close_iters=args.close_iters,
                dilate_iters=args.dilate_iters,
                min_component_area=args.min_component_area,
            )
            white_ratio = round(float((mask > 127).sum() / max(1, mask.size)), 4)

            if not args.dry_run:
                dest_mask.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(dest_mask), mask)
                write_bootstrap_preview(image_path=image_path, mask=mask, output_path=preview_path)
            generated_count += 1
            generated_rows.append(
                {
                    "stem": image_path.stem,
                    "source_image": str(image_path),
                    "dest_mask": str(dest_mask),
                    "preview": str(preview_path),
                    "action": "generated" if not args.dry_run else "dry_run_generated",
                    "white_ratio": white_ratio,
                    "error": "",
                }
            )
        except Exception as exc:
            failed += 1
            generated_rows.append(
                {
                    "stem": image_path.stem,
                    "source_image": str(image_path),
                    "dest_mask": str(dest_mask),
                    "preview": str(preview_path),
                    "action": "failed",
                    "white_ratio": None,
                    "error": str(exc),
                }
            )

    audit_command = (
        f"python -m src.manage_ml_dataset audit --images-dir {source_images_dir} "
        f"--masks-dir {masks_dir}"
    )
    report = {
        "command": "bootstrap",
        "source_images_dir": str(source_images_dir),
        "masks_dir": str(masks_dir),
        "preview_dir": str(preview_dir),
        "dry_run": args.dry_run,
        "params": {
            "blur_kernel": args.blur_kernel,
            "canny_low": args.canny_low,
            "canny_high": args.canny_high,
            "close_iters": args.close_iters,
            "dilate_iters": args.dilate_iters,
            "min_component_area": args.min_component_area,
            "overwrite": args.overwrite,
        },
        "counts": {
            "source_images": len(source_images),
            "generated_masks": generated_count,
            "skipped_existing": skipped_existing,
            "failed": failed,
        },
        "next_commands": {
            "audit": audit_command,
        },
        "rows": generated_rows,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv_report(
        report_path.with_suffix(".csv"),
        ["stem", "source_image", "dest_mask", "preview", "action", "white_ratio", "error"],
        generated_rows,
    )

    print(f"Source images found: {len(source_images)}")
    print(f"Generated draft masks: {generated_count}")
    print(f"Skipped existing masks: {skipped_existing}")
    print(f"Failed: {failed}")
    print(f"Wrote bootstrap report to: {report_path}")


def handle_audit(args: argparse.Namespace) -> None:
    images_dir = Path(args.images_dir)
    masks_dir = Path(args.masks_dir)
    report_path = Path(args.report_path)

    report = audit_dataset(images_dir=images_dir, masks_dir=masks_dir)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Dataset audit")
    print(f"Images: {report['counts']['image_files']}")
    print(f"Masks: {report['counts']['mask_files']}")
    print(f"Matched pairs: {report['counts']['matched_pairs']}")
    print(f"Exact binary masks: {report['counts']['exact_binary_masks']}")
    print(f"Binary-like masks: {report['counts']['binary_like_masks']}")
    print(f"Suspicious pairs: {report['counts']['suspicious_pairs']}")
    print(f"Wrote audit report to: {report_path}")


def handle_split(args: argparse.Namespace) -> None:
    images_dir = Path(args.images_dir)
    masks_dir = Path(args.masks_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.val_ratio < 0.0 or args.test_ratio < 0.0 or (args.val_ratio + args.test_ratio) >= 1.0:
        raise ValueError("val_ratio and test_ratio must be >= 0 and sum to less than 1.")

    pairs, _, _ = match_image_mask_pairs(images_dir=images_dir, masks_dir=masks_dir)
    if not pairs:
        raise ValueError("No matched image/mask pairs were found; cannot create splits.")

    shuffled = pairs[:]
    random.Random(args.seed).shuffle(shuffled)

    total = len(shuffled)
    test_count = int(total * args.test_ratio)
    val_count = int(total * args.val_ratio)
    train_count = total - val_count - test_count

    train_pairs = shuffled[:train_count]
    val_pairs = shuffled[train_count : train_count + val_count]
    test_pairs = shuffled[train_count + val_count :]

    train_manifest_path = output_dir / "train_manifest.json"
    val_manifest_path = output_dir / "val_manifest.json"
    test_manifest_path = output_dir / "test_manifest.json"

    save_manifest_pairs(train_pairs, train_manifest_path, split_name="train", seed=args.seed)
    if val_pairs:
        save_manifest_pairs(val_pairs, val_manifest_path, split_name="val", seed=args.seed)
    if test_pairs:
        save_manifest_pairs(test_pairs, test_manifest_path, split_name="test", seed=args.seed)

    train_command = f"python -m src.train --train-manifest {train_manifest_path} --output models/line_model.pt"
    if val_pairs:
        train_command += f" --val-manifest {val_manifest_path}"

    split_summary = {
        "seed": args.seed,
        "paths": {
            "images_dir": str(images_dir),
            "masks_dir": str(masks_dir),
            "output_dir": str(output_dir),
        },
        "counts": {
            "total_pairs": total,
            "train_pairs": len(train_pairs),
            "val_pairs": len(val_pairs),
            "test_pairs": len(test_pairs),
        },
        "manifests": {
            "train": str(train_manifest_path),
            "val": "" if not val_pairs else str(val_manifest_path),
            "test": "" if not test_pairs else str(test_manifest_path),
        },
        "next_commands": {
            "train": train_command,
        },
    }

    (output_dir / "split_summary.json").write_text(json.dumps(split_summary, indent=2), encoding="utf-8")

    print(f"Total pairs: {total}")
    print(f"Train pairs: {len(train_pairs)}")
    print(f"Val pairs: {len(val_pairs)}")
    print(f"Test pairs: {len(test_pairs)}")
    print(f"Wrote split summary to: {output_dir / 'split_summary.json'}")


def main() -> None:
    args = parse_args()
    if args.command == "import":
        handle_import(args)
        return
    if args.command == "prepare-photosketch":
        handle_prepare_photosketch(args)
        return
    if args.command == "bootstrap":
        handle_bootstrap(args)
        return
    if args.command == "audit":
        handle_audit(args)
        return
    if args.command == "split":
        handle_split(args)
        return
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
