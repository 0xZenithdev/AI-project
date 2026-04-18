"""
Check whether the repository is ready to start the ML phase.

This script is intentionally lightweight: it does not train anything.
It verifies the dataset folders, matched image/mask pairs, and checkpoint
presence so we can answer "are we ready for ML?" with concrete evidence.

Example:
python -m src.check_ml_readiness
python -m src.check_ml_readiness --checkpoint models/line_model.pt --write-report output/ml_readiness.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.ml_dataset_utils import audit_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check ML readiness for the robot-drawing project.")
    parser.add_argument("--images-dir", type=str, default="dataset/images", help="Training images directory")
    parser.add_argument("--masks-dir", type=str, default="dataset/masks", help="Training masks directory")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/line_model.pt",
        help="Checkpoint path to check for inference readiness",
    )
    parser.add_argument(
        "--write-report",
        type=str,
        default="",
        help="Optional JSON report output path",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    images_dir = Path(args.images_dir)
    masks_dir = Path(args.masks_dir)
    checkpoint_path = Path(args.checkpoint)
    audit = audit_dataset(images_dir=images_dir, masks_dir=masks_dir)

    # Memory note:
    # - "code_ready" means the ML path exists in the repo structure
    # - "training_ready" requires at least one matched pair
    # - "inference_ready" requires an actual trained checkpoint
    report = {
        "paths": {
            "images_dir": str(images_dir),
            "masks_dir": str(masks_dir),
            "checkpoint": str(checkpoint_path),
        },
        "exists": {
            "images_dir": images_dir.exists(),
            "masks_dir": masks_dir.exists(),
            "checkpoint": checkpoint_path.exists(),
        },
        "counts": audit["counts"],
        "examples": audit["examples"],
        "status": {
            "code_ready": True,
            "training_ready": audit["counts"]["matched_pairs"] > 0,
            "dataset_quality_ready": (
                audit["counts"]["matched_pairs"] > 0
                and audit["counts"]["binary_like_masks"] == audit["counts"]["matched_pairs"]
                and audit["counts"]["unreadable_pairs"] == 0
            ),
            "inference_ready": checkpoint_path.exists(),
        },
        "dataset_audit": {
            "suspicious_pairs": audit["counts"]["suspicious_pairs"],
            "exact_binary_masks": audit["counts"]["exact_binary_masks"],
            "binary_like_masks": audit["counts"]["binary_like_masks"],
            "same_size_pairs": audit["counts"]["same_size_pairs"],
            "unreadable_pairs": audit["counts"]["unreadable_pairs"],
        },
        "next_step": "",
    }

    if audit["counts"]["matched_pairs"] == 0:
        report["next_step"] = (
            "Create paired files first, or use python -m src.manage_ml_dataset import "
            "to bring source images/masks into dataset/images and dataset/masks."
        )
    elif audit["counts"]["binary_like_masks"] < audit["counts"]["matched_pairs"]:
        report["next_step"] = (
            "Run python -m src.manage_ml_dataset audit and fix the suspicious masks "
            "before training."
        )
    elif not checkpoint_path.exists():
        report["next_step"] = (
            "Create train/val manifests with python -m src.manage_ml_dataset split, "
            "then train the first checkpoint with python -m src.train."
        )
    else:
        report["next_step"] = "Run python -m src.infer or main.py --vision-mode ml with the checkpoint."

    if args.write_report:
        out_path = Path(args.write_report)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("ML readiness report")
    print(f"Images dir exists: {report['exists']['images_dir']} ({images_dir})")
    print(f"Masks dir exists: {report['exists']['masks_dir']} ({masks_dir})")
    print(f"Checkpoint exists: {report['exists']['checkpoint']} ({checkpoint_path})")
    print(f"Image files: {report['counts']['image_files']}")
    print(f"Mask files: {report['counts']['mask_files']}")
    print(f"Matched pairs: {report['counts']['matched_pairs']}")
    print(f"Training ready: {report['status']['training_ready']}")
    print(f"Dataset quality ready: {report['status']['dataset_quality_ready']}")
    print(f"Inference ready: {report['status']['inference_ready']}")
    print(f"Next step: {report['next_step']}")


if __name__ == "__main__":
    main()
