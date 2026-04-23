"""
Prepare a logo-focused dataset for fine-tuning with minimal manual steps.

This command is intentionally orchestration-first:
- import a public or custom logo dataset into dedicated repo folders
- optionally bootstrap missing masks with the classical CV pipeline
- audit the result
- create fixed train/val/test manifests
- print the exact train / infer / evaluation commands to run next
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import cv2

from src.manage_ml_dataset import handle_audit, handle_bootstrap, handle_import, handle_split, sanitize_stem
from src.ml_dataset_utils import convert_binary_mask_to_contour_mask, list_supported_files


DEFAULT_RESUME_FROM = "models/line_model_photosketch_w5_aspect_overnight.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a logo dataset for ML fine-tuning.")
    parser.add_argument("--source-images-dir", type=str, required=True, help="Downloaded logo image directory")
    parser.add_argument("--source-masks-dir", type=str, default="", help="Optional downloaded logo mask directory")
    parser.add_argument("--dataset-name", type=str, default="logo", help="Short label used in reports and checkpoints")
    parser.add_argument("--images-dir", type=str, default="dataset/logo_images", help="Repo image destination")
    parser.add_argument("--masks-dir", type=str, default="dataset/logo_masks", help="Repo mask destination")
    parser.add_argument("--output-root", type=str, default="output/logo_ml", help="Report and split output folder")
    parser.add_argument("--prefix", type=str, default="", help="Optional prefix added to imported stems")
    parser.add_argument("--recursive", dest="recursive", action="store_true", default=True, help="Recursively scan nested source folders")
    parser.add_argument("--no-recursive", dest="recursive", action="store_false", help="Only scan the top level of source folders")
    parser.add_argument(
        "--mask-mode",
        choices=["copy", "contours"],
        default="copy",
        help="Whether imported masks should stay as-is or be converted to contour masks",
    )
    parser.add_argument("--contour-thickness", type=int, default=1, help="Contour thickness when --mask-mode contours is used")
    parser.add_argument(
        "--bootstrap-missing-masks",
        action="store_true",
        help="Generate draft masks for any images still missing a mask after import",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing imported files and generated masks")
    parser.add_argument("--dry-run", action="store_true", help="Preview the import/bootstrap flow without writing files")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed for split creation")
    parser.add_argument("--device", type=str, default="cpu", help="Training device for the suggested commands")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size for the suggested commands")
    parser.add_argument("--quick-epochs", type=int, default=3, help="Epochs for the suggested quick run")
    parser.add_argument("--full-epochs", type=int, default=10, help="Epochs for the suggested full fine-tune")
    parser.add_argument(
        "--resume-from",
        type=str,
        default=DEFAULT_RESUME_FROM,
        help="Checkpoint used to initialize the suggested fine-tuning runs",
    )
    parser.add_argument("--quick-output", type=str, default="", help="Optional quick-run checkpoint path override")
    parser.add_argument("--full-output", type=str, default="", help="Optional full-run checkpoint path override")
    parser.add_argument(
        "--run-quick-train",
        action="store_true",
        help="After preparation, immediately launch the suggested quick fine-tuning run",
    )
    return parser.parse_args()


def build_command(parts: list[str]) -> str:
    return subprocess.list2cmdline(parts)


def build_train_command(
    *,
    images_dir: Path,
    masks_dir: Path,
    train_manifest: Path,
    val_manifest: Path | None,
    epochs: int,
    batch_size: int,
    device: str,
    resume_from: str,
    output_path: Path,
) -> list[str]:
    command = [
        "python",
        "-m",
        "src.train",
        "--images-dir",
        str(images_dir),
        "--masks-dir",
        str(masks_dir),
        "--train-manifest",
        str(train_manifest),
        "--epochs",
        str(int(epochs)),
        "--batch-size",
        str(int(batch_size)),
        "--device",
        device,
        "--output",
        str(output_path),
    ]
    if val_manifest is not None and val_manifest.exists():
        command.extend(["--val-manifest", str(val_manifest)])
    if resume_from:
        command.extend(["--resume-from", resume_from])
    return command


def build_infer_command(checkpoint_path: Path, output_root: Path) -> list[str]:
    return [
        "python",
        "-m",
        "src.infer",
        "--image",
        "images/Testlogo.jpeg",
        "--checkpoint",
        str(checkpoint_path),
        "--output",
        str(output_root / "testlogo_pred_mask.png"),
    ]


def build_evaluate_command(
    *,
    images_dir: Path,
    masks_dir: Path,
    checkpoint_path: Path,
    output_root: Path,
) -> list[str]:
    return [
        "python",
        "-m",
        "src.evaluate_pipeline",
        "--image-dir",
        str(images_dir),
        "--vision-mode",
        "classical",
        "--vision-mode",
        "ml",
        "--model-checkpoint",
        str(checkpoint_path),
        "--masks-dir",
        str(masks_dir),
        "--output-dir",
        str(output_root / "evaluation"),
    ]


def write_summary_markdown(path: Path, summary: dict) -> None:
    lines = [
        "# Logo ML Setup Summary",
        "",
        f"- Dataset name: `{summary['dataset_name']}`",
        f"- Images dir: `{summary['paths']['images_dir']}`",
        f"- Masks dir: `{summary['paths']['masks_dir']}`",
        f"- Output root: `{summary['paths']['output_root']}`",
        f"- Import report: `{summary['paths']['import_report']}`",
        f"- Audit report: `{summary['paths']['audit_report']}`",
        f"- Split summary: `{summary['paths']['split_summary']}`",
        "",
        "## Counts",
        "",
        f"- Imported images: {summary['counts']['imported_images']}",
        f"- Imported masks: {summary['counts']['imported_masks']}",
        f"- Missing masks after import: {summary['counts']['missing_masks_after_import']}",
        f"- Matched pairs after audit: {summary['counts']['matched_pairs_after_audit']}",
        "",
        "## Next Commands",
        "",
    ]

    for label, command in summary["commands"].items():
        lines.extend([f"### {label}", "", "```powershell", command, "```", ""])

    path.write_text("\n".join(lines), encoding="utf-8")


def convert_imported_masks_to_contours(
    *,
    masks_dir: Path,
    contour_thickness: int,
    report_path: Path,
) -> dict:
    converted = 0
    rows: list[dict] = []

    for mask_path in list_supported_files(masks_dir):
        mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_gray is None:
            rows.append({"mask_path": str(mask_path), "action": "unreadable"})
            continue
        contour_mask = convert_binary_mask_to_contour_mask(mask_gray, contour_thickness=contour_thickness)
        cv2.imwrite(str(mask_path), contour_mask)
        converted += 1
        rows.append({"mask_path": str(mask_path), "action": "converted"})

    report = {
        "masks_dir": str(masks_dir),
        "contour_thickness": int(contour_thickness),
        "converted_masks": converted,
        "rows": rows[:200],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def run_logo_setup(args: argparse.Namespace) -> dict:
    dataset_slug = sanitize_stem(args.dataset_name)
    images_dir = Path(args.images_dir)
    masks_dir = Path(args.masks_dir)
    output_root = Path(args.output_root)

    import_report = output_root / "import_report.json"
    mask_conversion_report = output_root / "mask_conversion_report.json"
    bootstrap_report = output_root / "bootstrap" / "bootstrap_report.json"
    audit_report = output_root / "dataset_audit.json"
    split_dir = output_root / "splits"
    split_summary = split_dir / "split_summary.json"
    summary_json = output_root / "logo_setup_summary.json"
    summary_md = output_root / "logo_setup_summary.md"

    quick_output = Path(args.quick_output) if args.quick_output else Path("models") / f"line_model_{dataset_slug}_quick.pt"
    full_output = Path(args.full_output) if args.full_output else Path("models") / f"line_model_{dataset_slug}_finetuned.pt"

    import_args = SimpleNamespace(
        source_images_dir=args.source_images_dir,
        source_masks_dir=args.source_masks_dir,
        images_dir=str(images_dir),
        masks_dir=str(masks_dir),
        prefix=args.prefix,
        recursive=bool(args.recursive),
        overwrite=bool(args.overwrite),
        dry_run=bool(args.dry_run),
        report_path=str(import_report),
    )
    handle_import(import_args)
    import_payload = json.loads(import_report.read_text(encoding="utf-8"))

    if args.dry_run:
        summary = {
            "dataset_name": dataset_slug,
            "paths": {
                "images_dir": str(images_dir),
                "masks_dir": str(masks_dir),
                "output_root": str(output_root),
                "import_report": str(import_report),
                "mask_conversion_report": "",
                "bootstrap_report": "",
                "audit_report": "",
                "split_summary": "",
                "quick_output": str(quick_output),
                "full_output": str(full_output),
            },
            "counts": {
                "imported_images": int(import_payload["counts"]["source_images"]),
                "imported_masks": int(import_payload["counts"]["copied_masks"]),
                "missing_masks_after_import": int(import_payload["counts"]["missing_masks"]),
                "matched_pairs_after_audit": 0,
            },
            "params": {
                "recursive": bool(args.recursive),
                "mask_mode": args.mask_mode,
                "contour_thickness": int(args.contour_thickness),
                "bootstrap_missing_masks": bool(args.bootstrap_missing_masks),
                "overwrite": bool(args.overwrite),
                "dry_run": True,
                "val_ratio": float(args.val_ratio),
                "test_ratio": float(args.test_ratio),
                "seed": int(args.seed),
                "device": args.device,
                "batch_size": int(args.batch_size),
                "quick_epochs": int(args.quick_epochs),
                "full_epochs": int(args.full_epochs),
                "resume_from": args.resume_from,
            },
            "commands": {},
        }
        output_root.mkdir(parents=True, exist_ok=True)
        summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        write_summary_markdown(summary_md, summary)
        return summary

    mask_conversion_payload = None
    if args.mask_mode == "contours":
        mask_conversion_payload = convert_imported_masks_to_contours(
            masks_dir=masks_dir,
            contour_thickness=args.contour_thickness,
            report_path=mask_conversion_report,
        )

    if args.bootstrap_missing_masks:
        bootstrap_args = SimpleNamespace(
            source_images_dir=str(images_dir),
            masks_dir=str(masks_dir),
            preview_dir=str(output_root / "bootstrap" / "previews"),
            blur_kernel=3,
            canny_low=80,
            canny_high=160,
            close_iters=1,
            dilate_iters=0,
            min_component_area=16,
            overwrite=bool(args.overwrite),
            dry_run=bool(args.dry_run),
            report_path=str(bootstrap_report),
        )
        handle_bootstrap(bootstrap_args)

    audit_args = SimpleNamespace(
        images_dir=str(images_dir),
        masks_dir=str(masks_dir),
        report_path=str(audit_report),
    )
    handle_audit(audit_args)

    audit_payload = json.loads(audit_report.read_text(encoding="utf-8"))
    matched_pairs = int(audit_payload["counts"]["matched_pairs"])

    if matched_pairs > 0:
        split_args = SimpleNamespace(
            images_dir=str(images_dir),
            masks_dir=str(masks_dir),
            val_ratio=float(args.val_ratio),
            test_ratio=float(args.test_ratio),
            seed=int(args.seed),
            output_dir=str(split_dir),
        )
        handle_split(split_args)

    train_manifest = split_dir / "train_manifest.json"
    val_manifest = split_dir / "val_manifest.json"

    commands: dict[str, str] = {}
    if matched_pairs > 0 and train_manifest.exists():
        quick_train_cmd = build_train_command(
            images_dir=images_dir,
            masks_dir=masks_dir,
            train_manifest=train_manifest,
            val_manifest=val_manifest if val_manifest.exists() else None,
            epochs=args.quick_epochs,
            batch_size=args.batch_size,
            device=args.device,
            resume_from=args.resume_from,
            output_path=quick_output,
        )
        full_train_cmd = build_train_command(
            images_dir=images_dir,
            masks_dir=masks_dir,
            train_manifest=train_manifest,
            val_manifest=val_manifest if val_manifest.exists() else None,
            epochs=args.full_epochs,
            batch_size=args.batch_size,
            device=args.device,
            resume_from=args.resume_from,
            output_path=full_output,
        )
        commands["quick_train"] = build_command(quick_train_cmd)
        commands["full_train"] = build_command(full_train_cmd)
        commands["check_training_progress"] = "python -m src.check_training_progress"
        commands["quick_infer"] = build_command(build_infer_command(quick_output, output_root))
        commands["quick_evaluate"] = build_command(
            build_evaluate_command(
                images_dir=images_dir,
                masks_dir=masks_dir,
                checkpoint_path=quick_output,
                output_root=output_root,
            )
        )
    elif not args.source_masks_dir and not args.bootstrap_missing_masks:
        commands["bootstrap_missing_masks"] = build_command(
            [
                "python",
                "-m",
                "src.setup_logo_dataset",
                "--source-images-dir",
                args.source_images_dir,
                "--dataset-name",
                args.dataset_name,
                "--images-dir",
                str(images_dir),
                "--masks-dir",
                str(masks_dir),
                "--output-root",
                str(output_root),
                "--recursive" if args.recursive else "--no-recursive",
                "--bootstrap-missing-masks",
            ]
        )

    summary = {
        "dataset_name": dataset_slug,
        "paths": {
            "images_dir": str(images_dir),
            "masks_dir": str(masks_dir),
            "output_root": str(output_root),
            "import_report": str(import_report),
            "mask_conversion_report": str(mask_conversion_report) if mask_conversion_payload is not None else "",
            "bootstrap_report": str(bootstrap_report) if args.bootstrap_missing_masks else "",
            "audit_report": str(audit_report),
            "split_summary": str(split_summary) if split_summary.exists() else "",
            "quick_output": str(quick_output),
            "full_output": str(full_output),
        },
        "counts": {
            "imported_images": int(import_payload["counts"]["copied_images"]),
            "imported_masks": int(import_payload["counts"]["copied_masks"]),
            "missing_masks_after_import": int(import_payload["counts"]["missing_masks"]),
            "matched_pairs_after_audit": matched_pairs,
        },
        "params": {
            "recursive": bool(args.recursive),
            "mask_mode": args.mask_mode,
            "contour_thickness": int(args.contour_thickness),
            "bootstrap_missing_masks": bool(args.bootstrap_missing_masks),
            "overwrite": bool(args.overwrite),
            "dry_run": bool(args.dry_run),
            "val_ratio": float(args.val_ratio),
            "test_ratio": float(args.test_ratio),
            "seed": int(args.seed),
            "device": args.device,
            "batch_size": int(args.batch_size),
            "quick_epochs": int(args.quick_epochs),
            "full_epochs": int(args.full_epochs),
            "resume_from": args.resume_from,
        },
        "commands": commands,
    }

    output_root.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_summary_markdown(summary_md, summary)

    if args.run_quick_train and "quick_train" in commands and not args.dry_run:
        subprocess.run(quick_train_cmd, check=True)

    return summary


def main() -> None:
    args = parse_args()
    summary = run_logo_setup(args)

    print(f"Logo dataset setup summary written to: {Path(args.output_root) / 'logo_setup_summary.md'}")
    print(f"Matched pairs after audit: {summary['counts']['matched_pairs_after_audit']}")
    for label, command in summary["commands"].items():
        print(f"[{label}] {command}")


if __name__ == "__main__":
    main()
