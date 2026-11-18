"""Export reviewed annotations, then fine-tune and validate a collector detector."""

import argparse
import csv
import json
import time
from pathlib import Path

from dataset import prepare_dataset
from vision import COLLECTORS, KINDS


def completed_epochs(folder, fallback, limit):
    """Final validation callbacks can run after the trainer advances its counter."""
    try:
        with (Path(folder) / "results.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        return min(max(int(float(row["epoch"])) for row in rows), limit)
    except (OSError, KeyError, ValueError):
        return min(fallback, limit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/collectors")
    parser.add_argument(
        "--classes",
        choices=["all", "collectors"],
        default="all",
        help="Classes to export from local reviewed frames",
    )
    parser.add_argument(
        "--dataset-yaml",
        help="Use an already prepared YOLO dataset, including a collector-only import",
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--run-name", default="train")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Use validation only while iterating; leave the test split untouched",
    )
    parser.add_argument(
        "--device", default="cpu", help="cpu, or 0 for a CUDA-enabled PyTorch installation"
    )
    parser.add_argument(
        "--learning-rate", type=float, help="Use AdamW with this explicit fine-tuning rate"
    )
    parser.add_argument(
        "--freeze",
        type=int,
        default=0,
        help="Number of initial layers to freeze during fine-tuning",
    )
    args = parser.parse_args()
    if args.learning_rate is not None and not 0 < args.learning_rate < 1:
        parser.error("Learning rate must be between 0 and 1")
    if args.freeze < 0:
        parser.error("Freeze must be nonnegative")
    if args.dataset_yaml:
        specification = Path(args.dataset_yaml).resolve()
        if not specification.is_file():
            raise ValueError("Dataset configuration does not exist: " + str(specification))
        report_path = specification.parent / "import-report.json"
        report = (
            json.loads(report_path.read_text())
            if report_path.exists()
            else {"dataset": str(specification)}
        )
    else:
        specification, report = prepare_dataset(
            args.data, classes=COLLECTORS if args.classes == "collectors" else KINDS
        )
    print(json.dumps(report, indent=2))
    print("Dataset:", specification)
    if args.prepare_only:
        return
    from ultralytics import YOLO

    model = YOLO(args.model)
    started = time.time()

    def record_progress(trainer):
        folder = Path(trainer.save_dir)
        progress = {
            "status": "training",
            "epoch": completed_epochs(folder, trainer.epoch + 1, trainer.epochs),
            "epochs": trainer.epochs,
            "elapsed_seconds": round(time.time() - started),
            "metrics": trainer.metrics,
            "run_directory": str(folder.resolve()),
        }
        temporary = folder / "progress.tmp.json"
        temporary.write_text(json.dumps(progress, indent=2, default=float))
        temporary.replace(folder / "progress.json")

    model.add_callback("on_fit_epoch_end", record_progress)
    tune = {}
    if args.learning_rate is not None:
        tune.update(
            optimizer="AdamW",
            lr0=args.learning_rate,
            warmup_bias_lr=args.learning_rate,
            warmup_epochs=1,
        )
    if args.freeze:
        tune["freeze"] = args.freeze
    model.train(
        data=str(specification),
        epochs=args.epochs,
        imgsz=args.image_size,
        batch=args.batch,
        device=args.device,
        workers=0,
        project=str(Path("runs/collectors").resolve()),
        name=args.run_name,
        seed=42,
        patience=args.patience,
        **tune,
    )
    checkpoint = Path(model.trainer.best)
    validation = YOLO(str(checkpoint)).val(
        data=str(specification), imgsz=args.image_size, device=args.device, workers=0
    )
    report_path = checkpoint.parent / "validation.json"
    report_path.write_text(json.dumps(validation.results_dict, indent=2))
    dataset_info = (
        json.loads(specification.read_text())
        if specification.read_text().lstrip().startswith("{")
        else {}
    )
    if dataset_info.get("test") and not args.skip_test:
        test = YOLO(str(checkpoint)).val(
            data=str(specification),
            split="test",
            imgsz=args.image_size,
            device=args.device,
            workers=0,
            project=str(Path("runs/collectors").resolve()),
            name=args.run_name + "-test",
        )
        (checkpoint.parent / "test.json").write_text(json.dumps(test.results_dict, indent=2))
    progress_path = checkpoint.parent.parent / "progress.json"
    progress = json.loads(progress_path.read_text())
    progress.update(status="complete", checkpoint=str(checkpoint.resolve()))
    progress_path.write_text(json.dumps(progress, indent=2))
    print("Checkpoint:", checkpoint.resolve())
    print("Validation:", report_path.resolve())
    print(
        "Inspect errors on unseen bases before using this model for attacks. No config was changed."
    )


if __name__ == "__main__":
    try:
        main()
    except (ValueError, ImportError) as error:
        raise SystemExit(str(error))
