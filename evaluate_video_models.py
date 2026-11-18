"""Compare collector checkpoints on unchanged validation bases and video calibration."""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from neural_detector import NeuralDetector
from vision import overlap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--output", default="evaluation/video-model-comparison.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    config = json.loads(Path("config.json").read_text())
    labels = json.loads(Path("video_expectations.json").read_text())["collector_pairs"]["objects"]
    reports = []
    for checkpoint in args.checkpoints:
        settings = dict(
            config["vision"], model_path=checkpoint, model_device="cpu", model_confidence=0.5
        )
        detector = NeuralDetector(
            settings, required_classes=("gold_mine", "elixir_collector", "dark_drill")
        )
        checks = []
        for file, after in [
            ("frame-009.png", False),
            ("frame-012.png", False),
            ("frame-042.png", True),
            ("frame-045.png", True),
        ]:
            found = detector.detect(
                Image.open(Path("data/video-review/2026-11-14-004714") / file).convert("RGB")
            )
            for index, label in enumerate(labels):
                actual = any(
                    d.kind == label["kind"] and overlap(d.box, label["box"]) > 0.5 for d in found
                )
                expected = not (after and label["destroyed_after"])
                checks.append(
                    {
                        "frame": file,
                        "object": index,
                        "expected_present": expected,
                        "present": actual,
                        "passed": expected == actual,
                    }
                )
        validation = detector.model.val(
            data="data/training/collectors-video-01/dataset.yaml",
            split="val",
            imgsz=1280,
            batch=1,
            device="cpu",
            workers=0,
            plots=False,
            project="evaluation/video-validation",
            name=Path(checkpoint).parent.parent.name + "-" + Path(checkpoint).stem,
        )
        report = {
            "checkpoint": checkpoint,
            "validation": validation.results_dict,
            "video_calibration_passed": sum(c["passed"] for c in checks),
            "video_calibration_total": len(checks),
            "video_checks": checks,
            "note": "Video crops were used for training; video checks are not independent validation. Reserved validation bases are unchanged.",
        }
        reports.append(report)
        Path(args.output).write_text(json.dumps(reports, indent=2, default=float))
        print(
            json.dumps({k: v for k, v in report.items() if k != "video_checks"}, default=float),
            flush=True,
        )


if __name__ == "__main__":
    main()
