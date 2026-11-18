"""Select per-class thresholds using validation images only."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image

from evaluate_saved import score_detections
from vision import COLLECTORS, Detection


def select_thresholds(frames, precision_target=0.98):
    tables = defaultdict(list)
    for integer in range(10, 96, 5):
        threshold = integer / 100
        totals = {kind: {"tp": 0, "fp": 0, "fn": 0} for kind in COLLECTORS}
        for frame in frames:
            predictions = [
                Detection(d["kind"], tuple(d["box"]), d["score"])
                for d in frame["predictions"]
                if d["score"] >= threshold
            ]
            result = score_detections(predictions, frame["truth"])
            for kind, counts in result["by_class"].items():
                totals[kind]["tp"] += counts["true_positive"]
                totals[kind]["fp"] += counts["false_positive"]
                totals[kind]["fn"] += counts["false_negative"]
        for kind, counts in totals.items():
            tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
            precision = tp / (tp + fp) if tp + fp else 0
            recall = tp / (tp + fn) if tp + fn else 0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
            tables[kind].append(
                dict(threshold=threshold, precision=precision, recall=recall, f1=f1, **counts)
            )
    chosen = {}
    for kind, rows in tables.items():
        qualified = [row for row in rows if row["precision"] >= precision_target and row["tp"] > 0]
        chosen[kind] = max(
            qualified or rows, key=lambda row: (row["f1"], row["precision"], row["threshold"])
        )
        chosen[kind] = dict(chosen[kind], precision_target_met=bool(qualified))
    return chosen, dict(tables)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--dataset", default="data/training/collector-baseline/dataset.yaml")
    parser.add_argument("--output", default="evaluation/calibration-assets-02.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", type=float, default=0.98)
    args = parser.parse_args()
    from ultralytics import YOLO

    model = YOLO(args.checkpoint)
    spec = json.loads(Path(args.dataset).read_text())
    if set(spec["names"]) != set(COLLECTORS):
        raise ValueError("Calibration needs the three collector classes")
    root = Path(spec["path"])
    frames = []
    for path in sorted((root / spec["val"]).glob("*")):
        with Image.open(path) as image:
            image = image.convert("RGB")
            result = model.predict(
                image, imgsz=1280, conf=0.05, device=args.device, max_det=1000, verbose=False
            )[0]
            predictions = []
            for box, score, cid in zip(
                result.boxes.xyxy.cpu().tolist(),
                result.boxes.conf.cpu().tolist(),
                result.boxes.cls.cpu().tolist(),
            ):
                x1, y1, x2, y2 = box
                predictions.append(
                    {
                        "kind": result.names[int(cid)],
                        "score": float(score),
                        "box": [x1, y1, x2 - x1, y2 - y1],
                    }
                )
            truth = []
            for line in (root / "labels" / "val" / (path.stem + ".txt")).read_text().splitlines():
                if not line.strip():
                    continue
                cid, cx, cy, w, h = map(float, line.split())
                truth.append(
                    {
                        "kind": spec["names"][int(cid)],
                        "box": [
                            (cx - w / 2) * image.width,
                            (cy - h / 2) * image.height,
                            w * image.width,
                            h * image.height,
                        ],
                    }
                )
            frames.append({"image": path.name, "predictions": predictions, "truth": truth})
    chosen, tables = select_thresholds(frames, args.precision)
    report = {
        "checkpoint": args.checkpoint,
        "split": "validation",
        "images": len(frames),
        "image_size": 1280,
        "precision_target": args.precision,
        "chosen": chosen,
        "model_class_confidence": {kind: row["threshold"] for kind, row in chosen.items()},
        "sweep": tables,
        "note": "Threshold selection on validation; not an unbiased test estimate.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {key: report[key] for key in ("images", "chosen", "model_class_confidence")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
