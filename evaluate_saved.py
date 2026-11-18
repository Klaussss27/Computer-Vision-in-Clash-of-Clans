"""Score collector detections on explicitly labeled saved files; never captures the game."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from PIL import Image, ImageDraw

from neural_detector import NeuralDetector
from vision import COLLECTORS


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    intersection = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(
        0, min(ay + ah, by + bh) - max(ay, by)
    )
    return intersection / max(1, aw * ah + bw * bh - intersection)


def score_detections(predictions, truth, threshold=0.5):
    unused = set(range(len(truth)))
    matches, false_positives = [], []
    for prediction in sorted(predictions, key=lambda item: item.score, reverse=True):
        candidates = [
            (iou(prediction.box, truth[index]["box"]), index)
            for index in unused
            if truth[index]["kind"] == prediction.kind
        ]
        best = max(candidates, default=(0, None))
        if best[0] >= threshold:
            unused.remove(best[1])
            matches.append({"kind": prediction.kind, "truth_index": best[1], "iou": best[0]})
        else:
            false_positives.append(asdict(prediction))
    missed = [truth[index] for index in sorted(unused)]
    counts = {}
    for kind in COLLECTORS:
        tp = sum(match["kind"] == kind for match in matches)
        fp = sum(item["kind"] == kind for item in false_positives)
        fn = sum(item["kind"] == kind for item in missed)
        counts[kind] = {
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "precision": tp / (tp + fp) if tp + fp else 0,
            "recall": tp / (tp + fn) if tp + fn else 0,
        }
    return {
        "by_class": counts,
        "matches": matches,
        "false_positives": false_positives,
        "missed": missed,
        "true_positive": len(matches),
        "false_positive": len(false_positives),
        "false_negative": len(missed),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--labels", default="evaluation_labels/scouting-01.json")
    parser.add_argument("--output", default="evaluation/saved-score")
    parser.add_argument("--confidence", type=float, default=0.6)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--calibration", help="Load per-class thresholds selected on validation data"
    )
    args = parser.parse_args()
    record = json.loads(Path(args.labels).read_text())
    image_path = Path(record["image"])
    if hashlib.sha256(image_path.read_bytes()).hexdigest() != record["sha256"]:
        raise ValueError("Saved image no longer matches the labeled file")
    config = json.loads(Path("config.json").read_text())
    settings = dict(
        config["vision"],
        model_path=args.checkpoint,
        model_confidence=args.confidence,
        model_device=args.device,
    )
    if args.calibration:
        calibration = json.loads(Path(args.calibration).read_text())
        if Path(calibration["checkpoint"]).resolve() != Path(args.checkpoint).resolve():
            raise ValueError("Calibration belongs to a different checkpoint")
        settings["model_class_confidence"] = calibration["model_class_confidence"]
    detector = NeuralDetector(settings, required_classes=COLLECTORS)
    image = Image.open(image_path).convert("RGB")
    predictions = detector.detect(image)
    result = score_detections(predictions, record["objects"])
    result.update(
        checkpoint=args.checkpoint,
        labels=args.labels,
        confidence=args.confidence,
        role=record["role"],
        note=record["annotation_note"],
        class_confidence=settings.get("model_class_confidence"),
        edge_margin=settings.get("model_edge_margin", 0),
    )
    root = Path(args.output)
    root.parent.mkdir(parents=True, exist_ok=True)
    root.with_suffix(".json").write_text(json.dumps(result, indent=2))
    draw = ImageDraw.Draw(image)
    for item in record["objects"]:
        x, y, w, h = item["box"]
        draw.rectangle((x, y, x + w, y + h), outline="cyan", width=1)
    for item in predictions:
        x, y, w, h = item.box
        draw.rectangle((x, y, x + w, y + h), outline="lime", width=3)
        draw.text(
            (x, y),
            f"{item.kind} {item.score:.2f}",
            fill="lime",
            stroke_width=1,
            stroke_fill="black",
        )
    image.save(root.with_suffix(".png"))
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("matches", "false_positives", "missed", "note")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
