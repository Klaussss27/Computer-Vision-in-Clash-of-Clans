"""Compare wall segmentation and template matching at sparse reviewed points."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from dataset import Dataset
from vision import Detector
from wall_model import WallSegmenter


def metrics(rows, key):
    tp = sum(r["label"] == 1 and r[key] for r in rows)
    fp = sum(r["label"] == 0 and r[key] for r in rows)
    fn = sum(r["label"] == 1 and not r[key] for r in rows)
    tn = sum(r["label"] == 0 and not r[key] for r in rows)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/walls/sparse-01/last.pt")
    parser.add_argument("--output", default="evaluation/walls-sparse01")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    source = Dataset("data/scouting-bases")
    records = {r["group"]: r for r in source.records()}
    annotations = json.loads(Path("wall_annotations.json").read_text())
    segmenter = WallSegmenter(args.checkpoint, "0")
    config = json.loads(Path("config.json").read_text())["vision"]
    settings = dict(
        config, template_dir="templates/structures-v2", confidence=0.8, playfield=[0, 0, 1920, 905]
    )
    detector = Detector(settings)
    rows = []
    for group, positive in annotations["validation_points"].items():
        record = records[group]
        if record["split"] != "val":
            raise ValueError("Sparse evaluation must use validation bases only")
        image = Image.open(source.root / "images" / (record["id"] + ".png")).convert("RGB")
        field = image.crop((0, 0, 1920, 905))
        probability = segmenter.probabilities(field)
        wall_mask = probability >= 0.5
        # Save raw mask as well as overlay, so dense false detections are visible.
        Image.fromarray((probability * 255).astype(np.uint8)).save(
            output / (group + "-probability.png")
        )
        overlay = np.asarray(field).copy()
        overlay[wall_mask] = (overlay[wall_mask] * 0.4 + np.array([255, 0, 255]) * 0.6).astype(
            "uint8"
        )
        overlay = Image.fromarray(overlay)
        detected = detector.detect(image)
        negatives = [
            ((o["box"][0] + o["box"][2]) // 2, (o["box"][1] + o["box"][3]) // 2)
            for o in record["objects"]
            if o["kind"] in ("gold_mine", "elixir_collector", "dark_drill")
        ]
        negatives += [(50, 400), (1850, 400), (50, 650), (1850, 650)]
        for label, points in ((1, positive), (0, negatives)):
            for x, y in points:
                if not 0 <= y < 905:
                    continue
                score = float(probability[y, x])
                match = any(
                    bx <= x < bx + bw and by <= y < by + bh
                    for d in detected
                    for bx, by, bw, bh in [d.box]
                )
                rows.append(
                    {
                        "base": group,
                        "point": [x, y],
                        "label": label,
                        "score": score,
                        "neural": score >= 0.5,
                        "templates": match,
                    }
                )
                ImageDraw.Draw(overlay).ellipse(
                    (x - 5, y - 5, x + 5, y + 5), outline="lime" if label else "cyan", width=2
                )
        overlay.save(output / (group + "-overlay.jpg"), quality=92)
    report = {
        "checkpoint": args.checkpoint,
        "threshold": 0.5,
        "positive_points": sum(r["label"] for r in rows),
        "negative_points": sum(not r["label"] for r in rows),
        "neural": metrics(rows, "neural"),
        "templates": metrics(rows, "templates"),
        "points": rows,
        "note": "Sparse development point test, not mask IoU, wall recall across the full image, or route correctness. Test split untouched.",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "points"}, indent=2))


if __name__ == "__main__":
    main()
