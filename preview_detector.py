"""Inspect collector model predictions on a saved game capture without sending input."""

import argparse
import json
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from PIL import Image, ImageDraw

from neural_detector import NeuralDetector
from vision import COLLECTORS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--image", default="preview.png")
    parser.add_argument("--output", default="evaluation/scouting")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--confidence", type=float, help="Override the confidence threshold for this preview only"
    )
    args = parser.parse_args()
    config = json.loads(Path("config.json").read_text())
    settings = dict(config["vision"], model_path=args.checkpoint, model_device=args.device)
    if args.confidence is not None:
        if not 0 < args.confidence <= 1:
            parser.error("--confidence must be between 0 and 1")
        settings["model_confidence"] = args.confidence
    detector = NeuralDetector(settings, required_classes=COLLECTORS)
    image = Image.open(args.image).convert("RGB")
    detector.detect(image)  # Warm up the inference engine before timing.
    started = time.perf_counter()
    detections = detector.detect(image)
    elapsed = (time.perf_counter() - started) * 1000
    draw = ImageDraw.Draw(image)
    colors = {"gold_mine": "yellow", "elixir_collector": "magenta", "dark_drill": "cyan"}
    for item in detections:
        x, y, w, h = item.box
        color = colors[item.kind]
        draw.rectangle((x, y, x + w, y + h), outline=color, width=3)
        draw.text(
            (x, max(0, y - 14)),
            f"{item.kind} {item.score:.2f}",
            fill=color,
            stroke_width=1,
            stroke_fill="black",
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output.with_suffix(".png"))
    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "source": args.image,
        "device": args.device,
        "confidence_threshold": settings.get("model_confidence", 0.6),
        "edge_margin": settings.get("model_edge_margin", 0),
        "warm_inference_ms": round(elapsed, 2),
        "counts": dict(Counter(d.kind for d in detections)),
        "detections": [asdict(item) for item in detections],
        "note": "Predictions on one screenshot; these counts are not an accuracy measurement.",
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "detections"}, indent=2))


if __name__ == "__main__":
    main()
