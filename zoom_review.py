"""Review saved zoom frames without opening or controlling a game window."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from state_recognition import load_state_templates, marker_match
from vision import VisualPlanner


def view_changed(before, after, field, exclusions=()):
    """Ignore small animations and UI changes when checking wheel response."""
    x, y, w, h = field
    first = np.asarray(before.convert("RGB"))[y : y + h, x : x + w]
    second = np.asarray(after.convert("RGB"))[y : y + h, x : x + w]
    if first.shape != second.shape:
        return True
    mask = np.ones(first.shape[:2], np.uint8)
    for bx, by, bw, bh in exclusions:
        left, top = max(0, bx - x), max(0, by - y)
        right, bottom = min(w, bx + bw - x), min(h, by + bh - y)
        if left < right and top < bottom:
            mask[top:bottom, left:right] = 0
    delta = cv2.absdiff(
        cv2.GaussianBlur(first, (7, 7), 0), cv2.GaussianBlur(second, (7, 7), 0)
    ).max(axis=2)
    return bool(
        np.count_nonzero((delta > 18) & (mask != 0)) > max(1, np.count_nonzero(mask)) * 0.04
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frames", nargs="+", type=Path, help="Saved images in zoom-step order")
    parser.add_argument("--output", type=Path, default=Path("evaluation/zoom-review"))
    args = parser.parse_args()
    config = json.loads(Path("config.json").read_text())
    settings = dict(
        config["vision"],
        backend="hybrid",
        model_path="models/collectors-scouting-23.pt",
        model_device="0",
    )
    planner = VisualPlanner(settings)
    markers = load_state_templates(config)
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    previous = None
    for index, path in enumerate(args.frames):
        image = Image.open(path).convert("RGB")
        if image.size != tuple(config["capture_size"]):
            raise ValueError(f"{path}: expected {config['capture_size']}, got {image.size}")
        region = planner.visible_region(image)
        states = [
            name
            for name, marker in markers.items()
            if marker is not None and marker_match(image, marker, config["template_confidence"])
        ]
        record = {
            "step": index,
            "source": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "states": states,
            "region_ready": not region.problem,
            "problem": region.problem,
            "scouting_zoom_allowed": "scouting" in states,
            "view_changed": None
            if previous is None
            else view_changed(
                previous, image, settings["playfield"], settings.get("ui_exclusions", [])
            ),
        }
        overlay = np.asarray(image).copy()
        x, y, w, h = settings["playfield"]
        area = overlay[y : y + h, x : x + w]
        for mask, color in ((region.inside, (220, 60, 60)), (region.outside, (60, 200, 90))):
            area[mask] = (area[mask] * 0.65 + np.array(color) * 0.35).astype(np.uint8)
        Image.fromarray(overlay).save(args.output / f"step-{index}.jpg")
        records.append(record)
        previous = image
        print(json.dumps(record), flush=True)
    (args.output / "report.json").write_text(
        json.dumps(
            {"role": "offline calibration; not live geometry approval", "frames": records}, indent=2
        )
    )


if __name__ == "__main__":
    main()
