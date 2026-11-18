"""Evaluate sparse saved boundary references; never captures or controls a game."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from deployment_map import red_edge_mask, visible_deployment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, default=Path("boundary_annotations.json"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/boundary-reference-check"))
    args = parser.parse_args()
    import torch

    torch.set_num_threads(4)
    references = json.loads(args.references.read_text())
    config = json.loads(Path("config.json").read_text())["vision"]
    args.output.mkdir(parents=True, exist_ok=True)
    reports = []
    for record in references["frames"]:
        path = Path(record["image"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Reference source changed: {path}")
        image = Image.open(path).convert("RGB")
        ref = np.zeros((image.height, image.width), np.uint8)
        for line in record["lines"]:
            cv2.polylines(ref, [np.asarray(line, np.int32)], False, 1, 1)
        overlay = image.copy()
        draw = ImageDraw.Draw(overlay)
        for line in record["lines"]:
            draw.line([tuple(p) for p in line], fill="cyan", width=3)
        for name, color in (("inside", "red"), ("outside", "lime")):
            for x, y in record[name]:
                draw.ellipse((x - 8, y - 8, x + 8, y + 8), outline=color, width=3)
        overlay.save(args.output / f"{record['group']}-references.jpg")
        for mode in ("legacy", "ridge", "directional", "learned"):
            edge = red_edge_mask(np.asarray(image), mode)
            close = cv2.dilate(edge, np.ones((11, 11), np.uint8)) != 0
            line_hits = int(np.count_nonzero(close & (ref != 0)))
            region = visible_deployment(
                image,
                config["playfield"],
                config["ui_exclusions"],
                record["inside"],
                edge_mode=mode,
            )
            correct_inside = sum(bool(region.inside[y, x]) for x, y in record["inside"])
            correct_outside = sum(bool(region.outside[y, x]) for x, y in record["outside"])
            report = {
                "base": record["group"],
                "split": record["split"],
                "mode": mode,
                "line_hits": line_hits,
                "line_points": int(ref.sum()),
                "inside_correct": correct_inside,
                "inside_points": len(record["inside"]),
                "outside_correct": correct_outside,
                "outside_points": len(record["outside"]),
                "problem": region.problem,
            }
            reports.append(report)
            print(json.dumps(report), flush=True)
            Image.fromarray(edge).save(args.output / f"{record['group']}-{mode}-edges.png")
    (args.output / "report.json").write_text(
        json.dumps({"note": references["role"], "frames": reports}, indent=2)
    )


if __name__ == "__main__":
    main()
