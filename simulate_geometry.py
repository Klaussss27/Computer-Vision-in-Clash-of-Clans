"""Generate route proposals from a saved image; contains no capture/input code."""

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from deployment_map import visible_deployment
from neural_detector import HybridDetector
from vision import COLLECTORS, collector_visibility_problem, draw_plan, grass_terrain, make_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--output", default="evaluation/geometry-proposal")
    args = parser.parse_args()
    settings = json.loads(Path("config.json").read_text())["vision"]
    settings.update(
        backend="hybrid",
        model_path="models/collectors-scouting-23.pt",
        model_device="0",
        model_confidence=0.5,
        terrain_guard=True,
        max_approach_pixels=240,
        approach_slack_pixels=32,
        wall_model_path="models/walls-sparse-02.pt",
        building_model_path="models/building-occupancy-01.pt",
        planning_preview_only=True,
        ui_exclusions=[
            (0, 0, 300, 260),
            (1560, 0, 360, 210),
            (0, 810, 480, 270),
            (1640, 750, 280, 330),
            (860, 0, 200, 110),
        ],
    )
    image = Image.open(args.image).convert("RGB")
    detector = HybridDetector(settings)
    detections = detector.detect_for_planning(image)
    region = visible_deployment(
        image,
        settings["playfield"],
        settings["ui_exclusions"],
        [
            d.ground_center
            for d in detections
            if d.kind in COLLECTORS and not collector_visibility_problem(d, settings)
        ],
        edge_mode=settings.get("boundary_edge_mode", "legacy"),
    )
    contours, _ = cv2.findContours(
        region.inside.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contour = max(contours, key=cv2.contourArea) if contours else None
    x, y, _, _ = settings["playfield"]
    boundary = (
        [(int(px) + x, int(py) + y) for px, py in cv2.approxPolyDP(contour, 3, True).reshape(-1, 2)]
        if contour is not None
        else []
    )
    # This calls only the pure planner. It does not change the active config or
    # live guards; proposals are explicitly unverified and cannot send input.
    proposal_settings = dict(settings, planning_preview_only=False, geometry_validated=True)
    plan = make_plan(
        detections,
        boundary,
        proposal_settings,
        detector.kinds,
        region,
        grass_terrain(image, settings["playfield"]),
    )
    plan.proposal_only = True
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    draw_plan(image, plan).save(output.with_suffix(".jpg"), quality=92)
    report = {
        "role": "unverified offline geometry proposals",
        "image": args.image,
        "objects": dict(Counter(d.kind for d in detections)),
        **plan.summary(),
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "targets"}, indent=2))
    for target in plan.targets:
        print(
            target.collector.kind,
            target.collector.box,
            "layers:",
            target.layers,
            "spawn:",
            target.spawn,
            target.reason,
        )


if __name__ == "__main__":
    main()
