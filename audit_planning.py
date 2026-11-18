"""Audit planning on reviewed validation screenshots, without game capture/input."""

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from dataset import Dataset
from deployment_map import visible_deployment
from vision import COLLECTORS, VisualPlanner, draw_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="models/collectors-scouting-23.pt")
    parser.add_argument("--source", default="data/scouting-reviewed-23")
    parser.add_argument("--output", default="evaluation/planning-scouting23")
    parser.add_argument(
        "--wall-references", help="Experimental wall template directory; saved-file audit only"
    )
    parser.add_argument(
        "--wall-network",
        action="store_true",
        help="Audit learned wall/building geometry in preview mode",
    )
    parser.add_argument(
        "--proposals",
        action="store_true",
        help="Evaluate unverified pure route proposals on saved images only",
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--heights", nargs="+", type=int, default=[810, 905])
    parser.add_argument(
        "--boundary-mode",
        choices=("legacy", "ridge", "directional", "learned", "ensemble"),
        default="legacy",
    )
    parser.add_argument("--boundary-threshold", type=float, default=0.8)
    args = parser.parse_args()
    settings = json.loads(Path("config.json").read_text())["vision"]
    settings.update(backend="hybrid", model_path=args.checkpoint, model_device="0")
    settings["boundary_edge_mode"] = args.boundary_mode
    if not 0 < args.boundary_threshold < 1:
        parser.error("Boundary threshold must be between 0 and 1")
    settings["boundary_threshold"] = args.boundary_threshold
    if args.wall_references:
        settings.update(
            structure_template_dir=args.wall_references,
            structure_confidence=0.80,
            planning_preview_only=True,
        )
    if args.wall_network:
        settings.update(
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
    if args.proposals:
        settings.update(geometry_validated=True)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    source = Dataset(args.source)
    records = [r for r in source.records() if r.get("reviewed") and r.get("split") == args.split]
    reports = []
    for height in args.heights:
        if not 1 <= height <= 1080:
            parser.error("Playfield heights must be between 1 and 1080")
        settings["playfield"] = [0, 0, 1920, height]
        planner = VisualPlanner(settings)
        for record in records:
            with Image.open(source.root / "images" / (record["id"] + ".png")) as image:
                image = image.convert("RGB")
            plan = planner.analyze(image)
            if args.proposals:
                # The detector remains preview-only. Only the pure planner is
                # exercised; no controller or game input is constructed.
                from vision import grass_terrain, make_plan

                region = planner.visible_region(image, plan.detections)
                plan = make_plan(
                    plan.detections,
                    plan.boundary,
                    dict(settings, planning_preview_only=False),
                    planner.detector.kinds,
                    region,
                    grass_terrain(image, settings["playfield"]),
                )
                plan.proposal_only = True
            report = {
                "base": record["group"],
                "playfield_height": height,
                "recognized_objects": dict(Counter(d.kind for d in plan.detections)),
                "boundary_visible": bool(plan.boundary),
                "unverified_proposals": args.proposals,
                "boundary_mode": args.boundary_mode,
                "boundary_threshold": args.boundary_threshold,
                **plan.summary(),
            }
            if args.wall_references:
                anchors = [d.center for d in plan.detections if d.kind in COLLECTORS]
                region = visible_deployment(
                    image,
                    settings["playfield"],
                    [
                        (0, 0, 300, 260),
                        (1560, 0, 360, 210),
                        (0, 810, 480, 270),
                        (1640, 750, 280, 330),
                        (860, 0, 200, 110),
                    ],
                    anchors,
                )
                report["experimental_deployment_problem"] = region.problem
            reports.append(report)
            draw_plan(image, plan).save(output / f"{record['group']}-{height}.jpg", quality=90)
            print(json.dumps({k: v for k, v in report.items() if k != "targets"}), flush=True)
    (output / "report.json").write_text(
        json.dumps(
            {
                "frames": reports,
                "note": "Offline validation audit, not attack success. Sparse wall matches cannot establish complete wall geometry. Gameplay settings unchanged.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
