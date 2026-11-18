"""Explicit manual-point calibration: select Goblins and deploy exactly one."""

import argparse
import json
import uuid
from pathlib import Path

from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="store_true",
        help="Only after explicit readiness for capture and one Goblin",
    )
    parser.add_argument(
        "--reference", type=Path, help="Current saved base image reviewed by the user"
    )
    parser.add_argument(
        "--spawn", nargs=2, type=int, help="Explicitly approved deployment point in window pixels"
    )
    args = parser.parse_args()
    if not args.run:
        print(
            "Inactive. Requires explicit user readiness, a reviewed reference, and an approved spawn point."
        )
        return
    if not args.reference or args.spawn is None:
        parser.error("--run requires --reference and --spawn X Y")
    config = json.loads(Path("config.json").read_text())
    reference = Image.open(args.reference).convert("RGB")
    if reference.size != tuple(config["capture_size"]):
        parser.error("Reference size must match capture_size")
    x, y = args.spawn
    fx, fy, fw, fh = config["vision"]["playfield"]
    if not (fx <= x < fx + fw and fy <= y < fy + fh):
        parser.error("Spawn must be inside the playfield")
    for bx, by, bw, bh in config["vision"].get("ui_exclusions", []):
        if bx <= x < bx + bw and by <= y < by + bh:
            parser.error("Spawn overlaps excluded interface")
    from farmer import Farmer
    from vision import VisualPlanner
    from window_capture import GameWindow, enable_dpi_awareness

    # This tool never invokes the automated planner/controller. The point must
    # be manually reviewed; global live-attack geometry guards stay unchanged.
    farmer = Farmer(config, live=True)
    alignment = VisualPlanner.__new__(VisualPlanner)
    alignment.settings = config["vision"]
    alignment.reference_gray = alignment.frame_gray(reference)
    enable_dpi_awareness()
    output = Path("data/calibration/deployment") / uuid.uuid4().hex[:8]
    window = GameWindow(config["window_title"], config["capture_size"])
    farmer.window = window
    try:
        current = farmer.screenshot()
        farmer.require_attack_screen(current)
        if not alignment.aligned(current):
            raise RuntimeError("Current view differs from the reviewed reference; no input sent")
        card = farmer.army_card(current, "sneaky_goblins")
        if card.count < 1:
            raise RuntimeError("No Goblins available; no input sent")
        output.mkdir(parents=True, exist_ok=True)
        current.save(output / "before.png")
        farmer.click(card.point)
        farmer.sleep(0.3)
        current = farmer.screenshot()
        farmer.require_attack_screen(current)
        if not alignment.aligned(current):
            raise RuntimeError("View changed after card selection; no troop deployed")
        farmer.click((x, y))
        farmer.sleep(2)
        after = farmer.screenshot()
        after.save(output / "after.png")
        report = {
            "purpose": "Explicit single-Goblin manual-point calibration",
            "reference": str(args.reference.resolve()),
            "spawn": [x, y],
            "before_count": card.count,
            "automated_attack": False,
        }
        try:
            remaining = farmer.army_card(after, "sneaky_goblins").count
            report.update(after_count=remaining, count_decreased_by_one=remaining == card.count - 1)
        except RuntimeError as error:
            report["read_error"] = str(error)
        (output / "report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
        print(output.resolve())
    finally:
        window.close()


if __name__ == "__main__":
    main()
