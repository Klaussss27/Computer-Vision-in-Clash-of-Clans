"""Explicit supervised zoom-only calibration; never deploys or clicks buttons."""

import argparse
import json
import uuid
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="store_true",
        help="Start only after the user is ready for capture and wheel input",
    )
    args = parser.parse_args()
    if not args.run:
        print(
            "Wait for explicit readiness, then use --run. This probe only captures and zooms; it does not attack."
        )
        return
    from farmer import Farmer
    from vision import VisualPlanner
    from window_capture import GameWindow, enable_dpi_awareness

    config = json.loads(Path("config.json").read_text())
    settings = dict(
        config["vision"],
        backend="hybrid",
        model_path="models/collectors-scouting-23.pt",
        model_device="0",
    )
    planner = VisualPlanner(settings)
    farmer = Farmer(config, live=True)
    enable_dpi_awareness()
    window = GameWindow(config["window_title"], config["capture_size"])
    farmer.window = window
    output = Path("data/calibration/zoom") / uuid.uuid4().hex[:8]
    records = []

    class Probe:
        def view_ready(self, image):
            output.mkdir(parents=True, exist_ok=True)
            name = f"step-{len(records)}.png"
            image.save(output / name)
            region = planner.visible_region(image)
            record = {"image": name, "region_ready": not region.problem, "problem": region.problem}
            records.append(record)
            print(json.dumps(record), flush=True)
            return not region.problem

    try:
        screen = farmer.wait_for_screen("scouting", 20)
        farmer.normalize_scouting_view(screen, Probe())
    finally:
        window.close()
        if records:
            (output / "report.json").write_text(
                json.dumps(
                    {
                        "purpose": "Supervised zoom-only calibration; no button/deployment input",
                        "max_ticks": settings.get("max_zoom_out_ticks", 6),
                        "frames": records,
                    },
                    indent=2,
                )
            )
            print(output.resolve())


if __name__ == "__main__":
    main()
