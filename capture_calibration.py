"""Capture one deliberately requested game frame, then immediately stop capture."""

import argparse
import hashlib
import json
import time
import uuid
from pathlib import Path


def wait_for_marker(window, template, timeout=20, clock=time.monotonic, sleep=time.sleep):
    from state_recognition import marker_match

    deadline = clock() + timeout
    while clock() < deadline:
        image = window.screenshot()
        if marker_match(image, template):
            return image
        sleep(0.2)
    raise RuntimeError("Requested screen did not appear; no calibration image saved")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "label",
        choices=[
            "home",
            "attack_menu",
            "army_confirmation",
            "scouting",
            "battle",
            "surrender",
            "results",
            "army_ready",
        ],
    )
    parser.add_argument(
        "--wait-for-state",
        help="Explicitly watch for this calibrated state and save only the matching frame",
    )
    parser.add_argument("--timeout", type=float, default=20)
    args = parser.parse_args()
    config = json.loads(Path("config.json").read_text())
    template = None
    if args.wait_for_state:
        if args.wait_for_state != args.label or not 0.1 <= args.timeout <= 60:
            parser.error(
                "Wait state must match the requested label; timeout must be 0.1 to 60 seconds"
            )
        import cv2

        path = config.get("templates", {}).get(args.wait_for_state)
        template = cv2.imread(path, 0) if path else None
        if template is None:
            parser.error("Requested state has no calibrated marker")
    from window_capture import GameWindow, enable_dpi_awareness

    enable_dpi_awareness()
    window = GameWindow(config["window_title"], config["capture_size"])
    try:
        image = (
            wait_for_marker(window, template, args.timeout)
            if template is not None
            else window.screenshot()
        )
    finally:
        window.close()
    output = Path("data/calibration/requested")
    output.mkdir(parents=True, exist_ok=True)
    path = output / (args.label + "-" + uuid.uuid4().hex[:8] + ".png")
    image.save(path)
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "label": args.label,
                "size": image.size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "waited_for_state": args.wait_for_state,
                "purpose": "Deliberate calibration frame, not automatically added to training.",
            },
            indent=2,
        )
    )
    print(path.resolve())


if __name__ == "__main__":
    main()
