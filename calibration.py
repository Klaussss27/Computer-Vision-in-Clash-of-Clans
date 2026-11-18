"""Inspect setup and save calibration from existing files without game capture."""

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image

from attack_policy import required_army

STATES = (
    "home",
    "attack_menu",
    "army_confirmation",
    "scouting",
    "battle",
    "results",
    "surrender",
    "army_ready",
)
BUTTONS = (
    "attack",
    "find_match",
    "confirm_attack",
    "next",
    "end_battle",
    "confirm_surrender",
    "return_home",
)


def readiness(config, root=Path("."), session=False):
    root = Path(root)
    missing = []
    for name in STATES if session else STATES[:-1]:
        if name == "army_ready" and config.get("session", {}).get("readiness") == "confirmation":
            continue
        path = config.get("templates", {}).get(name)
        if not path or not (root / path).is_file():
            missing.append("State marker: " + name)
    for name in BUTTONS:
        if name not in config.get("buttons", {}):
            missing.append("Button position: " + name)
    army = config.get("army_detection", {})
    for name in required_army(config.get("vision", {})):
        path = army.get("templates", {}).get(name)
        if not path or not (root / path).is_file():
            missing.append("Army icon: " + name)
        path = config.get("army_confirmation_detection", {}).get("templates", {}).get(name)
        if not path or not (root / path).is_file():
            missing.append("Confirmation army icon: " + name)
    vision = config.get("vision", {})
    if vision.get("geometry_validated") is not True:
        missing.append("Wall/deployment geometry validation")
    backend = vision.get("backend", "templates")
    kinds = (
        ("wall", "rubble")
        if backend == "hybrid"
        else ("gold_mine", "elixir_collector", "dark_drill", "wall", "rubble")
    )
    if backend != "yolo":
        for kind in kinds:
            if not list(
                (root / vision.get("template_dir", "templates/objects") / kind).glob("*.png")
            ):
                missing.append("Object examples: " + kind)
    if backend in ("hybrid", "yolo") and not (root / vision.get("model_path", "missing")).is_file():
        missing.append("Configured collector checkpoint")
    if vision.get("planning_preview_only"):
        missing.append("Experimental geometry cannot authorize live attacks")
    return {
        "configured": not missing,
        "missing": missing,
        "note": "File/configuration check only. Existing assets do not prove recognition, wall coverage, or gameplay reliability.",
    }


def save_marker(image_path, name, box, root=Path(".")):
    if name not in STATES:
        raise ValueError("Unsupported state marker")
    root = Path(root)
    config_path = root / "config.json"
    config = json.loads(config_path.read_text())
    with Image.open(image_path) as image:
        if list(image.size) != config["capture_size"]:
            raise ValueError("Calibration image must match capture_size")
        x, y, w, h = box
        if min(x, y) < 0 or min(w, h) < 10 or x + w > image.width or y + h > image.height:
            raise ValueError("Choose a marker inside the image, at least 10 x 10 pixels")
        crop = image.crop((x, y, x + w, y + h)).convert("RGB")
        import numpy as np

        if np.asarray(crop.convert("L")).std() < 3:
            raise ValueError("State marker needs visible detail")
    relative = Path("templates/interface") / (name + ".png")
    (root / relative).parent.mkdir(parents=True, exist_ok=True)
    crop.save(root / relative)
    config.setdefault("templates", {})[name] = relative.as_posix()
    temporary = config_path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(config, indent=2) + "\n")
    temporary.replace(config_path)
    provenance = root / "data/calibration/markers"
    provenance.mkdir(parents=True, exist_ok=True)
    (provenance / (name + ".json")).write_text(
        json.dumps(
            {
                "source": str(image_path),
                "sha256": hashlib.sha256(Path(image_path).read_bytes()).hexdigest(),
                "box": box,
            },
            indent=2,
        )
    )


def save_button(name, point, root=Path(".")):
    if name not in BUTTONS:
        raise ValueError("Unsupported button")
    path = Path(root) / "config.json"
    config = json.loads(path.read_text())
    w, h = config["capture_size"]
    if (
        len(point) != 2
        or any(type(v) is not int for v in point)
        or not (0 <= point[0] < w and 0 <= point[1] < h)
    ):
        raise ValueError("Button must be inside capture_size")
    config.setdefault("buttons", {})[name] = list(point)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(config, indent=2) + "\n")
    temporary.replace(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            readiness(json.loads(Path("config.json").read_text()), session=args.session), indent=2
        )
    )
