"""Check the calibrated recognizers against labeled frames from a saved recording."""

import argparse
import json
from pathlib import Path

import pytesseract
from PIL import Image

from army import ArmyReader
from state_recognition import load_state_templates, marker_match
from ui_numbers import read_number
from vision import Detection, VisualPlanner


def replay(root, labels, config, device="cpu"):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["sha256"] != labels["source_sha256"]:
        raise ValueError("Video labels do not match the source recording")
    pytesseract.pytesseract.tesseract_cmd = config["tesseract_cmd"]
    checks = []

    def check(category, source, expected, actual):
        checks.append(
            {
                "category": category,
                "source": source,
                "expected": expected,
                "actual": actual,
                "passed": expected == actual,
            }
        )

    states = (
        "home",
        "attack_menu",
        "army_confirmation",
        "scouting",
        "battle",
        "surrender",
        "results",
    )
    templates = load_state_templates(config)
    for file, expected in labels["states"].items():
        image = Image.open(root / file).convert("RGB")
        found = [
            s
            for s in states
            if marker_match(
                image,
                templates[s],
                config["template_confidence"],
                config.get("state_max_mean_error", 0.12),
            )
        ]
        check("state", file, [] if expected == "unknown" else [expected], found)
    reader = ArmyReader(config["army_detection"])
    for file, counts in labels["army"].items():
        image = Image.open(root / file).convert("RGB")
        for kind, expected in counts.items():
            try:
                actual = reader.read(image, kind).count
            except RuntimeError as error:
                actual = {"error": str(error)}
            check("army", file + ":" + kind, expected, actual)
    observed = {}
    for category in ("loot", "results"):
        regions = config["loot_regions" if category == "loot" else "result_regions"]
        modes = config.get("loot_ocr_modes" if category == "loot" else "result_ocr_modes", {})
        for file, values in labels[category].items():
            image = Image.open(root / file).convert("RGB")
            observed[file] = {}
            for kind, expected in values.items():
                try:
                    actual = read_number(
                        image, regions[kind], pytesseract, modes.get(kind, ("gray", 175))
                    )
                except RuntimeError as error:
                    actual = {"error": str(error)}
                observed[file][kind] = actual
                check(category, file + ":" + kind, expected, actual)
    pairs = labels["collector_pairs"]
    settings = dict(
        config["vision"],
        backend="hybrid",
        model_path="models/collectors-scouting-23.pt",
        model_device=device,
        model_confidence=0.5,
    )
    # Only collector presence, alignment, and local rubble matching are checked;
    # this replay does not authorize route planning or gameplay actions.
    planner = VisualPlanner(settings)
    before = Image.open(root / pairs["before"]).convert("RGB")
    after = Image.open(root / pairs["after"]).convert("RGB")
    confirmation = (
        Image.open(root / pairs["confirmation"]).convert("RGB")
        if pairs.get("confirmation")
        else None
    )
    planner.reference_gray = planner.frame_gray(before)
    for index, item in enumerate(pairs["objects"]):
        target = Detection(item["kind"], tuple(item["box"]), 1.0)
        check("rubble", f"before:{index}", False, planner.destroyed(before, target))
        check("rubble", f"after:{index}", item["destroyed_after"], planner.destroyed(after, target))
        if confirmation is not None:
            check(
                "rubble",
                f"confirmation:{index}",
                item["destroyed_after"],
                planner.destroyed(confirmation, target),
            )
    before_values = observed[pairs["before"]]
    after_values = observed[pairs["after"]]
    final = observed[next(iter(labels["results"]))]
    for kind in before_values:
        values = [before_values[kind], after_values[kind], final[kind]]
        actual = (
            values[0] - values[1] == values[2] if all(isinstance(v, int) for v in values) else False
        )
        check("reconciliation", kind, True, actual)
    return {
        "role": labels["role"],
        "source_sha256": manifest["sha256"],
        "checks": checks,
        "passed": sum(c["passed"] for c in checks),
        "total": len(checks),
        "note": "Calibrated recording replay, not a live attack or independent generalization test.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", default="data/video-review/2026-11-14-004714")
    parser.add_argument("--labels", default="video_expectations.json")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    report = replay(
        args.frames,
        json.loads(Path(args.labels).read_text()),
        json.loads(Path("config.json").read_text()),
        args.device,
    )
    output = Path(args.frames) / "replay-report.json"
    output.write_text(json.dumps(report, indent=2))
    print(f"{report['passed']}/{report['total']} recording checks passed.")
    for check in report["checks"]:
        if not check["passed"]:
            print(json.dumps(check))
    raise SystemExit(0 if report["passed"] == report["total"] else 1)
