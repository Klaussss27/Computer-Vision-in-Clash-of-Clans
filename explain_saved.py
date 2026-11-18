"""Explain an actual saved-base model analysis without game capture or input."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from attack_policy import order_targets
from vision import VisualPlanner, draw_plan


def emit(kind, **values):
    print(json.dumps({"event": kind, **values}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--stop-file", required=True, type=Path)
    args = parser.parse_args()

    def activity(stage, action, evidence, next_action, **extra):
        if args.stop_file.exists():
            raise RuntimeError("Stopped by user")
        emit(
            "activity",
            stage=stage,
            action=action,
            evidence=evidence,
            next_action=next_action,
            mode="Saved-image preview",
            **extra,
        )

    config = json.loads(Path("config.json").read_text())
    activity(
        "loading",
        "Open saved base",
        args.image.name,
        "Load the collector, wall, building, and boundary models",
    )
    image = Image.open(args.image).convert("RGB")
    if image.size != tuple(config["capture_size"]):
        raise ValueError("Use a full screenshot matching capture_size")
    settings = dict(
        config["vision"],
        planning_preview_only=True,
        wall_model_path="models/walls-sparse-02.pt",
        building_model_path="models/building-occupancy-01.pt",
        boundary_edge_mode="ensemble",
        boundary_threshold=0.2,
        boundary_model_path="models/boundary-pixels-01.pt",
    )
    activity(
        "analysis",
        "Detect collectors and estimate routes",
        "Saved pixels are analyzed by the configured models",
        "Exclude hidden, walled, or unresolved targets",
    )
    planner = VisualPlanner(settings)
    plan = planner.analyze_offline(image)
    activity(
        "plan",
        f"{len(plan.targets)} collectors; {sum(t.eligible for t in plan.targets)} route proposals",
        "; ".join(plan.problems) or "Boundary and route checks completed for this saved image",
        "Show selected targets and skipped targets",
    )
    ordered = order_targets(plan.targets, settings)[: settings["max_targets"]]
    overlay = draw_plan(image, plan)
    draw = ImageDraw.Draw(overlay)
    for index, target in enumerate(ordered, 1):
        x, y = target.spawn
        draw.ellipse((x - 14, y - 14, x + 14, y + 14), fill="black", outline="cyan", width=3)
        draw.text((x - 4, y - 6), str(index), fill="white")
        activity(
            "proposal",
            f"#{index}: {target.collector.kind.replace('_', ' ')}",
            f"{target.reason}; detector confidence {target.collector.score:.0%}",
            "Proposed Goblin deployment",
            spawn=list(target.spawn),
            target=list(target.collector.center),
        )
    for target in plan.targets:
        if not target.eligible:
            activity(
                "skip",
                "Skip " + target.collector.kind.replace("_", " "),
                target.reason,
                "Continue checking other collectors",
                target=list(target.collector.center),
            )
    if args.stop_file.exists():
        raise RuntimeError("Stopped by user")
    overlay.thumbnail((960, 540))
    overlay.save("dashboard-frame.tmp.png")
    Path("dashboard-frame.tmp.png").replace("dashboard-frame.png")
    emit("frame", frames=1)
    emit("status", state="saved-image preview", loot=None, plan=plan.summary())
    Path("evaluation").mkdir(exist_ok=True)
    Path("evaluation/explained-plan.json").write_text(json.dumps(plan.summary(), indent=2))
    activity(
        "complete",
        "Saved-base explanation complete",
        f"{len(ordered)} numbered deployment proposals; all game controls remained inactive",
        "Inspect numbered markers and the Activity tab",
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        emit("error", message=str(error))
        raise SystemExit(1)
