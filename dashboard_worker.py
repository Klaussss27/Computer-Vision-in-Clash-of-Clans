"""Capture and farming worker used by the desktop dashboard."""

import argparse
import json
import logging
import threading
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

from farmer import Farmer, validate_config
from window_capture import GameWindow, enable_dpi_awareness


def emit(kind, **values):
    print(json.dumps({"event": kind, **values}), flush=True)


class DashboardLog(logging.Handler):
    def emit(self, record):
        emit("log", message=self.format(record))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--session", action="store_true")
    parser.add_argument("--max-battles", type=int, default=0)
    parser.add_argument("--stop-file", required=True)
    args = parser.parse_args()
    if args.session and not args.live:
        parser.error("--session requires --live")
    Path("logs").mkdir(exist_ok=True)
    handlers = [
        DashboardLog(),
        RotatingFileHandler("logs/farmer.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8"),
    ]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=handlers)
    stop = threading.Event()

    def receive_stop():
        while not stop.wait(0.1):
            if Path(args.stop_file).exists():
                stop.set()

    threading.Thread(target=receive_stop, daemon=True).start()
    config = json.loads(Path("config.json").read_text())
    if args.live:
        validate_config(config)
        if args.session:
            from session_runner import validate_session

            config.setdefault("session", {})["max_battles"] = args.max_battles
            validate_session(config)
    enable_dpi_awareness()
    last_frame = 0
    frames = 0

    def publish_frame(image):
        nonlocal last_frame, frames
        if time.monotonic() - last_frame < 1:
            return
        last_frame = time.monotonic()
        frames += 1
        preview = image.copy()
        preview.thumbnail((960, 540))
        preview.save("dashboard-frame.tmp.png")
        Path("dashboard-frame.tmp.png").replace("dashboard-frame.png")
        emit("frame", frames=frames)

    farmer = Farmer(
        config,
        args.live,
        stop_event=stop,
        on_frame=publish_frame,
        on_result=lambda loot: emit("loot_gained", loot=loot),
        on_activity=lambda details: emit(
            "activity", mode="Live attack" if args.live else "Monitoring", **details
        ),
    )
    planner = None
    if config.get("attack_mode") == "visual":
        from vision import VisualPlanner, draw_plan

        planner = VisualPlanner(config["vision"])
    window = GameWindow(config["window_title"], config["capture_size"])
    farmer.window = window
    try:
        if args.live:
            logging.info("Configured attack mode starts in 8 seconds. Focus the game now.")
            farmer.sleep(8)
            if args.session:
                from session_runner import SessionRunner

                SessionRunner(farmer, on_battle=lambda count: emit("completed", battles=1)).run()
            else:
                farmer.run()
                emit("completed", battles=1)
        else:
            farmer.on_frame = None
            logging.info("Monitoring started. No mouse input will be sent.")
            while not stop.is_set():
                screen = farmer.screenshot()
                priority = (
                    "surrender",
                    "results",
                    "army_confirmation",
                    "attack_menu",
                    "scouting",
                    "battle",
                    "home",
                )
                state = next(
                    (
                        name
                        for name in priority
                        if name in farmer.templates and farmer.matches_screen(screen, name)
                    ),
                    "unknown",
                )
                emit(
                    "activity",
                    mode="Monitoring",
                    stage="screen",
                    action="Recognized " + state.replace("_", " "),
                    evidence="Matching game UI markers"
                    if state != "unknown"
                    else "No supported UI marker matched",
                    next_action="Analyze scouting view"
                    if state == "scouting"
                    else "Wait for a scouting screen",
                )
                loot = None
                screen.save("vision-frame.tmp.png")
                Path("vision-frame.tmp.png").replace("vision-frame.png")
                plan = None
                if state == "scouting":
                    try:
                        loot = farmer.read_loot(screen)
                    except RuntimeError as error:
                        logging.warning("%s", error)
                    if planner:
                        plan = planner.analyze(screen)
                        emit(
                            "activity",
                            mode="Monitoring",
                            stage="plan",
                            action=f"{len(plan.targets)} collectors detected",
                            evidence="; ".join(plan.problems) or "Planning checks completed",
                            next_action="Display the current plan",
                        )
                        screen = draw_plan(screen, plan)
                publish_frame(screen)
                emit("status", state=state, loot=loot, plan=plan.summary() if plan else None)
                stop.wait(2)
    finally:
        stop.set()
        window.close()
        logging.info("Worker stopped")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        logging.exception("Worker stopped: %s", error)
        emit("error", message=str(error))
        raise SystemExit(1)
