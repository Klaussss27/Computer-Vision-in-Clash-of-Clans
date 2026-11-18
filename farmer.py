"""Window-based farming controller. Positions are relative to the captured window."""

import argparse
import json
import logging
import threading
import time
from pathlib import Path

from attack_policy import army_profile, jumps_enabled, order_targets
from attacks import run_attack, validate_attack


class Farmer:
    def __init__(
        self, config, live=False, stop_event=None, on_frame=None, on_result=None, on_activity=None
    ):
        import cv2
        import numpy as np
        import pyautogui
        import pytesseract

        self.cv2, self.np, self.gui, self.ocr = cv2, np, pyautogui, pytesseract
        self.config = config
        self.live = live
        self.stop_event = stop_event or threading.Event()
        self.on_frame = on_frame
        self.on_result = on_result
        self.on_activity = on_activity
        self.current_plan = None
        self.gui.FAILSAFE = True
        self.gui.PAUSE = config["click_delay_seconds"]
        self.ocr.pytesseract.tesseract_cmd = config["tesseract_cmd"]
        self.army_reader = None
        self.confirmation_reader = None
        if config.get("army_detection"):
            from army import ArmyReader

            self.army_reader = ArmyReader(
                army_profile(config["army_detection"], config.get("vision", {})), self.ocr
            )
        if config.get("army_confirmation_detection"):
            from army import ArmyReader

            self.confirmation_reader = ArmyReader(
                army_profile(config["army_confirmation_detection"], config.get("vision", {})),
                self.ocr,
            )
        from state_recognition import load_state_templates

        self.templates = load_state_templates(config)

    def activity(self, stage, action, evidence, next_action, **details):
        callback = getattr(self, "on_activity", None)
        if callback:
            callback(
                {
                    "stage": stage,
                    "action": action,
                    "evidence": evidence,
                    "next_action": next_action,
                    **details,
                }
            )

    def screenshot(self):
        self.check_stopped()
        screen = self.window.screenshot()
        if self.on_frame:
            if self.current_plan:
                from vision import draw_plan

                self.on_frame(draw_plan(screen, self.current_plan))
            else:
                self.on_frame(screen)
        return screen

    def check_stopped(self):
        if self.stop_event.is_set():
            raise RuntimeError("Stopped by user")

    def sleep(self, seconds):
        if self.stop_event.wait(seconds):
            self.check_stopped()

    def matches_screen(self, screen, name):
        from state_recognition import marker_match

        return marker_match(
            screen,
            self.templates[name],
            self.config["template_confidence"],
            self.config.get("state_max_mean_error", 0.12),
        )

    def click(self, point):
        self.check_stopped()
        logging.info("%s click at %s", "LIVE" if self.live else "Preview", point)
        if self.live:
            self.gui.click(*self.window.desktop_point(point))

    def click_button(self, name):
        self.click(self.config["buttons"][name])

    def click_button_on_state(self, name, *states):
        current = self.screenshot()
        if not any(self.matches_screen(current, state) for state in states):
            raise RuntimeError(f"Screen changed before {name}; no click sent")
        self.activity(
            "navigation",
            name.replace("_", " ").capitalize(),
            "Fresh screen matches " + ", ".join(states),
            "Wait for the next screen",
        )
        self.click_button(name)

    def require_attack_screen(self, screen):
        if not any(self.matches_screen(screen, state) for state in ("scouting", "battle")):
            raise RuntimeError("Expected scouting or battle screen; no deployment sent")

    def wait_for_screen(self, name, seconds=30):
        self.activity(
            "waiting",
            "Waiting for " + name.replace("_", " "),
            f"Timeout: {seconds} seconds",
            "Check the next captured frame",
        )
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            screen = self.screenshot()
            if self.matches_screen(screen, name):
                self.activity(
                    "screen",
                    "Recognized " + name.replace("_", " "),
                    "Screen marker matched",
                    "Continue the configured flow",
                )
                return screen
            self.sleep(1)
        raise RuntimeError(f"Timed out waiting for {name}; stopped without further clicks")

    def read_loot(self, screen):
        from ui_numbers import read_number

        return {
            resource: read_number(
                screen,
                box,
                self.ocr,
                self.config.get("loot_ocr_modes", {}).get(resource, ("gray", 175)),
            )
            for resource, box in self.config["loot_regions"].items()
        }

    def read_results(self, screen):
        from ui_numbers import read_number

        return {
            resource: read_number(
                screen,
                box,
                self.ocr,
                self.config.get("result_ocr_modes", {}).get(resource, ("gray", 175)),
            )
            for resource, box in self.config.get("result_regions", {}).items()
        }

    def confirm_goblin_deployment(self, expected_count, seconds=3):
        """A click on forbidden terrain does not consume a troop."""
        deadline = time.monotonic() + seconds
        last_count = None
        while time.monotonic() < deadline:
            self.sleep(0.25)
            screen = self.screenshot()
            if self.matches_screen(screen, "results"):
                return False
            self.require_attack_screen(screen)
            try:
                remaining = self.army_card(screen, "sneaky_goblins").count
            except RuntimeError:
                continue
            if remaining == expected_count:
                self.activity(
                    "verification",
                    "Troop deployment confirmed",
                    f"{remaining} Goblins remain, matching the expected count",
                    "Watch the target for rubble",
                )
                return True
            if remaining < expected_count:
                raise RuntimeError("Unexpected Goblin count decrease; stopping deployment")
            last_count = remaining
        raise RuntimeError(
            f"Goblin deployment not confirmed: expected {expected_count}, observed {last_count}; stopped"
        )

    def visual_attack(self, plan, planner):
        from vision import overlap

        if plan.proposal_only:
            raise RuntimeError("Offline proposal cannot execute an attack; no input sent")
        settings = self.config["vision"]
        use_jumps = jumps_enabled(settings)
        goblins, jumps = (
            settings["goblin_budget"],
            settings.get("jump_budget", 0) if use_jumps else 0,
        )
        targets = order_targets(plan.targets, settings)
        if not targets:
            raise RuntimeError(
                "No targets allowed by the current attack policy; no deployment sent"
            )
        for target in targets[: settings["max_targets"]]:
            self.activity(
                "target",
                "Check " + target.collector.kind.replace("_", " "),
                target.reason,
                "Verify the current target and army count",
                target=list(target.collector.center),
                spawn=list(target.spawn),
                confidence=round(target.collector.score, 3),
            )
            screen = self.screenshot()
            if self.matches_screen(screen, "results"):
                return
            self.require_attack_screen(screen)
            if not planner.aligned(screen):
                raise RuntimeError("Camera moved since planning; no deployment sent")
            # Re-detect before acting: the old scouting frame is not sufficient.
            detect = getattr(planner.detector, "detect_collectors", planner.detector.detect)
            present = any(
                d.kind == target.collector.kind and overlap(d.box, target.collector.box) > 0.5
                for d in detect(screen)
            )
            if not present:
                logging.info(
                    "Target no longer confidently visible; skipping %s", target.collector.center
                )
                continue
            if self.army_card(screen, "sneaky_goblins").count <= 0:
                logging.info("No Goblins remain; ending attack without spending another spell")
                break
            if target.layers == 1:
                if jumps <= 0:
                    continue
                card = self.army_card(screen, "jump_spell")
                if card.count == 0:
                    continue
                self.click(card.point)
                self.click(target.jump)
                jumps -= 1
            for _ in range(settings["max_batches_per_target"]):
                current = self.screenshot()
                if self.matches_screen(current, "results"):
                    return
                self.require_attack_screen(current)
                if not planner.aligned(current):
                    raise RuntimeError("Camera moved before deployment; no deployment sent")
                card = self.army_card(current, "sneaky_goblins")
                count = min(settings["goblins_per_batch"], goblins, card.count)
                if count <= 0:
                    break
                self.activity(
                    "deployment",
                    f"Deploy {count} Sneaky Goblins",
                    f"{card.count} visible; {goblins} left in the attack budget",
                    "Confirm the troop count decreases",
                    spawn=list(target.spawn),
                )
                self.click(card.point)
                for _ in range(count):
                    self.click(target.spawn)
                if not self.confirm_goblin_deployment(card.count - count):
                    return
                goblins -= count
                logging.info(
                    "Deployed %s goblins toward %s at %s (%s estimated wall layers)",
                    count,
                    target.collector.kind,
                    target.collector.center,
                    target.layers,
                )
                confirmations = 0
                self.activity(
                    "observation",
                    "Watching the collector",
                    "Destruction requires disappearance plus two rubble confirmations",
                    "Stop waiting on confirmation or timeout",
                )
                deadline = time.monotonic() + settings["target_wait_seconds"]
                while time.monotonic() < deadline:
                    self.sleep(1)
                    current = self.screenshot()
                    if self.matches_screen(current, "results"):
                        return
                    self.require_attack_screen(current)
                    confirmations = (
                        confirmations + 1 if planner.destroyed(current, target.collector) else 0
                    )
                    if confirmations >= 2:
                        break
                if confirmations >= 2:
                    logging.info("Confirmed rubble at %s", target.collector.center)
                    break
            if goblins <= 0:
                break
        logging.info("Reachable target pass finished; ending attack")
        self.activity(
            "ending",
            "Reachable target pass finished",
            "No more targets in the current pass or Goblin budget exhausted",
            "Surrender and read results",
        )
        self.click_button_on_state("end_battle", "scouting", "battle")
        self.wait_for_screen("surrender", 10)
        self.click_button_on_state("confirm_surrender", "surrender")

    def army_card(self, screen, kind):
        self.check_stopped()
        if self.army_reader is None:
            raise RuntimeError("Army recognition is not calibrated; no selection made")
        return self.army_reader.read(screen, kind)

    def confirm_army(self, screen):
        self.check_stopped()
        if self.confirmation_reader is None:
            raise RuntimeError(
                "Army confirmation recognition is not calibrated; matchmaking not started"
            )
        goblins = self.confirmation_reader.read(screen, "sneaky_goblins").count
        jumps = (
            self.confirmation_reader.read(screen, "jump_spell").count
            if jumps_enabled(self.config.get("vision", {}))
            else None
        )
        minimum = self.config.get(
            "minimum_goblins", self.config.get("vision", {}).get("goblins_per_batch", 1)
        )
        if goblins < minimum:
            raise RuntimeError(
                f"Only {goblins} Goblins ready; need {minimum}. Matchmaking not started."
            )
        logging.info(
            "Confirmed army: %s Sneaky Goblins; %s",
            goblins,
            f"{jumps} Jump Spells" if jumps is not None else "spells disabled",
        )
        self.click_button_on_state("confirm_attack", "army_confirmation")

    def zoom_view_changed(self, before, after):
        from zoom_review import view_changed

        settings = self.config["vision"]
        return view_changed(before, after, settings["playfield"], settings.get("ui_exclusions", []))

    def normalize_scouting_view(self, screen, planner=None):
        from vision import deployment_boundary

        settings = self.config["vision"]
        if not self.live:
            return screen
        limit = settings.get("max_zoom_out_ticks", 6)
        unchanged = 0
        for tick in range(limit + 1):
            self.check_stopped()
            if not self.matches_screen(screen, "scouting"):
                raise RuntimeError("Scouting screen changed while adjusting zoom")
            ready = (
                planner.view_ready(screen)
                if planner
                else bool(deployment_boundary(screen, settings["playfield"]))
            )
            if ready:
                logging.info("Scouting viewport ready after %s zoom-out ticks", tick)
                return screen
            if unchanged >= 2:
                logging.info(
                    "Zoom no longer changes the viewport; unresolved targets will be skipped"
                )
                return screen
            if tick == limit:
                logging.info("Zoom limit reached; unresolved viewport will be skipped by planning")
                return screen
            x, y, w, h = settings["playfield"]
            point = self.window.desktop_point((x + w // 2, y + h // 2))
            self.check_stopped()
            self.gui.moveTo(*point)
            self.check_stopped()
            self.gui.scroll(-1)
            self.sleep(0.6)
            previous = screen
            screen = self.screenshot()
            unchanged = 0 if self.zoom_view_changed(previous, screen) else unchanged + 1

    def run(self):
        if self.live:
            validate_config(self.config)
        self.current_plan = None
        screen = self.screenshot()
        if not self.live:
            screen.save("preview.png")
            logging.info("Saved preview.png. No input sent.")
            if "scouting" in self.templates and self.matches_screen(screen, "scouting"):
                logging.info("Detected loot: %s", self.read_loot(screen))
            return
        planner = None
        plan = None
        if self.config.get("attack_mode") == "visual":
            from vision import VisualPlanner

            planner = VisualPlanner(self.config["vision"])
        self.wait_for_screen("home")
        self.click_button_on_state("attack", "home")
        self.wait_for_screen("attack_menu")
        self.click_button_on_state("find_match", "attack_menu")
        confirmation = self.wait_for_screen("army_confirmation")
        self.confirm_army(confirmation)
        for _ in range(self.config["max_searches"]):
            screen = self.wait_for_screen("scouting")
            if planner:
                screen = self.normalize_scouting_view(screen, planner)
            amounts = self.read_loot(screen)
            logging.info("Available loot: %s", amounts)
            enough_loot = all(
                amounts[resource] >= minimum
                for resource, minimum in self.config["minimum_loot"].items()
            )
            if enough_loot:
                if planner:
                    from vision import draw_plan

                    plan = planner.analyze(screen)
                    draw_plan(screen, plan).save("vision-overlay.png")
                    Path("vision-plan.json").write_text(json.dumps(plan.summary(), indent=2))
                    logging.info("Visual plan: %s", plan.summary())
                    if not plan.problems and any(
                        t.eligible and (t.layers == 0 or jumps_enabled(self.config["vision"]))
                        for t in plan.targets
                    ):
                        self.current_plan = plan
                        break
                else:
                    break
            self.click_button_on_state("next", "scouting")
            self.sleep(3)
        else:
            raise RuntimeError("Search limit reached; stopped on scouting screen")
        if planner:
            self.visual_attack(plan, planner)
        else:
            run_attack(self.config["attack_steps"], self.click, self.sleep)
        result_screen = self.wait_for_screen("results", self.config["battle_timeout_seconds"])
        if self.config.get("result_regions"):
            try:
                loot = self.read_results(result_screen)
                logging.info("Loot gained: %s", loot)
                if self.on_result:
                    self.on_result(loot)
            except RuntimeError as error:
                logging.warning("Results could not be read: %s", error)
        self.click_button_on_state("return_home", "results")
        self.wait_for_screen("home")
        logging.info("Completed one attack.")


def validate_config(config):
    for name in ("home", "attack_menu", "army_confirmation", "scouting", "battle", "results"):
        if name not in config["templates"]:
            raise ValueError(f"Missing template: {name}")
    for name in ("attack", "find_match", "confirm_attack", "next", "return_home"):
        if name not in config["buttons"]:
            raise ValueError(f"Missing button position: {name}")
    if not config.get("army_confirmation_detection"):
        raise ValueError("Calibrate the army confirmation screen before matchmaking.")
    minimum = config.get("minimum_goblins", config.get("vision", {}).get("goblins_per_batch", 1))
    if type(minimum) is not int or not 1 <= minimum <= 300:
        raise ValueError("minimum_goblins must be an integer from 1 to 300")
    if set(config["loot_regions"]) != {"gold", "elixir", "dark_elixir"}:
        raise ValueError("Configure all three loot_regions")
    width, height = config["capture_size"]
    if config.get("attack_mode") == "visual":
        from vision import validate_vision

        validate_vision(config)
    else:
        validate_attack(config["attack_steps"], (width, height))
    points = list(config["buttons"].values())
    if any(len(p) != 2 or not (0 <= p[0] < width and 0 <= p[1] < height) for p in points):
        raise ValueError("Click positions must be inside capture_size")
    for x, y, w, h in config["loot_regions"].values():
        if min(x, y) < 0 or min(w, h) <= 0 or x + w > width or y + h > height:
            raise ValueError("Loot regions must be inside capture_size")
    if not 1 <= config["max_searches"] <= 100:
        raise ValueError("max_searches must be between 1 and 100")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--live", action="store_true", help="Enable mouse input for one attack")
    parser.add_argument(
        "--session",
        action="store_true",
        help="Repeat attacks while the configured army-ready indicator is visible",
    )
    parser.add_argument("--list-windows", action="store_true", help="Show available window titles")
    parser.add_argument(
        "--preview-attack", action="store_true", help="Print attack steps without input"
    )
    args = parser.parse_args()
    if args.session and not args.live:
        parser.error("--session requires --live")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.list_windows:
        from window_capture import enable_dpi_awareness, list_windows

        enable_dpi_awareness()
        for _, title in list_windows():
            print(title)
        return
    config = json.loads(Path(args.config).read_text())
    if args.preview_attack:
        validate_attack(config["attack_steps"], config["capture_size"])
        for number, step in enumerate(config["attack_steps"], 1):
            print(number, step)
        return
    if args.live:
        validate_config(config)
        if args.session:
            from session_runner import validate_session

            validate_session(config)
    from window_capture import GameWindow, enable_dpi_awareness

    enable_dpi_awareness()
    farmer = Farmer(config, args.live)
    window = GameWindow(config["window_title"], config["capture_size"])
    farmer.window = window
    try:
        if args.session:
            from session_runner import SessionRunner

            SessionRunner(farmer).run()
        else:
            farmer.run()
    finally:
        window.close()


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError) as error:
        logging.error("%s", error)
        raise SystemExit(1)
