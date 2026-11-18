"""Cancellable repeated attacks with explicit home and army readiness checks."""

import logging
import time


def validate_session(config):
    settings = config.get("session", {})
    mode = settings.get("readiness", "home_indicator")
    if mode == "confirmation":
        if not config.get("templates", {}).get("army_confirmation") or not config.get(
            "army_confirmation_detection"
        ):
            raise ValueError(
                "Sessions need calibrated army confirmation recognition before matchmaking."
            )
    elif mode == "home_indicator":
        if not config.get("templates", {}).get("army_ready"):
            raise ValueError("Continuous sessions need an army_ready template on the home screen.")
    else:
        raise ValueError("Unsupported session readiness mode")
    for name, default, low, high in [
        ("max_battles", 0, 0, 100000),
        ("ready_timeout_seconds", 900, 1, 86400),
        ("poll_seconds", 2, 1, 60),
        ("between_battles_seconds", 5, 0, 3600),
    ]:
        value = settings.get(name, default)
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"Invalid session setting: {name}")


class SessionRunner:
    def __init__(self, farmer, on_battle=None, clock=time.monotonic):
        if not farmer.live:
            raise ValueError(
                "Session runner requires live mode; use saved-file tests for simulation."
            )
        validate_session(farmer.config)
        self.farmer = farmer
        self.settings = farmer.config.get("session", {})
        self.on_battle = on_battle or (lambda count: None)
        self.clock = clock

    def wait_ready(self):
        farmer = self.farmer
        deadline = self.clock() + self.settings.get("ready_timeout_seconds", 900)
        announced = False
        while self.clock() < deadline:
            farmer.check_stopped()
            image = farmer.screenshot()
            if not farmer.matches_screen(image, "home"):
                raise RuntimeError(
                    "Home screen is not recognized; session stopped without recovery clicks."
                )
            if self.settings.get("readiness") == "confirmation":
                return  # Farmer checks unit counts in the confirmation dialog before matchmaking.
            if farmer.matches_screen(image, "army_ready"):
                return
            if not announced:
                logging.info("Waiting on home screen for the configured army-ready indicator")
                announced = True
            farmer.sleep(self.settings.get("poll_seconds", 2))
        raise RuntimeError("Army readiness timed out; session stopped.")

    def run(self):
        completed = 0
        maximum = self.settings.get("max_battles", 0)
        while not maximum or completed < maximum:
            self.farmer.check_stopped()
            self.wait_ready()
            self.farmer.run()
            completed += 1
            self.on_battle(completed)
            logging.info("Session completed %s attack(s)", completed)
            if maximum and completed >= maximum:
                break
            self.farmer.sleep(self.settings.get("between_battles_seconds", 5))
        return completed
