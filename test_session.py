import threading
import unittest

from session_runner import SessionRunner, validate_session


class ReplayFarmer:
    def __init__(self, maximum=2, ready_after=1, home=True):
        self.live = True
        self.config = {
            "templates": {"army_ready": "saved-ready-marker.png"},
            "session": {
                "max_battles": maximum,
                "ready_timeout_seconds": 5,
                "poll_seconds": 1,
                "between_battles_seconds": 1,
            },
        }
        self.stop = threading.Event()
        self.ticks = 0
        self.frames = 0
        self.attacks = 0
        self.ready_after = ready_after
        self.home = home

    def check_stopped(self):
        if self.stop.is_set():
            raise RuntimeError("Stopped by user")

    def screenshot(self):
        self.frames += 1
        return self.frames

    def matches_screen(self, image, name):
        return self.home if name == "home" else self.frames >= self.ready_after

    def sleep(self, seconds):
        self.check_stopped()
        self.ticks += seconds

    def run(self):
        self.attacks += 1


class SessionTests(unittest.TestCase):
    def test_confirmation_mode_defers_unit_check_to_pre_match_dialog(self):
        farmer = ReplayFarmer(1)
        farmer.config["session"]["readiness"] = "confirmation"
        farmer.config["templates"] = {"army_confirmation": "marker.png"}
        farmer.config["army_confirmation_detection"] = {"region": [0, 0, 100, 100]}

        def matches(image, name):
            self.assertEqual(name, "home")
            return True

        farmer.matches_screen = matches
        self.assertEqual(SessionRunner(farmer, clock=lambda: farmer.ticks).run(), 1)

    def test_waits_for_ready_and_obeys_attack_limit(self):
        farmer = ReplayFarmer(2, ready_after=3)
        counts = []
        result = SessionRunner(farmer, counts.append, lambda: farmer.ticks).run()
        self.assertEqual(result, 2)
        self.assertEqual(farmer.attacks, 2)
        self.assertEqual(counts, [1, 2])
        self.assertGreaterEqual(farmer.frames, 4)

    def test_unknown_screen_and_ready_timeout_do_not_start_attacks(self):
        for farmer, error in [
            (ReplayFarmer(home=False), "not recognized"),
            (ReplayFarmer(ready_after=100), "timed out"),
        ]:
            with self.subTest(error=error), self.assertRaisesRegex(RuntimeError, error):
                SessionRunner(farmer, clock=lambda: farmer.ticks).run()
            self.assertEqual(farmer.attacks, 0)

    def test_unlimited_session_stops_on_cancellation_before_next_attack(self):
        farmer = ReplayFarmer(0)
        with self.assertRaisesRegex(RuntimeError, "Stopped by user"):
            SessionRunner(
                farmer, on_battle=lambda count: farmer.stop.set(), clock=lambda: farmer.ticks
            ).run()
        self.assertEqual(farmer.attacks, 1)

    def test_missing_readiness_and_invalid_limits_reject_session(self):
        with self.assertRaisesRegex(ValueError, "army_ready"):
            validate_session({})
        for maximum in (-1, True, 1.5):
            with self.subTest(maximum=maximum), self.assertRaisesRegex(ValueError, "max_battles"):
                validate_session(
                    {"templates": {"army_ready": "ready.png"}, "session": {"max_battles": maximum}}
                )


if __name__ == "__main__":
    unittest.main()
