import threading
import unittest

from attacks import run_attack, validate_attack
from farmer import Farmer


class AttackTests(unittest.TestCase):
    def test_stop_blocks_clicks_and_interrupts_waits(self):
        farmer = Farmer.__new__(Farmer)
        farmer.stop_event = threading.Event()
        farmer.stop_event.set()
        with self.assertRaisesRegex(RuntimeError, "Stopped by user"):
            farmer.click([1, 1])
        with self.assertRaisesRegex(RuntimeError, "Stopped by user"):
            farmer.sleep(240)

    def test_selection_deployment_and_wait_keep_their_order(self):
        steps = [
            {"action": "click", "point": [300, 670]},
            {"action": "click", "point": [100, 300], "count": 2, "interval": 0.2},
            {"action": "wait", "seconds": 5},
        ]
        validate_attack(steps, (1280, 720))
        events = []
        run_attack(
            steps, lambda p: events.append(("click", p)), lambda s: events.append(("wait", s))
        )
        self.assertEqual(
            events,
            [
                ("click", [300, 670]),
                ("wait", 0.3),
                ("click", [100, 300]),
                ("wait", 0.2),
                ("click", [100, 300]),
                ("wait", 0.2),
                ("wait", 5),
            ],
        )

    def test_invalid_steps_are_rejected_before_input(self):
        for step in [
            {"action": "click", "point": [1280, 0]},
            {"action": "click", "point": [1, 1], "count": -1},
            {"action": "click", "point": [1, 1], "interval": -1},
            {"action": "wait", "seconds": -1},
            {"action": "unknown"},
        ]:
            with self.subTest(step=step), self.assertRaises(ValueError):
                validate_attack([step], (1280, 720))


if __name__ == "__main__":
    unittest.main()
