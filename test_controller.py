"""Exercise the complete controller against a simulated game, without input."""

import threading
import unittest
from unittest.mock import Mock, patch

from army import ArmyCard
from farmer import Farmer
from vision import Detection, Plan, Target


class ControllerTests(unittest.TestCase):
    def deployment_fixture(self, counts):
        farmer = Farmer.__new__(Farmer)
        farmer.sleep = lambda seconds: None
        farmer.screenshot = lambda: "battle"
        farmer.matches_screen = lambda image, state: image == state
        iterator = iter(counts)
        farmer.army_card = lambda image, kind: ArmyCard(kind, (20, 20), next(iterator), 0.99)
        return farmer

    def test_deployment_confirmation_waits_for_the_actual_count(self):
        farmer = self.deployment_fixture([84, 83, 82])
        with patch("farmer.time.monotonic", side_effect=[0, 0, 1, 2]):
            self.assertTrue(farmer.confirm_goblin_deployment(82))

    def test_rejected_deployment_and_unexpected_consumption_stop(self):
        farmer = self.deployment_fixture([84])
        with patch("farmer.time.monotonic", side_effect=[0, 0, 4]):
            with self.assertRaisesRegex(RuntimeError, "deployment not confirmed"):
                farmer.confirm_goblin_deployment(82)
        farmer = self.deployment_fixture([81])
        with self.assertRaisesRegex(RuntimeError, "Unexpected Goblin count decrease"):
            farmer.confirm_goblin_deployment(82)

    def test_results_during_deployment_end_the_pass(self):
        farmer = self.deployment_fixture([])
        farmer.screenshot = lambda: "results"
        self.assertFalse(farmer.confirm_goblin_deployment(82))

    def test_goblin_only_attack_returns_home_and_reports_loot(self):
        farmer = Farmer.__new__(Farmer)
        farmer.live = True
        farmer.stop_event = threading.Event()
        buttons = {
            "attack": (1, 1),
            "find_match": (2, 2),
            "confirm_attack": (3, 3),
            "end_battle": (4, 4),
            "confirm_surrender": (5, 5),
            "return_home": (6, 6),
            "next": (7, 7),
        }
        settings = {
            "use_jump_spells": False,
            "goblin_budget": 2,
            "max_targets": 1,
            "goblins_per_batch": 3,
            "max_batches_per_target": 1,
            "target_wait_seconds": 0,
        }
        farmer.config = {
            "buttons": buttons,
            "vision": settings,
            "attack_mode": "visual",
            "max_searches": 1,
            "minimum_loot": {"gold": 1},
            "minimum_goblins": 1,
            "battle_timeout_seconds": 2,
            "result_regions": {"gold": [0, 0, 1, 1]},
        }
        current = ["home"]
        transitions = {
            buttons["attack"]: "attack_menu",
            buttons["find_match"]: "army_confirmation",
            buttons["confirm_attack"]: "scouting",
            buttons["end_battle"]: "surrender",
            buttons["confirm_surrender"]: "results",
            buttons["return_home"]: "home",
            (50, 50): "battle",
        }
        clicks = []

        def click(point):
            clicks.append(point)
            current[0] = transitions.get(point, current[0])

        farmer.click = click
        farmer.screenshot = lambda: current[0]
        farmer.matches_screen = lambda image, state: image == state
        farmer.sleep = lambda seconds: None
        farmer.read_loot = lambda image: {"gold": 100}
        farmer.read_results = lambda image: {"gold": 42}
        farmer.on_result = Mock()
        activities = []
        farmer.on_activity = activities.append
        collector = Detection("gold_mine", (100, 100, 30, 30), 0.99)
        plan = Plan([collector], [Target(collector, (50, 50), 0, None, "exposed")], [], [])
        planner = Mock()
        planner.view_ready.return_value = True
        planner.analyze.return_value = plan
        planner.aligned.return_value = True
        planner.detector.detect_collectors.return_value = [collector]
        kinds = []

        class Reader:
            def read(self, image, kind):
                kinds.append(kind)
                if kind != "sneaky_goblins":
                    raise AssertionError("Spell read in Goblins-only attack")
                return ArmyCard(kind, (20, 20), 84 - clicks.count((50, 50)), 0.99)

        farmer.confirmation_reader = farmer.army_reader = Reader()
        with (
            patch("farmer.validate_config"),
            patch("vision.VisualPlanner", return_value=planner),
            patch("vision.draw_plan", return_value=Mock()),
            patch("farmer.Path"),
        ):
            farmer.run()
        self.assertEqual(current[0], "home")
        self.assertEqual(
            clicks, [(1, 1), (2, 2), (3, 3), (20, 20), (50, 50), (50, 50), (4, 4), (5, 5), (6, 6)]
        )
        farmer.on_result.assert_called_once_with({"gold": 42})
        self.assertTrue(kinds)
        self.assertTrue(any(e["action"] == "Deploy 2 Sneaky Goblins" for e in activities))
        self.assertTrue(any(e["action"] == "Troop deployment confirmed" for e in activities))
        self.assertTrue(
            all({"stage", "action", "evidence", "next_action"} <= set(e) for e in activities)
        )


if __name__ == "__main__":
    unittest.main()
