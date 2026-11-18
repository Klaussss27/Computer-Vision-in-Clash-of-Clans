import threading
import unittest

from attack_policy import army_profile, order_targets, required_army
from farmer import Farmer
from vision import Detection, Target, make_plan


class GoblinOnlyTests(unittest.TestCase):
    def test_shared_entry_attacks_nearer_collector_first(self):
        near = Target(Detection("gold_mine", (90, 140, 20, 20), 1), (100, 200), 0, None, "exposed")
        far = Target(
            Detection("elixir_collector", (90, 90, 20, 20), 1), (100, 200), 0, None, "exposed"
        )
        self.assertEqual(order_targets([far, near], {"use_jump_spells": False}), [near, far])

    def test_spell_assets_are_not_required_or_loaded_when_disabled(self):
        settings = {"use_jump_spells": False, "jump_budget": 0}
        self.assertEqual(required_army(settings), ("sneaky_goblins",))
        profile = army_profile(
            {
                "templates": {"sneaky_goblins": "g.png", "jump_spell": "missing.png"},
                "count_offsets": {"sneaky_goblins": [0, 0, 20, 20], "jump_spell": [1, 1, 20, 20]},
            },
            settings,
        )
        self.assertEqual(profile["templates"], {"sneaky_goblins": "g.png"})
        self.assertNotIn("jump_spell", profile["count_offsets"])

    def test_confirmation_does_not_try_to_read_a_spell(self):
        farmer = Farmer.__new__(Farmer)
        farmer.stop_event = threading.Event()
        farmer.config = {
            "vision": {"use_jump_spells": False},
            "minimum_goblins": 3,
            "buttons": {"confirm_attack": (10, 10)},
        }
        calls = []

        class Reader:
            def read(self, image, kind):
                calls.append(kind)
                if kind != "sneaky_goblins":
                    raise AssertionError("Spell must not be read")

                class Card:
                    count = 84

                return Card()

        farmer.confirmation_reader = Reader()
        farmer.screenshot = lambda: "confirmation"
        farmer.matches_screen = lambda image, state: state == "army_confirmation"
        farmer.click = lambda point: None
        farmer.confirm_army("confirmation")
        self.assertEqual(calls, ["sneaky_goblins"])

    def test_one_wall_is_skipped_and_exposed_targets_remain_available(self):
        boundary = [(50, 50), (350, 50), (350, 350), (50, 350)]
        settings = {
            "playfield": [0, 0, 400, 400],
            "grid_size": 8,
            "use_jump_spells": False,
            "jump_budget": 0,
        }
        collector = Detection("gold_mine", (190, 190, 20, 20), 0.99)
        walls = [
            Detection("wall", box, 0.99)
            for box in [
                (120, 120, 160, 8),
                (120, 280, 160, 8),
                (120, 120, 8, 168),
                (280, 120, 8, 168),
            ]
        ]
        blocked = make_plan([collector, *walls], boundary, settings, {"gold_mine", "wall"})
        self.assertEqual(blocked.targets[0].layers, 1)
        self.assertFalse(blocked.targets[0].eligible)
        self.assertIsNone(blocked.targets[0].jump)
        exposed = make_plan(
            [collector, Detection("wall", (70, 70, 8, 8), 0.99)],
            boundary,
            settings,
            {"gold_mine", "wall"},
        )
        self.assertTrue(exposed.targets[0].eligible)
        self.assertEqual(exposed.targets[0].layers, 0)


if __name__ == "__main__":
    unittest.main()
