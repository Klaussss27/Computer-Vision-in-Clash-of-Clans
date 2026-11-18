import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from army import ArmyCard, ArmyReader
from farmer import Farmer
from vision import Detection, Plan, Target


class FakeOCR:
    def __init__(self, values=("84", "84")):
        self.values = iter(values)

    def image_to_string(self, *args, **kwargs):
        return next(self.values)


class ArmyTests(unittest.TestCase):
    def test_deployment_click_rejects_a_changed_screen(self):
        farmer = Farmer.__new__(Farmer)
        farmer.screenshot = lambda: "dialog"
        farmer.matches_screen = lambda image, state: False
        clicks = []
        farmer.click = lambda point: clicks.append(point)
        farmer.config = {"buttons": {"next": (10, 10)}}
        with self.assertRaisesRegex(RuntimeError, "Screen changed"):
            farmer.click_button_on_state("next", "scouting")
        self.assertEqual(clicks, [])

    def test_empty_portrait_requires_a_zero_header_marker(self):
        with tempfile.TemporaryDirectory() as folder:
            reader, image = self.fixture(folder, FakeOCR())
            sample = reader.samples["sneaky_goblins"]
            reader.samples["sneaky_goblins"] = np.random.default_rng(4).integers(
                0, 255, sample.shape, dtype=np.uint8
            )
            reader.empty_samples["sneaky_goblins"] = sample
            marker = np.random.default_rng(5).integers(0, 255, (15, 20), dtype=np.uint8)
            reader.zero_samples["sneaky_goblins"] = marker
            reader.settings["empty_count_offsets"] = {"sneaky_goblins": [18, -42, 66, 28]}
            image.paste(Image.fromarray(marker).convert("RGB"), (120, 62))
            self.assertEqual(reader.read(image, "sneaky_goblins").count, 0)
            image.paste("black", (118, 58, 184, 86))
            with self.assertRaisesRegex(RuntimeError, "zero-count"):
                reader.read(image, "sneaky_goblins")

    def test_confirmation_checks_counts_before_starting_matchmaking(self):
        farmer = Farmer.__new__(Farmer)
        farmer.stop_event = threading.Event()
        farmer.config = {"buttons": {"confirm_attack": (1625, 921)}, "minimum_goblins": 3}
        clicks = []
        farmer.click = lambda point: clicks.append(point)
        farmer.screenshot = lambda: "saved-confirmation"
        farmer.matches_screen = lambda image, state: state == "army_confirmation"

        class Reader:
            count = 0

            def read(self, image, kind):
                return ArmyCard(kind, (0, 0), self.count if kind == "sneaky_goblins" else 4, 0.99)

        farmer.confirmation_reader = Reader()
        with self.assertRaisesRegex(RuntimeError, "Matchmaking not started"):
            farmer.confirm_army("saved-confirmation")
        self.assertEqual(clicks, [])
        farmer.confirmation_reader.count = 84
        farmer.confirm_army("saved-confirmation")
        self.assertEqual(clicks, [(1625, 921)])

    def fixture(self, folder, ocr, duplicate=False):
        sample = np.random.default_rng(1).integers(0, 255, (50, 70, 3), dtype=np.uint8)
        path = Path(folder) / "icon.png"
        Image.fromarray(sample).save(path)
        image = Image.new("RGB", (400, 200))
        image.paste(Image.fromarray(sample), (100, 100))
        if duplicate:
            image.paste(Image.fromarray(sample), (250, 100))
        reader = ArmyReader(
            {"region": [0, 0, 400, 200], "templates": {"sneaky_goblins": str(path)}}, ocr
        )
        return reader, image

    def test_card_position_comes_from_icon_and_agreed_counts(self):
        with tempfile.TemporaryDirectory() as folder:
            reader, image = self.fixture(folder, FakeOCR())
            card = reader.read(image, "sneaky_goblins")
            self.assertEqual(card.point, (135, 125))
            self.assertEqual(card.count, 84)

    def test_ambiguous_icons_or_disagreeing_counts_reject_selection(self):
        with tempfile.TemporaryDirectory() as folder:
            reader, image = self.fixture(folder, FakeOCR(), True)
            with self.assertRaisesRegex(RuntimeError, "Ambiguous"):
                reader.read(image, "sneaky_goblins")
            reader, image = self.fixture(folder, FakeOCR(("84", "64")))
            with self.assertRaisesRegex(RuntimeError, "disagree"):
                reader.read(image, "sneaky_goblins")
            with self.assertRaisesRegex(RuntimeError, "not confidently"):
                reader.read(Image.new("RGB", (400, 200)), "sneaky_goblins")

    def test_attack_uses_recognized_cards_and_never_exceeds_visible_count(self):
        farmer = Farmer.__new__(Farmer)
        farmer.stop_event = threading.Event()
        farmer.config = {
            "vision": {
                "goblin_budget": 10,
                "jump_budget": 1,
                "max_targets": 8,
                "max_batches_per_target": 1,
                "goblins_per_batch": 3,
                "target_wait_seconds": 1,
            },
            "buttons": {"end_battle": (1, 1), "confirm_surrender": (2, 2)},
            "army_slots": {"sneaky_goblins": (999, 999), "jump_spell": (998, 998)},
        }
        collector = Detection("gold_mine", (100, 100, 20, 20), 0.99)
        target = Target(collector, (40, 70), 1, (80, 80), "one layer")
        clicks = []
        farmer.click = lambda point: clicks.append(point)
        farmer.screenshot = lambda: "saved-frame"
        farmer.matches_screen = lambda image, state: state in ("battle", "surrender")
        farmer.wait_for_screen = lambda *args: None
        farmer.sleep = lambda seconds: None

        class Reader:
            def read(self, image, kind):
                return ArmyCard(
                    kind,
                    (734, 997) if kind == "jump_spell" else (337, 995),
                    1 if kind == "jump_spell" else 2 - clicks.count((40, 70)),
                    0.99,
                )

        class Detector:
            def detect(self, image):
                return [collector]

        class Planner:
            detector = Detector()

            def destroyed(self, image, object):
                return True

            def aligned(self, image):
                return True

        farmer.army_reader = Reader()
        farmer.visual_attack(Plan([collector], [target], [], []), Planner())
        self.assertEqual(
            clicks, [(734, 997), (80, 80), (337, 995), (40, 70), (40, 70), (1, 1), (2, 2)]
        )
        farmer.config["vision"]["use_jump_spells"] = False
        clicks.clear()
        with self.assertRaisesRegex(RuntimeError, "No targets allowed"):
            farmer.visual_attack(Plan([collector], [target], [], []), Planner())
        self.assertEqual(clicks, [])


if __name__ == "__main__":
    unittest.main()
