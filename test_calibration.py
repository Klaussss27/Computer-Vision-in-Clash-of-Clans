import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from calibration import readiness, save_button, save_marker


class CalibrationTests(unittest.TestCase):
    def test_saved_frame_calibration_updates_only_requested_assets(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = {
                "capture_size": [200, 100],
                "templates": {"scouting": "existing.png"},
                "buttons": {"next": [50, 50]},
            }
            (root / "config.json").write_text(json.dumps(config))
            image = root / "saved.png"
            Image.fromarray(
                np.random.default_rng(3).integers(0, 255, (100, 200, 3), dtype=np.uint8)
            ).save(image)
            save_marker(image, "home", [10, 20, 30, 40], root)
            save_button("attack", [40, 60], root)
            after = json.loads((root / "config.json").read_text())
            self.assertEqual(after["templates"]["scouting"], "existing.png")
            self.assertEqual(after["buttons"]["next"], [50, 50])
            self.assertEqual(after["buttons"]["attack"], [40, 60])
            with Image.open(root / after["templates"]["home"]) as marker:
                self.assertEqual(marker.size, (30, 40))
            with self.assertRaisesRegex(ValueError, "inside"):
                save_button("attack", [200, 60], root)
            with self.assertRaisesRegex(ValueError, "inside"):
                save_marker(image, "results", [190, 20, 30, 40], root)

    def test_readiness_lists_all_missing_states_without_capturing(self):
        with tempfile.TemporaryDirectory() as folder:
            report = readiness({}, Path(folder), session=True)
            self.assertFalse(report["configured"])
            self.assertIn("State marker: army_ready", report["missing"])
            self.assertIn("Button position: return_home", report["missing"])


if __name__ == "__main__":
    unittest.main()
