import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from neural_detector import HybridDetector


class HybridDetectorTests(unittest.TestCase):
    def test_wall_pack_preserves_original_rubble_references(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rng = np.random.default_rng(8)
            for relative in ("objects/rubble", "objects/wall", "walls/wall"):
                target = root / relative
                target.mkdir(parents=True)
                Image.fromarray(rng.integers(0, 255, (12, 12, 3), dtype=np.uint8)).save(
                    target / "sample.png"
                )
            settings = {
                "template_dir": str(root / "objects"),
                "structure_template_dir": str(root / "walls"),
                "playfield": [0, 0, 100, 100],
                "scales": [1],
                "confidence": 0.99,
            }
            with patch("neural_detector.NeuralDetector") as model:
                model.return_value.kinds = {"gold_mine", "elixir_collector", "dark_drill"}
                model.return_value.detect.return_value = []
                detector = HybridDetector(settings)
            self.assertEqual([kind for kind, _ in detector.structures.samples], ["wall", "rubble"])
            self.assertIn("rubble", detector.kinds)
            image = Image.new("RGB", (100, 100))
            with Image.open(root / "objects/rubble/sample.png") as sample:
                image.paste(sample, (60, 60))
            self.assertFalse(any(d.kind == "rubble" for d in detector.detect_for_planning(image)))
            rubble = detector.detect_rubble(image, (50, 50, 40, 40))
            self.assertEqual([(d.kind, d.box) for d in rubble], [("rubble", (60, 60, 12, 12))])


if __name__ == "__main__":
    unittest.main()
