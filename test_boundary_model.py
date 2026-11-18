import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from boundary_model import features, learned_edges, load_model, network


class BoundaryModelTests(unittest.TestCase):
    def test_checkpoint_inference_preserves_size_and_rejects_wrapped_border_pixels(self):
        rgb = np.zeros((32, 48, 3), np.uint8)
        values = features(rgb)
        self.assertEqual(values.shape, (32, 48, 14))
        self.assertTrue(np.isfinite(values).all())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.pt"
            model = network()
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
            torch.save({"format": "boundary-pixels-v1", "weights": model.state_dict()}, path)
            mask = learned_edges(rgb, path, 0.2)
            self.assertEqual(mask.shape, (32, 48))
            self.assertEqual(mask.dtype, np.uint8)
            self.assertTrue((mask[:7] == 0).all())
            self.assertTrue((mask[7:-7, 7:-7] == 255).all())
            self.assertFalse(learned_edges(rgb, path, 0.8).any())
        load_model.cache_clear()

    def test_invalid_threshold_and_checkpoint_format_are_rejected(self):
        rgb = np.zeros((32, 48, 3), np.uint8)
        with self.assertRaises(ValueError):
            learned_edges(rgb, "missing", 1)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "wrong.pt"
            torch.save({"format": "unsupported"}, path)
            with self.assertRaisesRegex(ValueError, "Unsupported boundary checkpoint"):
                learned_edges(rgb, path)
        load_model.cache_clear()


if __name__ == "__main__":
    unittest.main()
