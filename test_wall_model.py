import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from neural_detector import HybridDetector
from train_walls import sparse_loss
from vision import validate_vision
from wall_model import WallNet, WallSegmenter, wall_cells


class WallModelTests(unittest.TestCase):
    def test_unknown_pixels_have_no_loss_or_gradient(self):
        prediction = torch.tensor([[0.0, 0.0, 100000.0]], requires_grad=True)
        labels = torch.tensor([[1.0, 0.0, -1.0]])
        loss = sparse_loss(prediction, labels)
        self.assertAlmostEqual(float(loss.detach()), np.log(2), places=6)
        loss.backward()
        self.assertEqual(float(prediction.grad[0, 2]), 0)
        with self.assertRaisesRegex(ValueError, "No reviewed"):
            sparse_loss(prediction, torch.full_like(labels, -1))

    def test_wall_rasterization_preserves_enclosure_holes_and_exclusions(self):
        p = np.ones((32, 32), np.float32)
        p[8:24, 8:24] = 0
        detections = wall_cells(p, (10, 20, 32, 32), excluded=[(10, 20, 8, 32)])

        def covered(x, y):
            return any(
                bx <= x < bx + bw and by <= y < by + bh
                for d in detections
                for bx, by, bw, bh in [d.box]
            )

        self.assertFalse(covered(26, 36))
        self.assertFalse(covered(14, 24))
        self.assertTrue(covered(38, 36))

    def test_tiled_inference_preserves_odd_image_dimensions(self):
        with tempfile.TemporaryDirectory() as folder:
            model = WallNet()
            for parameter in model.parameters():
                parameter.data.zero_()
            path = Path(folder) / "wall.pt"
            torch.save({"architecture": "wallnet-v1", "state_dict": model.state_dict()}, path)
            segmenter = WallSegmenter(path)
            p = segmenter.probabilities(Image.new("RGB", (35, 29)), tile=24, margin=8)
            self.assertEqual(p.shape, (29, 35))
            np.testing.assert_allclose(p, 0.5)

    def test_wall_network_and_preview_profile_cannot_enable_live_attacks(self):
        with self.assertRaisesRegex(ValueError, "offline previews only"):
            HybridDetector({"wall_model_path": "missing.pt"})
        with self.assertRaisesRegex(ValueError, "cannot be used for live"):
            validate_vision({"vision": {"planning_preview_only": True}})


if __name__ == "__main__":
    unittest.main()
