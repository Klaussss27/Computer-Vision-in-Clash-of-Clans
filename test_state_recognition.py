import unittest

import cv2
import numpy as np
from PIL import Image

from state_recognition import marker_match


class StateRecognitionTests(unittest.TestCase):
    def test_variants_preserve_dimmed_background_rejection(self):
        rgb = np.random.default_rng(18).integers(40, 250, (30, 50, 3), dtype=np.uint8)
        marker = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        variants = [np.zeros((20, 20), np.uint8), marker]
        self.assertTrue(marker_match(Image.fromarray(rgb), variants))
        self.assertFalse(marker_match(Image.fromarray((rgb * 0.5).astype(np.uint8)), variants))

    def test_dimmed_modal_background_does_not_count_as_active_home(self):
        rgb = np.random.default_rng(14).integers(40, 250, (70, 100, 3), dtype=np.uint8)
        template = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        self.assertTrue(marker_match(Image.fromarray(rgb), template))
        dimmed = Image.fromarray((rgb * 0.5).astype(np.uint8))
        self.assertFalse(marker_match(dimmed, template))

    def test_shifted_marker_still_matches_and_invalid_sizes_reject(self):
        rgb = np.random.default_rng(15).integers(0, 255, (20, 30, 3), dtype=np.uint8)
        template = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        image = Image.new("RGB", (200, 100))
        image.paste(Image.fromarray(rgb), (100, 40))
        self.assertTrue(marker_match(image, template))
        self.assertFalse(marker_match(Image.new("RGB", (10, 10)), template))
        self.assertFalse(marker_match(image, np.zeros((20, 30), np.uint8)))


if __name__ == "__main__":
    unittest.main()
