import unittest

import cv2
import numpy as np
from PIL import Image

from capture_calibration import wait_for_marker


class CaptureWaitTests(unittest.TestCase):
    def test_transition_frames_are_skipped_until_requested_screen_matches(self):
        rgb = np.random.default_rng(21).integers(0, 255, (40, 60, 3), dtype=np.uint8)
        template = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        expected = Image.fromarray(rgb)

        class Window:
            frames = iter([Image.new("RGB", (60, 40), "gray"), expected])
            reads = 0

            def screenshot(self):
                self.reads += 1
                return next(self.frames)

        window = Window()
        result = wait_for_marker(window, template, clock=lambda: 0, sleep=lambda seconds: None)
        self.assertIs(result, expected)
        self.assertEqual(window.reads, 2)

    def test_timeout_saves_no_unmatched_frame(self):
        clock = [0.0]
        template = np.random.default_rng(22).integers(0, 255, (20, 20), dtype=np.uint8)

        class Window:
            def screenshot(self):
                return Image.new("RGB", (40, 40), "gray")

        def sleep(seconds):
            clock[0] += seconds

        with self.assertRaisesRegex(RuntimeError, "no calibration image saved"):
            wait_for_marker(Window(), template, timeout=0.5, clock=lambda: clock[0], sleep=sleep)


if __name__ == "__main__":
    unittest.main()
