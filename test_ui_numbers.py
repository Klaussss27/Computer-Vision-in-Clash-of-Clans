import unittest

from PIL import Image

from ui_numbers import read_number


class NumberTests(unittest.TestCase):
    def test_two_preparations_must_agree_and_bad_regions_reject(self):
        class OCR:
            values = iter(["1 059 920\n", "1059920\n"])

            def image_to_string(self, *args, **kwargs):
                return next(self.values)

        image = Image.new("RGB", (100, 80), "white")
        self.assertEqual(read_number(image, (10, 10, 50, 30), OCR()), 1059920)

        class Disagree:
            values = iter(["1059320", "1059920"])

            def image_to_string(self, *args, **kwargs):
                return next(self.values)

        with self.assertRaisesRegex(RuntimeError, "disagree"):
            read_number(image, (10, 10, 50, 30), Disagree())
        with self.assertRaisesRegex(ValueError, "fit"):
            read_number(image, (99, 10, 50, 30), OCR())
