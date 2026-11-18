import unittest

from prepare_video_training import crop_labels


class VideoTrainingTests(unittest.TestCase):
    def test_destroyed_collectors_become_negatives_but_drills_remain_labeled(self):
        objects = [
            {"kind": "gold_mine", "box": [110, 210, 40, 50], "destroyed_after": True},
            {"kind": "dark_drill", "box": [160, 230, 30, 40], "destroyed_after": False},
        ]
        roi = (100, 200, 120, 120)
        before = crop_labels(objects, roi, False)
        after = crop_labels(objects, roi, True)
        self.assertEqual(len(before), 2)
        self.assertEqual(len(after), 1)
        self.assertEqual(after[0].split()[0], "2")
        values = list(map(float, before[0].split()[1:]))
        self.assertAlmostEqual(values[0], 0.25)
        self.assertAlmostEqual(values[2], 1 / 3)

    def test_partial_collector_crops_are_rejected_including_destroyed_targets(self):
        objects = [{"kind": "gold_mine", "box": [90, 210, 40, 50], "destroyed_after": True}]
        for after in (False, True):
            with self.assertRaisesRegex(ValueError, "cuts through"):
                crop_labels(objects, (100, 200, 120, 120), after)
        self.assertEqual(crop_labels(objects, (500, 500, 100, 100)), [])


if __name__ == "__main__":
    unittest.main()
