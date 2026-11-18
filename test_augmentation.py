import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw

from augment_collectors import boxes_overlap, build, load_assets
from calibrate_detector import select_thresholds
from evaluate_saved import score_detections
from neural_detector import confidence_settings
from train_detector import completed_epochs
from vision import Detection


class AugmentationTests(unittest.TestCase):
    def fixture(self, root):
        source = root / "source"
        manifest = []
        for split in ("train", "val", "test"):
            (source / "images" / split).mkdir(parents=True)
            (source / "labels" / split).mkdir(parents=True)
            Image.new("RGB", (200, 200), "#48602c").save(source / "images" / split / "base.png")
            labels = ["0 0.1 0.1 0.1 0.1"]
            (source / "labels" / split / "base.txt").write_text(labels[0] + "\n")
            manifest.append({"split": split, "stem": "base", "extension": ".png", "labels": labels})
        (source / "dataset.yaml").write_text(
            json.dumps(
                {
                    "path": str(source),
                    "train": "images/train",
                    "val": "images/val",
                    "test": "images/test",
                    "names": ["gold_mine", "elixir_collector", "dark_drill"],
                }
            )
        )
        (source / "manifest.json").write_text(json.dumps(manifest))
        (source / "ATTRIBUTION.md").write_text("Fixture attribution")
        archive = root / "assets.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            categories = [
                {"id": index, "name": name}
                for index, name in enumerate(
                    ["gold_mine", "elexir_collector", "dark_elexir_drill", "gold_storage"]
                )
            ]
            images, annotations = [], []
            for index, category in enumerate(categories):
                image = Image.new("RGB", (40, 40), "black")
                ImageDraw.Draw(image).rectangle((4, 4, 35, 35), fill=(40 + index * 50, 180, 80))
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                name = f"asset-{index}.png"
                bundle.writestr("train/" + name, buffer.getvalue())
                images.append(
                    {"id": index, "file_name": name, "extra": {"user_tags": ["building"]}}
                )
                annotations.append(
                    {
                        "image_id": index,
                        "category_id": category["id"],
                        "bbox": [4, 4, 32, 32],
                        "segmentation": [[4, 4, 35, 4, 35, 35, 4, 35]],
                    }
                )
            bundle.writestr(
                "train/_annotations.coco.json",
                json.dumps(
                    {"images": images, "annotations": annotations, "categories": categories}
                ),
            )
        return source, archive

    def test_augmented_data_preserves_held_out_images_and_uses_training_assets(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, archive = self.fixture(root)
            assets = load_assets(archive)
            self.assertEqual(
                set(assets), {"gold_mine", "elixir_collector", "dark_drill", "negative"}
            )
            spec, report = build(source, archive, root / "result", count=2)
            self.assertEqual(len(list((spec.parent / "images/train").glob("*"))), 3)
            self.assertFalse(report["validation_and_test_changed"])
            for relative, digest in report["held_out_hashes"].items():
                self.assertEqual(
                    hashlib.sha256((spec.parent / relative).read_bytes()).hexdigest(), digest
                )
            for path in (spec.parent / "labels/train").glob("*.txt"):
                for line in path.read_text().splitlines():
                    values = list(map(float, line.split()))
                    self.assertIn(values[0], (0, 1, 2))
                    self.assertTrue(all(0 <= v <= 1 for v in values[1:]))

    def test_overlap_reserves_existing_collector_footprints(self):
        self.assertTrue(boxes_overlap((5, 5, 15, 15), (10, 10, 20, 20)))
        self.assertFalse(boxes_overlap((0, 0, 5, 5), (20, 20, 30, 30)))


class SavedEvaluationTests(unittest.TestCase):
    def test_final_validation_does_not_count_an_extra_training_epoch(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "results.csv").write_text("epoch,fitness\n1,0.2\n2,0.3\n")
            self.assertEqual(completed_epochs(folder, 3, 10), 2)

    def test_threshold_selection_enforces_precision_target(self):
        frame = {
            "truth": [
                {"kind": "gold_mine", "box": [0, 0, 10, 10]},
                {"kind": "gold_mine", "box": [20, 20, 10, 10]},
            ],
            "predictions": [
                {"kind": "gold_mine", "box": [0, 0, 10, 10], "score": 0.9},
                {"kind": "gold_mine", "box": [40, 40, 10, 10], "score": 0.7},
                {"kind": "gold_mine", "box": [20, 20, 10, 10], "score": 0.6},
            ],
        }
        chosen, _ = select_thresholds([frame], 0.98)
        self.assertTrue(chosen["gold_mine"]["precision_target_met"])
        self.assertGreater(chosen["gold_mine"]["threshold"], 0.7)
        self.assertEqual(chosen["gold_mine"]["fp"], 0)
        self.assertFalse(chosen["dark_drill"]["precision_target_met"])

    def test_invalid_or_misspelled_thresholds_are_rejected(self):
        for settings in (
            {"model_confidence": 0},
            {"model_class_confidence": {"mine": 0.5}},
            {"model_class_confidence": {"gold_mine": float("nan")}},
        ):
            with self.assertRaises(ValueError):
                confidence_settings(settings, ("gold_mine", "elixir_collector", "dark_drill"))
        default, values, minimum = confidence_settings(
            {"model_class_confidence": {"gold_mine": 0.3}}, ("gold_mine", "dark_drill")
        )
        self.assertEqual((default, minimum), (0.6, 0.3))

    def test_duplicate_predictions_cannot_count_twice(self):
        truth = [
            {"kind": "gold_mine", "box": [10, 10, 20, 20]},
            {"kind": "dark_drill", "box": [50, 50, 20, 20]},
        ]
        predictions = [
            Detection("gold_mine", (10, 10, 20, 20), 0.9),
            Detection("gold_mine", (10, 10, 20, 20), 0.8),
        ]
        result = score_detections(predictions, truth)
        self.assertEqual(
            (result["true_positive"], result["false_positive"], result["false_negative"]), (1, 1, 1)
        )

    def test_matching_requires_class_and_location(self):
        result = score_detections(
            [Detection("elixir_collector", (10, 10, 20, 20), 0.9)],
            [{"kind": "gold_mine", "box": [10, 10, 20, 20]}],
        )
        self.assertEqual(result["true_positive"], 0)


if __name__ == "__main__":
    unittest.main()
