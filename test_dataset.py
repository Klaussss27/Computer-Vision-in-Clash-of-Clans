import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from dataset import Dataset, prepare_dataset, yolo_line
from neural_detector import NeuralDetector
from prepare_reviewed_scouting import prepare as prepare_reviewed
from vision import COLLECTORS, KINDS


class DatasetTests(unittest.TestCase):
    def test_reviewed_snapshot_excludes_drafts_and_preserves_originals(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dataset = Dataset(root / "source")
            for index, role in enumerate(("train", "val", "test", "train")):
                record = dataset.frame(Image.new("RGB", (100, 100), (index * 50, 0, 0)))
                record.update(
                    group=f"base-{index}",
                    split=role,
                    reviewed=index < 3,
                    grouping_reviewed=index < 3,
                    reviewed_classes=list(COLLECTORS),
                    objects=[{"kind": k, "box": [10, 10, 30, 30]} for k in COLLECTORS],
                )
                dataset.save(record)
            original = {
                p.name: p.read_bytes() for p in (dataset.root / "annotations").glob("*.json")
            }
            spec, report = prepare_reviewed(dataset.root, root / "snapshot", root / "exports")
            self.assertEqual(report["excluded_frames"], 1)
            self.assertEqual(report["scouting_splits"]["frames"], {"train": 1, "val": 1, "test": 1})
            self.assertTrue(spec.exists())
            self.assertEqual(
                original,
                {p.name: p.read_bytes() for p in (dataset.root / "annotations").glob("*.json")},
            )

    def test_reserved_splits_survive_export_and_reject_base_leakage(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset = Dataset(Path(folder) / "source")
            records = []
            for index, role in enumerate(("train", "val", "test")):
                record = dataset.frame(Image.new("RGB", (100, 100), (index * 70, 0, 0)))
                record.update(
                    group=f"base-{index}",
                    split=role,
                    reviewed=True,
                    grouping_reviewed=True,
                    reviewed_classes=list(COLLECTORS),
                    objects=[{"kind": kind, "box": [10, 10, 30, 30]} for kind in COLLECTORS],
                )
                dataset.save(record)
                records.append(record)
            spec, report = prepare_dataset(
                dataset.root, Path(folder) / "exports", classes=COLLECTORS
            )
            self.assertEqual(report["frames"], {"train": 1, "val": 1, "test": 1})
            self.assertEqual(json.loads(spec.read_text())["test"], "images/test")
            records[2]["group"] = records[0]["group"]
            dataset.save(records[2])
            with self.assertRaisesRegex(ValueError, "same base"):
                prepare_dataset(dataset.root, Path(folder) / "exports", classes=COLLECTORS)
            records[2]["group"] = "base-2"
            records[2]["reviewed"] = False
            dataset.save(records[2])
            with self.assertRaisesRegex(ValueError, "all boxes"):
                prepare_dataset(dataset.root, Path(folder) / "exports", classes=COLLECTORS)

    def test_collector_only_review_cannot_be_exported_as_wall_training(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset = Dataset(Path(folder) / "source")
            for index in range(3):
                frame = dataset.frame(Image.new("RGB", (100, 100), (index * 80, 100, 50)))
                frame.update(
                    group=f"base-{index}",
                    reviewed=True,
                    reviewed_classes=list(COLLECTORS),
                    objects=[{"kind": kind, "box": [10, 10, 20, 20]} for kind in COLLECTORS],
                )
                dataset.save(frame)
            spec, _ = prepare_dataset(dataset.root, Path(folder) / "exports", classes=COLLECTORS)
            self.assertEqual(json.loads(spec.read_text())["names"], list(COLLECTORS))
            with self.assertRaisesRegex(ValueError, "at least two"):
                prepare_dataset(dataset.root, Path(folder) / "full-export")

    def test_box_coordinates_export_as_normalized_yolo(self):
        self.assertEqual(
            yolo_line({"kind": "gold_mine", "box": [10, 20, 30, 60]}, (100, 100)),
            "0 0.20000000 0.40000000 0.20000000 0.40000000",
        )
        with self.assertRaises(ValueError):
            yolo_line({"kind": "wall", "box": [-1, 0, 5, 5]}, (100, 100))

    def test_duplicate_frame_reuses_annotations(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Dataset(directory)
            image = Image.new("RGB", (100, 100), "green")
            frame = dataset.frame(image)
            frame["objects"].append({"kind": "wall", "box": [1, 2, 8, 9]})
            dataset.save(frame)
            self.assertEqual(dataset.frame(image)["objects"], frame["objects"])
            self.assertEqual(len(dataset.records()), 1)

    def test_export_excludes_unreviewed_frames_and_keeps_bases_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Dataset(Path(directory) / "source")
            for index in range(7):
                frame = dataset.frame(Image.new("RGB", (100, 100), (index * 30, 0, 0)))
                frame.update(
                    group="base-" + str(index // 2),
                    reviewed=index < 6,
                    objects=[{"kind": kind, "box": [1, 2, 10, 20]} for kind in KINDS],
                )
                dataset.save(frame)
            specification, report = prepare_dataset(dataset.root, Path(directory) / "exports")
            self.assertEqual(sum(report["frames"].values()), 6)
            self.assertFalse(set(report["groups"]["train"]) & set(report["groups"]["val"]))
            exported = json.loads(specification.read_text())
            self.assertEqual(exported["names"], list(KINDS))
            self.assertTrue(list((specification.parent / "labels" / "train").glob("*.txt")))

    def test_no_reviewed_data_cannot_start_training(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "at least two"):
                prepare_dataset(directory, Path(directory) / "exports")


class InferenceTests(unittest.TestCase):
    def test_predictions_map_back_to_full_window_coordinates(self):
        class Tensor:
            def __init__(self, data):
                self.data = data

            def cpu(self):
                return self

            def tolist(self):
                return self.data

        class Boxes:
            xyxy = Tensor([[5, 6, 25, 36]])
            cls = Tensor([0])
            conf = Tensor([0.95])

        class Result:
            boxes = Boxes()
            names = {0: "gold_mine"}

        class Model:
            def predict(self, image, **kwargs):
                self.size = image.size
                return [Result()]

        detector = NeuralDetector.__new__(NeuralDetector)
        detector.settings = {"playfield": [20, 30, 100, 100]}
        detector.model = Model()
        found = detector.detect(Image.new("RGB", (200, 200)))
        self.assertEqual(detector.model.size, (100, 100))
        self.assertEqual(found[0].box, (25, 36, 20, 30))
        self.assertEqual(found[0].kind, "gold_mine")
        detector.settings["model_edge_margin"] = 8
        self.assertEqual(detector.detect(Image.new("RGB", (200, 200))), [])

    def test_missing_checkpoint_does_not_download_a_general_model(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Trained detector missing"):
                NeuralDetector({"model_path": str(Path(directory) / "missing.pt")})


if __name__ == "__main__":
    unittest.main()
