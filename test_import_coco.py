import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

from import_coco import CLASSES, convert_box, import_archive
from prepare_baseline import prepare


def make_archive(path, unsafe=False):
    categories = [
        {"id": 13, "name": "dark_elexir_drill"},
        {"id": 17, "name": "elexir_collector"},
        {"id": 19, "name": "gold_mine"},
        {"id": 20, "name": "gold_storage"},
    ]
    with zipfile.ZipFile(path, "w") as bundle:
        for index, split in enumerate(("train", "valid", "test")):
            images, annotations = [], []
            for image_id, tag in enumerate(("village", "building")):
                name = (
                    "../escape.jpg"
                    if unsafe and split == "test" and image_id == 0
                    else f"{split}-{tag}.png"
                )
                buffer = io.BytesIO()
                Image.new("RGB", (100, 100), (30 + index * 70, image_id * 100, 50)).save(
                    buffer, format="PNG"
                )
                bundle.writestr(split + "/" + name, buffer.getvalue())
                images.append(
                    {
                        "id": image_id,
                        "file_name": name,
                        "width": 100,
                        "height": 100,
                        "extra": {"user_tags": [tag], "name": name},
                    }
                )
                for category in categories:
                    annotations.append(
                        {
                            "id": len(annotations),
                            "image_id": image_id,
                            "category_id": category["id"],
                            "bbox": [10, 20, 20, 40],
                        }
                    )
            bundle.writestr(
                split + "/_annotations.coco.json",
                json.dumps(
                    {"categories": categories, "images": images, "annotations": annotations}
                ),
            )
        bundle.writestr("README.dataset.txt", "License: CC BY 4.0")


class ImportTests(unittest.TestCase):
    def test_corrections_leave_original_labels_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            archive = folder / "dataset.zip"
            make_archive(archive)
            spec, report = import_archive(archive, folder / "source")
            manifest = json.loads((spec.parent / "manifest.json").read_text())
            item = next(i for i in manifest if i["split"] == "train")
            original = (spec.parent / "labels/train" / (item["stem"] + ".txt")).read_text()
            corrections = folder / "corrections.json"
            corrections.write_text(
                json.dumps(
                    {
                        "source_archive_sha256": report["archive_sha256"],
                        "remove": [
                            {
                                "split": "train",
                                "stem": item["stem"],
                                "remove_line": item["labels"][0],
                                "reason": "fixture",
                            }
                        ],
                    }
                )
            )
            corrected = prepare(spec.parent, folder / "corrected", corrections)
            self.assertEqual(
                (spec.parent / "labels/train" / (item["stem"] + ".txt")).read_text(), original
            )
            lines = (
                (corrected.parent / "labels/train" / (item["stem"] + ".txt"))
                .read_text()
                .splitlines()
            )
            self.assertEqual(len(lines), 2)
            self.assertNotIn(item["labels"][0], lines)

    def test_source_splits_and_collector_names_are_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            archive = Path(folder) / "dataset.zip"
            make_archive(archive)
            specification, report = import_archive(archive, Path(folder) / "output")
            self.assertEqual(json.loads(specification.read_text())["names"], list(CLASSES))
            for split in ("train", "val", "test"):
                self.assertEqual(report["splits"][split]["images"], 1)
                self.assertEqual(report["splits"][split]["excluded_building_crops"], 1)
                label = (
                    next((specification.parent / "labels" / split).glob("*.txt"))
                    .read_text()
                    .splitlines()
                )
                self.assertEqual({int(line.split()[0]) for line in label}, {0, 1, 2})
                self.assertEqual(len(label), 3)  # Gold storages are not collectors.
            self.assertTrue((specification.parent / "README.dataset.txt").exists())
            with self.assertRaisesRegex(ValueError, "already exists"):
                import_archive(archive, specification.parent)

    def test_archive_paths_cannot_escape_output(self):
        with tempfile.TemporaryDirectory() as folder:
            archive = Path(folder) / "bad.zip"
            make_archive(archive, unsafe=True)
            with self.assertRaisesRegex(ValueError, "Unsafe image filename"):
                import_archive(archive, Path(folder) / "output")
            self.assertFalse((Path(folder) / "escape.jpg").exists())
            self.assertFalse((Path(folder) / "output").exists())

    def test_invalid_bounds_are_rejected_and_rounding_is_clipped(self):
        self.assertEqual(convert_box([0, 0, 100.2, 100], 100, 100), (0.5, 0.5, 1, 1))
        for box in ([0, 0, -1, 1], [0, 0, 200, 100], [float("nan"), 0, 1, 1]):
            with self.assertRaises(ValueError):
                convert_box(box, 100, 100)


if __name__ == "__main__":
    unittest.main()
