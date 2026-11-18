"""Local, reviewed box annotations and base-separated YOLO dataset exports."""

import hashlib
import json
import random
import shutil
import uuid
from pathlib import Path

from PIL import Image

from vision import KINDS


class Dataset:
    def __init__(self, directory="data/collectors"):
        self.root = Path(directory)
        (self.root / "images").mkdir(parents=True, exist_ok=True)
        (self.root / "annotations").mkdir(exist_ok=True)

    def frame(self, image):
        rgb = image.convert("RGB")
        key = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()[:24]
        path = self.root / "annotations" / (key + ".json")
        if path.exists():
            return json.loads(path.read_text())
        rgb.save(self.root / "images" / (key + ".png"))
        record = {"id": key, "size": list(rgb.size), "group": "", "reviewed": False, "objects": []}
        self.save(record)
        return record

    def save(self, record):
        # IDs are generated here, never filesystem paths supplied by an annotation.
        key = record["id"]
        if len(key) != 24 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("Invalid frame ID")
        for item in record["objects"]:
            yolo_line(item, record["size"])
        path = self.root / "annotations" / (key + ".json")
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(path)

    def records(self):
        return [
            json.loads(path.read_text())
            for path in sorted((self.root / "annotations").glob("*.json"))
        ]


def yolo_line(item, size, classes=KINDS):
    width, height = size
    x1, y1, x2, y2 = item["box"]
    if item["kind"] not in classes or not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("Invalid annotation class or bounding box")
    values = ((x1 + x2) / 2 / width, (y1 + y2) / 2 / height, (x2 - x1) / width, (y2 - y1) / height)
    return str(classes.index(item["kind"])) + " " + " ".join(f"{v:.8f}" for v in values)


def prepare_dataset(source="data/collectors", output="data/exports", seed=42, classes=KINDS):
    classes = tuple(classes)
    if not classes or len(set(classes)) != len(classes) or not set(classes) <= set(KINDS):
        raise ValueError("Choose distinct supported dataset classes")
    dataset = Dataset(source)
    records = [
        r
        for r in dataset.records()
        if r.get("reviewed") and set(classes) <= set(r.get("reviewed_classes", KINDS))
    ]
    groups = sorted({r["group"].strip() for r in records})
    if len(groups) < 2 or "" in groups:
        raise ValueError(
            "Review frames from at least two distinct bases for the requested class scope, and give each base a group ID."
        )
    reserved = any("split" in r for r in dataset.records())
    if reserved:
        all_records = dataset.records()
        if any(
            not r.get("reviewed")
            or not r.get("grouping_reviewed")
            or not set(classes) <= set(r.get("reviewed_classes", []))
            for r in all_records
        ):
            raise ValueError(
                "Reserved dataset requires all boxes and base groups reviewed before export."
            )
        group_roles = {}
        for record in all_records:
            role = record.get("split")
            if role not in ("train", "val", "test"):
                raise ValueError("Every reserved frame needs a train, val, or test split.")
            group = record["group"].strip()
            if group in group_roles and group_roles[group] != role:
                raise ValueError("The same base cannot appear in different reserved splits.")
            group_roles[group] = role
        splits = {"train": [], "val": [], "test": []}
    else:
        random.Random(seed).shuffle(groups)
        validation = set(groups[: max(1, round(len(groups) * 0.2))])
        splits = {"train": [], "val": []}
    counts = {split: {kind: 0 for kind in classes} for split in splits}
    for record in records:
        split = (
            record["split"]
            if reserved
            else ("val" if record["group"].strip() in validation else "train")
        )
        for item in record["objects"]:
            yolo_line(item, record["size"])
            if item["kind"] in classes:
                counts[split][item["kind"]] += 1
        splits[split].append(record)
    missing = [kind for kind, count in counts["train"].items() if count == 0]
    if missing:
        raise ValueError(
            "Training split has no examples of: "
            + ", ".join(missing)
            + ". Label more bases before training."
        )
    target = Path(output).resolve() / uuid.uuid4().hex[:12]
    for split, frames in splits.items():
        (target / "images" / split).mkdir(parents=True)
        (target / "labels" / split).mkdir(parents=True)
        for record in frames:
            key = record["id"]
            image_path = dataset.root / "images" / (key + ".png")
            with Image.open(image_path) as image:
                if list(image.size) != record["size"]:
                    raise ValueError("Annotation size does not match image: " + key)
            shutil.copy2(image_path, target / "images" / split / image_path.name)
            lines = [
                yolo_line(item, record["size"], classes)
                for item in record["objects"]
                if item["kind"] in classes
            ]
            (target / "labels" / split / (key + ".txt")).write_text("\n".join(lines) + "\n")
    # JSON is also valid YAML, without adding a YAML dependency to the labeler.
    specification = {
        "path": target.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "names": list(classes),
    }
    if reserved:
        specification["test"] = "images/test"
    (target / "dataset.yaml").write_text(json.dumps(specification, indent=2))
    report = {
        "frames": {split: len(frames) for split, frames in splits.items()},
        "objects": counts,
        "groups": {split: sorted({r["group"] for r in frames}) for split, frames in splits.items()},
        "missing_validation_classes": [kind for kind, count in counts["val"].items() if count == 0],
    }
    (target / "split.json").write_text(json.dumps(report, indent=2))
    return target / "dataset.yaml", report
