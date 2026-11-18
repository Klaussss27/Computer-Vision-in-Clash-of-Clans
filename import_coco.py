"""Convert a Roboflow COCO ZIP into a collector-only YOLO dataset."""

import argparse
import hashlib
import io
import json
import math
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

from PIL import Image

CLASS_MAP = {
    "gold_mine": "gold_mine",
    "elexir_collector": "elixir_collector",
    "elixir_collector": "elixir_collector",
    "dark_elexir_drill": "dark_drill",
    "dark_elixir_drill": "dark_drill",
}
CLASSES = ("gold_mine", "elixir_collector", "dark_drill")


def convert_box(box, width, height):
    x, y, w, h = box
    if not all(math.isfinite(v) for v in box) or w <= 0 or h <= 0:
        raise ValueError(f"Invalid COCO bounding box: {box}")
    # Fractional segmentation bounds can extend a pixel past a resized image.
    if x < -1 or y < -1 or x + w > width + 1 or y + h > height + 1:
        raise ValueError(f"COCO bounding box lies outside its image: {box}")
    x1, y1, x2, y2 = max(0, x), max(0, y), min(width, x + w), min(height, y + h)
    if x1 >= x2 or y1 >= y2:
        raise ValueError("Empty COCO bounding box")
    return ((x1 + x2) / 2 / width, (y1 + y2) / 2 / height, (x2 - x1) / width, (y2 - y1) / height)


def import_archive(archive, output, include_crops=False):
    archive, output = Path(archive), Path(output).resolve()
    if output.exists():
        raise ValueError(f"Output already exists; choose a new --output directory: {output}")
    report = {
        "source_archive": archive.name,
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "source": "https://universe.roboflow.com/cocsegmentation/clash-of-clans-villages/dataset/28",
        "classes": list(CLASSES),
        "include_building_crops": include_crops,
        "splits": {},
        "discarded_duplicates": [],
        "missing_classes": ["wall", "rubble"],
        "limitations": [
            "Source tags and supplied labels have not been manually verified.",
            "Duplicate checks cover filenames and exact image pixels, not visually similar layouts.",
        ],
    }
    # Check held-out splits first so an exact duplicate never enters training.
    seen_pixels, seen_sources = set(), set()
    staged = []
    with zipfile.ZipFile(archive) as bundle:
        if sum(item.file_size for item in bundle.infolist()) > 2_000_000_000:
            raise ValueError("Archive exceeds the 2 GB uncompressed import limit")
        members = set(bundle.namelist())
        for source_split, split in (("test", "test"), ("valid", "val"), ("train", "train")):
            annotation_path = source_split + "/_annotations.coco.json"
            document = json.loads(bundle.read(annotation_path))
            categories = {item["id"]: item["name"] for item in document["categories"]}
            labels = defaultdict(list)
            for item in document["annotations"]:
                labels[item["image_id"]].append(item)
            counts = Counter()
            stats = {
                "source_images": len(document["images"]),
                "images": 0,
                "excluded_building_crops": 0,
                "duplicate_images": 0,
                "negative_images": 0,
                "objects": {kind: 0 for kind in CLASSES},
            }
            for item in document["images"]:
                tags = item.get("extra", {}).get("user_tags", [])
                if not include_crops and "village" not in tags:
                    stats["excluded_building_crops"] += 1
                    continue
                name = item["file_name"]
                path = PurePosixPath(name)
                if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
                    raise ValueError("Unsafe image filename in archive")
                member = source_split + "/" + name
                if member not in members:
                    raise ValueError("Image missing from archive: " + member)
                data = bundle.read(member)
                with Image.open(io.BytesIO(data)) as image:
                    image.load()
                    if image.size != (item["width"], item["height"]):
                        raise ValueError("Image dimensions differ from COCO metadata: " + member)
                    pixel_hash = hashlib.sha256(
                        str(image.size).encode() + image.convert("RGB").tobytes()
                    ).hexdigest()
                    extension = (
                        ".jpg"
                        if image.format == "JPEG"
                        else ".png"
                        if image.format == "PNG"
                        else None
                    )
                    if extension is None:
                        raise ValueError("Unsupported image format: " + member)
                original = item.get("extra", {}).get("name") or name.split(".rf.")[0]
                if pixel_hash in seen_pixels or original in seen_sources:
                    stats["duplicate_images"] += 1
                    report["discarded_duplicates"].append({"split": split, "source": name})
                    continue
                seen_pixels.add(pixel_hash)
                seen_sources.add(original)
                lines = []
                for annotation in labels[item["id"]]:
                    category = categories[annotation["category_id"]]
                    kind = CLASS_MAP.get(category)
                    if kind is None:
                        continue
                    if annotation.get("iscrowd", 0):
                        raise ValueError("Crowd annotation needs review: " + member)
                    values = convert_box(annotation["bbox"], item["width"], item["height"])
                    lines.append(
                        str(CLASSES.index(kind)) + " " + " ".join(f"{v:.8f}" for v in values)
                    )
                    counts[kind] += 1
                stats["images"] += 1
                stats["negative_images"] += not lines
                staged.append(
                    {
                        "split": split,
                        "member": member,
                        "stem": pixel_hash[:24],
                        "extension": extension,
                        "labels": lines,
                        "source_name": original,
                        "size": [item["width"], item["height"]],
                    }
                )
            stats["objects"] = {kind: counts[kind] for kind in CLASSES}
            report["splits"][split] = stats
        for split in ("train", "val", "test"):
            if not report["splits"][split]["images"]:
                raise ValueError(f"No images available for {split}; check source tags and filters")
            if any(count == 0 for count in report["splits"][split]["objects"].values()):
                raise ValueError(f"{split} has no examples of one or more collector classes")
        output.mkdir(parents=True)
        for item in staged:
            split = item["split"]
            images = output / "images" / split
            labels = output / "labels" / split
            images.mkdir(parents=True, exist_ok=True)
            labels.mkdir(parents=True, exist_ok=True)
            (images / (item["stem"] + item["extension"])).write_bytes(bundle.read(item["member"]))
            (labels / (item["stem"] + ".txt")).write_text("\n".join(item["labels"]) + "\n")
        for name in ("README.dataset.txt", "README.roboflow.txt"):
            if name in members:
                (output / name).write_bytes(bundle.read(name))
    specification = {
        "path": output.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": list(CLASSES),
    }
    (output / "dataset.yaml").write_text(json.dumps(specification, indent=2))
    (output / "manifest.json").write_text(json.dumps(staged, indent=2))
    (output / "import-report.json").write_text(json.dumps(report, indent=2))
    (output / "ATTRIBUTION.md").write_text(
        "# Source dataset\n\nClash of clans villages, by Cocsegmentation, version 28.\n"
        "https://universe.roboflow.com/cocsegmentation/clash-of-clans-villages/dataset/28\n\n"
        "License stated in source export: CC BY 4.0.\nhttps://creativecommons.org/licenses/by/4.0/\n\n"
        "Changes: filtered to village frames unless requested otherwise, retained collector classes only, "
        "renamed classes, converted COCO boxes to YOLO format, and removed exact duplicate images.\n"
        "Original export notices are preserved alongside this file.\n"
    )
    return output / "dataset.yaml", report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive")
    parser.add_argument("--output", default="data/imports/coc-v28-villages")
    parser.add_argument("--include-building-crops", action="store_true")
    args = parser.parse_args()
    specification, report = import_archive(args.archive, args.output, args.include_building_crops)
    print(json.dumps(report, indent=2))
    print("Training configuration:", specification)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, zipfile.BadZipFile) as error:
        raise SystemExit(str(error))
