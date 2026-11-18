"""Add masked training assets to training villages without changing held-out splits."""

import argparse
import hashlib
import io
import json
import random
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

from import_coco import CLASS_MAP, CLASSES


def boxes_overlap(a, b, margin=3):
    return (
        a[0] < b[2] + margin
        and a[2] + margin > b[0]
        and a[1] < b[3] + margin
        and a[3] + margin > b[1]
    )


def label_box(line, size):
    _, cx, cy, width, height = map(float, line.split())
    w, h = size
    return (
        (cx - width / 2) * w,
        (cy - height / 2) * h,
        (cx + width / 2) * w,
        (cy + height / 2) * h,
    )


def load_assets(archive, include_towers=True):
    """Only source-training assets are eligible, including hard-negative storages."""
    assets = defaultdict(list)
    with zipfile.ZipFile(archive) as bundle:
        document = json.loads(bundle.read("train/_annotations.coco.json"))
        classes = {item["id"]: item["name"] for item in document["categories"]}
        annotations = defaultdict(list)
        for item in document["annotations"]:
            annotations[item["image_id"]].append(item)
        for item in document["images"]:
            if "building" not in item.get("extra", {}).get("user_tags", []):
                continue
            selected = annotations[item["id"]]
            if len(selected) != 1:
                continue
            annotation = selected[0]
            source_class = classes[annotation["category_id"]]
            kind = CLASS_MAP.get(source_class)
            negative_classes = (
                ("gold_storage", "elexir_storage", "archer_tower", "wizard_tower")
                if include_towers
                else ("gold_storage", "elexir_storage")
            )
            if kind is None and source_class not in negative_classes:
                continue
            polygons = annotation.get("segmentation")
            if not isinstance(polygons, list) or not polygons:
                continue
            with Image.open(io.BytesIO(bundle.read("train/" + item["file_name"]))) as source:
                source = source.convert("RGB")
            x, y, w, h = annotation["bbox"]
            box = (
                max(0, int(x)),
                max(0, int(y)),
                min(source.width, int(x + w + 1)),
                min(source.height, int(y + h + 1)),
            )
            if box[2] - box[0] < 8 or box[3] - box[1] < 8:
                continue
            mask = Image.new("L", source.size, 0)
            draw = ImageDraw.Draw(mask)
            for polygon in polygons:
                if len(polygon) >= 6:
                    draw.polygon(list(zip(polygon[::2], polygon[1::2])), fill=255)
            mask = mask.crop(box)
            if mask.getbbox() is None:
                continue
            assets[kind or "negative"].append(
                {
                    "image": source.crop(box),
                    "mask": mask,
                    "source": item.get("extra", {}).get("name", item["file_name"]),
                }
            )
    if any(not assets[kind] for kind in CLASSES):
        raise ValueError("Training archive lacks masked examples of a collector class")
    return assets


def build(source, archive, output, count=240, seed=42, recipe="balanced"):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() or source in output.parents or output in source.parents:
        raise ValueError("Use a new output directory separate from the source")
    if not 1 <= count <= 2000:
        raise ValueError("Synthetic image count must be from 1 to 2000")
    if recipe not in ("original", "balanced"):
        raise ValueError("Recipe must be original or balanced")
    rng = random.Random(seed)
    manifest = json.loads((source / "manifest.json").read_text())
    backgrounds = [item for item in manifest if item["split"] == "train"]
    assets = load_assets(archive, include_towers=recipe == "balanced")
    shutil.copytree(source, output)
    added, counts = [], Counter()
    for index in range(count):
        background = rng.choice(backgrounds)
        with Image.open(
            source / "images/train" / (background["stem"] + background["extension"])
        ) as image:
            image = image.convert("RGB")
        lines = list(background["labels"])
        occupied = [label_box(line, image.size) for line in lines]
        inserted = []
        # Add more drills to offset their lower count in the original villages.
        positives = (
            [rng.choice(CLASSES) for _ in range(4)]
            if recipe == "original"
            else rng.choices(CLASSES, weights=(1, 1, 2), k=4)
        )
        for kind in positives + ["negative"] * 2:
            if not assets[kind]:
                continue
            asset = rng.choice(assets[kind])
            maximum = rng.randint(24, 96)
            scale = maximum / max(asset["image"].size)
            w, h = (
                max(8, round(asset["image"].width * scale)),
                max(8, round(asset["image"].height * scale)),
            )
            if w + 8 >= image.width or h + 8 >= image.height:
                continue
            for _ in range(60):
                x, y = rng.randint(4, image.width - w - 4), rng.randint(4, image.height - h - 4)
                box = (x, y, x + w, y + h)
                if not any(boxes_overlap(box, previous) for previous in occupied):
                    break
            else:
                continue
            tile = asset["image"].resize((w, h), Image.Resampling.LANCZOS)
            mask = asset["mask"].resize((w, h), Image.Resampling.LANCZOS)
            image.paste(tile, (x, y), mask)
            occupied.append(box)
            if kind != "negative":
                values = (
                    (x + w / 2) / image.width,
                    (y + h / 2) / image.height,
                    w / image.width,
                    h / image.height,
                )
                lines.append(str(CLASSES.index(kind)) + " " + " ".join(f"{v:.8f}" for v in values))
                counts[kind] += 1
            inserted.append({"kind": kind, "source_asset": asset["source"], "box": list(box)})
        name = f"synthetic-{seed}-{index:04d}"
        image.save(output / "images/train" / (name + ".jpg"), quality=94)
        (output / "labels/train" / (name + ".txt")).write_text("\n".join(lines) + "\n")
        added.append({"image": name, "background": background["stem"], "inserted": inserted})
    spec = json.loads((source / "dataset.yaml").read_text())
    spec["path"] = output.as_posix()
    (output / "dataset.yaml").write_text(json.dumps(spec, indent=2))
    report = {
        "seed": seed,
        "recipe": recipe,
        "synthetic_images": count,
        "training_backgrounds": len(backgrounds),
        "assets_per_class": {kind: len(items) for kind, items in assets.items()},
        "added_collector_boxes": dict(counts),
        "source_asset_split": "train",
        "validation_and_test_changed": False,
        "limitations": [
            "Synthetic placements can be unrealistic; actual-game validation is still required.",
            "No new game captures or saved player-session captures are used.",
        ],
    }
    # Exact hashes make preservation of held-out labels and images inspectable.
    report["held_out_hashes"] = {
        str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
        for split in ("val", "test")
        for kind in ("images", "labels")
        for path in (source / kind / split).iterdir()
        if path.is_file()
    }
    for relative, digest in report["held_out_hashes"].items():
        if hashlib.sha256((output / relative).read_bytes()).hexdigest() != digest:
            raise RuntimeError("Held-out data changed during augmentation")
    (output / "augmentation-report.json").write_text(json.dumps(report, indent=2))
    metadata_path = output / "import-report.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        metadata["augmentation"] = {
            key: value for key, value in report.items() if key != "held_out_hashes"
        }
        metadata["effective_training_images"] = len(backgrounds) + count
        metadata_path.write_text(json.dumps(metadata, indent=2))
    (output / "synthetic-manifest.json").write_text(json.dumps(added, indent=2))
    with (output / "ATTRIBUTION.md").open("a") as handle:
        handle.write(
            "\nAdditional changes: masked source-training building assets pasted into training villages; "
            "sizes/positions randomized. Source validation and test files unchanged.\n"
        )
    return output / "dataset.yaml", report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/training/collector-baseline")
    parser.add_argument("--archive", default="Clash of clans villages.v28i.coco-segmentation.zip")
    parser.add_argument("--output", default="data/training/collector-assets-02")
    parser.add_argument("--count", type=int, default=240)
    parser.add_argument("--recipe", choices=["original", "balanced"], default="balanced")
    args = parser.parse_args()
    spec, report = build(args.source, args.archive, args.output, args.count, recipe=args.recipe)
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "held_out_hashes"}, indent=2
        )
    )
    print("Dataset:", spec)
