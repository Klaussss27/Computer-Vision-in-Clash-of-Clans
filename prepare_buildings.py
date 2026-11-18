"""Derive a generic building detector dataset from the licensed COCO import."""

import json
import shutil
import zipfile
from pathlib import Path


def main():
    source = Path("data/training/collector-baseline")
    target = Path("data/training/building-occupancy-01")
    if target.exists():
        raise ValueError("Output exists; preserve the previous export")
    manifest = json.loads((source / "manifest.json").read_text())
    report = {
        "source": "Clash of clans villages.v28i.coco-segmentation.zip",
        "class": "building",
        "splits": {},
        "note": "All specific non-wall building categories collapsed to one class. Public test split excluded.",
    }
    with zipfile.ZipFile(report["source"]) as archive:
        for split, member in [("train", "train"), ("val", "valid")]:
            coco = json.loads(archive.read(member + "/_annotations.coco.json"))
            images = {i["file_name"]: i for i in coco["images"]}
            annotations = {}
            for item in coco["annotations"]:
                if item["category_id"] > 0:
                    annotations.setdefault(item["image_id"], []).append(item["bbox"])
            for folder in ("images", "labels"):
                (target / folder / split).mkdir(parents=True)
            frames = boxes = 0
            for frame in manifest:
                if frame["split"] != split:
                    continue
                image = images[Path(frame["member"]).name]
                width, height = image["width"], image["height"]
                lines = set()
                for x, y, w, h in annotations.get(image["id"], []):
                    left, top = max(0, x), max(0, y)
                    right, bottom = min(width, x + w), min(height, y + h)
                    if right <= left or bottom <= top:
                        continue
                    lines.add(
                        "0 "
                        + " ".join(
                            f"{v:.8f}"
                            for v in (
                                (left + right) / 2 / width,
                                (top + bottom) / 2 / height,
                                (right - left) / width,
                                (bottom - top) / height,
                            )
                        )
                    )
                name = frame["stem"] + frame["extension"]
                shutil.copy2(source / "images" / split / name, target / "images" / split / name)
                (target / "labels" / split / (frame["stem"] + ".txt")).write_text(
                    "\n".join(sorted(lines)) + "\n"
                )
                frames += 1
                boxes += len(lines)
            report["splits"][split] = {"frames": frames, "boxes": boxes}
    for name in ("ATTRIBUTION.md", "README.dataset.txt", "README.roboflow.txt"):
        shutil.copy2(source / name, target / name)
    spec = {
        "path": target.resolve().as_posix(),
        "train": "images/train",
        "val": "images/val",
        "names": ["building"],
    }
    (target / "dataset.yaml").write_text(json.dumps(spec, indent=2))
    (target / "provenance.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
