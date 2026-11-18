"""Add inspected video crops to training while preserving independent base splits."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from PIL import Image

from vision import COLLECTORS

ROIS = {
    "northwest": (470, 140, 260, 230),
    "northeast": (1250, 145, 230, 180),
    "east": (1600, 420, 240, 230),
    "southwest": (300, 595, 335, 285),
    "southeast": (1250, 685, 280, 200),
}


def crop_labels(objects, roi, after=False):
    x, y, w, h = roi
    labels = []
    for item in objects:
        bx, by, bw, bh = item["box"]
        intersects = bx < x + w and bx + bw > x and by < y + h and by + bh > y
        if not intersects:
            continue
        if not (x <= bx and y <= by and bx + bw <= x + w and by + bh <= y + h):
            raise ValueError("Crop cuts through an inspected collector")
        if after and item["destroyed_after"]:
            continue
        labels.append(
            f"{COLLECTORS.index(item['kind'])} {(bx - x + bw / 2) / w:.8f} {(by - y + bh / 2) / h:.8f} {bw / w:.8f} {bh / h:.8f}"
        )
    return labels


def prepare(baseline, frames, labels_path, output):
    baseline, frames, output = map(Path, (baseline, frames, output))
    labels = json.loads(Path(labels_path).read_text())
    manifest = json.loads((frames / "manifest.json").read_text())
    if manifest["sha256"] != labels["source_sha256"]:
        raise ValueError("Video labels do not match the source manifest")
    source = Path(manifest["source"])
    with source.open("rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != manifest["sha256"]:
            raise ValueError("Video source hash changed")
    if output.exists():
        raise ValueError("Choose a new output directory; previous training inputs are preserved")
    spec = json.loads((baseline / "dataset.yaml").read_text())
    if spec["names"] != list(COLLECTORS):
        raise ValueError("Class order differs from collector model")
    output.mkdir(parents=True)
    for split in ("train", "val", "test"):
        for kind in ("images", "labels"):
            shutil.copytree(baseline / kind / split, output / kind / split)
    pairs = labels["collector_pairs"]
    records = []
    for filename, after in [
        ("frame-009.png", False),
        ("frame-012.png", False),
        ("frame-042.png", True),
        ("frame-045.png", True),
    ]:
        path = frames / filename
        image = Image.open(path).convert("RGB")
        for name, roi in ROIS.items():
            x, y, w, h = roi
            lines = crop_labels(pairs["objects"], roi, after)
            key = "video-" + path.stem + "-" + name
            image.crop((x, y, x + w, y + h)).save(output / "images/train" / (key + ".png"))
            (output / "labels/train" / (key + ".txt")).write_text(
                "\n".join(lines) + ("\n" if lines else "")
            )
            records.append(
                {
                    "source_frame": filename,
                    "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "roi": list(roi),
                    "image": key + ".png",
                    "labels": len(lines),
                    "after_attack": after,
                    "group": "video-a2a79a9bac284eef",
                    "split": "train",
                }
            )
    spec["path"] = str(output.resolve()).replace("\\", "/")
    (output / "dataset.yaml").write_text(json.dumps(spec, indent=2))
    report = {
        "source_video_sha256": manifest["sha256"],
        "baseline": str(baseline.resolve()),
        "role": "All crops from the same video base are training only; validation and test unchanged",
        "review": "Five manually inspected regions covering ten intact collectors; eight become rubble and two drills survive",
        "video_crops": len(records),
        "video_boxes": sum(r["labels"] for r in records),
        "video_empty_crops": sum(r["labels"] == 0 for r in records),
        "frames": records,
        "splits": {
            s: len(list((output / "images" / s).iterdir())) for s in ("train", "val", "test")
        },
    }
    (output / "import-report.json").write_text(json.dumps(report, indent=2))
    for filename in ("ATTRIBUTION.md", "README.dataset.txt", "README.roboflow.txt"):
        if (baseline / filename).exists():
            shutil.copy2(baseline / filename, output / filename)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="data/exports/1f3a005c1374")
    parser.add_argument("--frames", default="data/video-review/2026-11-14-004714")
    parser.add_argument("--labels", default="video_expectations.json")
    parser.add_argument("--output", default="data/training/collectors-video-01")
    args = parser.parse_args()
    report = prepare(args.baseline, args.frames, args.labels, args.output)
    print(json.dumps({k: v for k, v in report.items() if k != "frames"}, indent=2))
