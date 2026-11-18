"""Import user-supplied scouting images and propose boxes for offline review."""

import argparse
import hashlib
import json
import random
import zipfile
from collections import Counter
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from dataset import Dataset
from neural_detector import NeuralDetector
from vision import COLLECTORS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive")
    parser.add_argument("--output", default="data/scouting-bases")
    parser.add_argument("--checkpoint", default="models/collectors-assets-03.pt")
    args = parser.parse_args()
    root = Path(args.output)
    if root.exists():
        parser.error("Output already exists; preserve the existing review work.")
    frames = []
    seen = set()
    with zipfile.ZipFile(args.archive) as archive:
        for entry in sorted(archive.infolist(), key=lambda e: e.filename):
            if entry.is_dir() or Path(entry.filename).suffix.lower() not in {
                ".png",
                ".jpg",
                ".jpeg",
                ".webp",
            }:
                continue
            if entry.file_size > 40_000_000:
                raise ValueError("Image exceeds import size limit")
            with Image.open(BytesIO(archive.read(entry))) as source:
                source.load()
                frame = ImageOps.exif_transpose(source).convert("RGB")
            if frame.size != (1920, 1080):
                raise ValueError(f"Expected 1920x1080: {entry.filename} has {frame.size}")
            digest = hashlib.sha256(str(frame.size).encode() + frame.tobytes()).hexdigest()
            if digest not in seen:
                frames.append((entry.filename, digest, frame))
                seen.add(digest)
    if len(frames) < 10:
        raise ValueError("At least ten distinct scouting frames required")
    # Freeze roles before model proposals or manual review. Similar bases must be
    # grouped during review before an export is allowed to use this split.
    indices = list(range(len(frames)))
    random.Random(42).shuffle(indices)
    held = max(1, round(len(frames) / 6))
    roles = {
        i: ("test" if rank < held else "val" if rank < held * 2 else "train")
        for rank, i in enumerate(indices)
    }
    settings = {
        "model_path": args.checkpoint,
        "model_device": "0",
        "model_confidence": 0.25,
        "model_image_size": 1280,
        "model_edge_margin": 3,
        "playfield": [0, 0, 1920, 905],
    }
    detector = NeuralDetector(settings, required_classes=COLLECTORS)
    dataset = Dataset(root)
    previews = root / "review"
    previews.mkdir()
    manifest = {
        "archive": Path(args.archive).name,
        "archive_sha256": hashlib.sha256(Path(args.archive).read_bytes()).hexdigest(),
        "checkpoint": args.checkpoint,
        "seed": 42,
        "status": "unreviewed_proposals",
        "frames": [],
        "note": "No training export until boxes and base grouping are reviewed. Predictions are not ground truth. Test role is reserved; do not tune against it.",
    }
    colors = {"gold_mine": "yellow", "elixir_collector": "magenta", "dark_drill": "cyan"}
    for index, (name, digest, frame) in enumerate(frames):
        record = dataset.frame(frame)
        record.update(
            group=f"scout-{index + 1:02d}",
            split=roles[index],
            source_name=name,
            reviewed=False,
            reviewed_classes=[],
            grouping_reviewed=False,
            proposal_checkpoint=args.checkpoint,
        )
        detections = detector.detect(frame)
        record["objects"] = [
            {
                "kind": d.kind,
                "box": [d.box[0], d.box[1], d.box[0] + d.box[2], d.box[1] + d.box[3]],
                "confidence": d.score,
            }
            for d in detections
        ]
        dataset.save(record)
        overlay = frame.copy()
        draw = ImageDraw.Draw(overlay)
        for number, item in enumerate(record["objects"]):
            box = item["box"]
            draw.rectangle(box, outline=colors[item["kind"]], width=3)
            draw.text(
                (box[0], max(0, box[1] - 13)),
                f"{number}: {item['confidence']:.2f}",
                fill=colors[item["kind"]],
                stroke_width=1,
                stroke_fill="black",
            )
        overlay.save(previews / f"{index + 1:02d}.jpg", quality=92)
        manifest["frames"].append(
            {
                "id": record["id"],
                "source": name,
                "pixel_sha256": digest,
                "group": record["group"],
                "split": roles[index],
                "proposed_counts": dict(Counter(d.kind for d in detections)),
            }
        )
        print(f"{index + 1}/{len(frames)} {roles[index]}: {len(detections)} proposals", flush=True)
    manifest["split_counts"] = dict(Counter(roles.values()))
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["split_counts"]))


if __name__ == "__main__":
    main()
