"""Freeze reviewed scouting frames, retaining reserved splits and source labels."""

import argparse
import json
import shutil
from pathlib import Path

from dataset import Dataset, prepare_dataset
from vision import COLLECTORS


def prepare(source, snapshot, exports, public=None):
    source = Dataset(source)
    snapshot = Path(snapshot)
    if snapshot.exists():
        raise ValueError("Choose a new snapshot directory to preserve previous labels.")
    records = source.records()
    reviewed = [
        r
        for r in records
        if r.get("reviewed")
        and r.get("grouping_reviewed")
        and set(COLLECTORS) <= set(r.get("reviewed_classes", []))
    ]
    if not reviewed:
        raise ValueError("No reviewed collector frames available.")
    frozen = Dataset(snapshot)
    for record in reviewed:
        filename = record["id"] + ".png"
        shutil.copy2(source.root / "images" / filename, frozen.root / "images" / filename)
        frozen.save(record)
    spec, report = prepare_dataset(snapshot, exports, classes=COLLECTORS)
    added = 0
    if public:
        public = Path(public)
        info = json.loads((public / "dataset.yaml").read_text())
        names = info["names"]
        if isinstance(names, dict):
            names = [names[str(i)] if str(i) in names else names[i] for i in range(len(names))]
        if list(names) != list(COLLECTORS):
            raise ValueError("Public training classes must have the same order.")
        for image in (public / "images" / "train").iterdir():
            if image.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            label = public / "labels" / "train" / (image.stem + ".txt")
            if not label.exists():
                raise ValueError("Missing public training label: " + str(label))
            shutil.copy2(image, spec.parent / "images" / "train" / ("public-" + image.name))
            shutil.copy2(label, spec.parent / "labels" / "train" / ("public-" + label.name))
            added += 1
        for name in ("ATTRIBUTION.md", "README.dataset.txt", "README.roboflow.txt"):
            if (public / name).exists():
                shutil.copy2(public / name, spec.parent / name)
    provenance = {
        "reviewed_source": str(source.root),
        "snapshot": str(snapshot),
        "scouting_splits": report,
        "public_training_frames": added,
        "excluded_frames": len(records) - len(reviewed),
        "public_validation_or_test_added": False,
    }
    (spec.parent / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return spec, provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/scouting-bases")
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--exports", default="data/exports")
    parser.add_argument("--public", default="data/training/collector-baseline")
    args = parser.parse_args()
    spec, report = prepare(args.source, args.snapshot, args.exports, args.public)
    print(json.dumps(report, indent=2))
    print("Dataset:", spec.resolve())
