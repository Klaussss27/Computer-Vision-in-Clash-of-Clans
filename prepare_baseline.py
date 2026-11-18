"""Apply verified label corrections to a separate copy of an imported dataset."""

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path


def prepare(source, output, corrections):
    source, output = Path(source).resolve(), Path(output).resolve()
    changes = json.loads(Path(corrections).read_text())
    report = json.loads((source / "import-report.json").read_text())
    if changes["source_archive_sha256"] != report["archive_sha256"]:
        raise ValueError("Corrections refer to a different source archive")
    if output.exists() or source == output or source in output.parents or output in source.parents:
        raise ValueError("Choose a new output directory separate from the source")
    manifest = json.loads((source / "manifest.json").read_text())
    for change in changes["remove"]:
        match = next(
            (
                item
                for item in manifest
                if item["stem"] == change["stem"] and item["split"] == change["split"]
            ),
            None,
        )
        if match is None or match["labels"].count(change["remove_line"]) != 1:
            raise ValueError("Correction does not uniquely match the source labels")
        match["labels"].remove(change["remove_line"])
    shutil.copytree(source, output)
    for split, stats in report["splits"].items():
        counts = Counter()
        for item in manifest:
            if item["split"] != split:
                continue
            (output / "labels" / split / (item["stem"] + ".txt")).write_text(
                "\n".join(item["labels"]) + "\n"
            )
            for line in item["labels"]:
                counts[int(line.split()[0])] += 1
        stats["objects"] = {kind: counts[index] for index, kind in enumerate(report["classes"])}
    report["label_review"] = changes
    report.pop("manual_spot_check", None)
    specification = json.loads((source / "dataset.yaml").read_text())
    specification["path"] = output.as_posix()
    (output / "dataset.yaml").write_text(json.dumps(specification, indent=2))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (output / "import-report.json").write_text(json.dumps(report, indent=2))
    (output / "label-corrections.json").write_text(json.dumps(changes, indent=2))
    with (output / "ATTRIBUTION.md").open("a") as handle:
        handle.write(
            "\nAdditional changes: verified label removals recorded in label-corrections.json.\n"
        )
    return output / "dataset.yaml"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/imports/coc-v28-villages")
    parser.add_argument("--output", default="data/training/collector-baseline")
    parser.add_argument("--corrections", default="label_corrections.json")
    args = parser.parse_args()
    print(prepare(args.source, args.output, args.corrections))
