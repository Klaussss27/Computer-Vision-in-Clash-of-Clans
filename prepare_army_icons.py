"""Extract army card artwork from a supplied training screenshot."""

import json
from pathlib import Path

from PIL import Image


def main():
    root = Path("data/scouting-bases")
    frame = json.loads((root / "manifest.json").read_text())["frames"][0]
    if frame["split"] != "train":
        raise ValueError("Army reference must come from training data")
    image = Image.open(root / "images" / (frame["id"] + ".png"))
    target = Path("templates/army")
    target.mkdir(exist_ok=True)
    boxes = {"sneaky_goblins": [298, 970, 377, 1020], "jump_spell": [703, 970, 766, 1025]}
    for kind, box in boxes.items():
        image.crop(box).save(target / (kind + ".png"))
    (target / "provenance.json").write_text(
        json.dumps(
            {
                "source_id": frame["id"],
                "group": frame["group"],
                "source_pixel_sha256": frame["pixel_sha256"],
                "boxes": boxes,
                "note": "Saved-file calibration; live selected/empty cards still need testing.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
