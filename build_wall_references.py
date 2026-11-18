"""Build inspected wall texture references from training bases only."""

import hashlib
import json
from pathlib import Path

from PIL import Image

# Full-frame coordinates inspected against the supplied original images.
REFERENCES = [
    (1, (491, 390, 521, 422)),
    (1, (617, 297, 646, 330)),
    (1, (784, 384, 813, 416)),
    (1, (1300, 306, 1330, 336)),
    (8, (687, 174, 719, 209)),
    (8, (1005, 402, 1037, 436)),
    (2, (686, 170, 717, 203)),
    (2, (705, 211, 738, 246)),
    (2, (1085, 195, 1115, 225)),
    (9, (767, 160, 799, 194)),
    (9, (1238, 279, 1272, 314)),
    (9, (694, 228, 726, 260)),
    (9, (1020, 340, 1052, 373)),
]


def main():
    root = Path("data/scouting-bases")
    manifest = json.loads((root / "manifest.json").read_text())
    target = Path("templates/structures-v2/wall")
    target.mkdir(parents=True, exist_ok=True)
    provenance = []
    for index, (number, box) in enumerate(REFERENCES):
        frame = manifest["frames"][number - 1]
        if frame["split"] != "train":
            raise ValueError("Wall reference must come from the training split")
        path = root / "images" / (frame["id"] + ".png")
        with Image.open(path) as image:
            image.crop(box).save(target / f"wall-{index:02d}.png")
        provenance.append(
            {
                "sample": f"wall-{index:02d}.png",
                "base": frame["group"],
                "source_id": frame["id"],
                "box": box,
                "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    (target.parent / "provenance.json").write_text(
        json.dumps(
            {
                "references": provenance,
                "status": "experimental texture matching; not complete wall geometry",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
