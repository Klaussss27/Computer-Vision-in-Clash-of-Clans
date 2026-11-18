"""Extract a timestamped storyboard and closing frames from a supplied video."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.video)
    output = Path(args.output)
    with source.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if (output / "manifest.json").exists() and json.loads((output / "manifest.json").read_text())[
        "sha256"
    ] != digest:
        raise ValueError("Output contains frames from another video")
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError("Cannot open video")
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if not math.isfinite(fps) or not math.isfinite(count) or fps <= 0 or count <= 0:
        cap.release()
        raise ValueError("Invalid video metadata")
    duration = count / fps
    output.mkdir(parents=True, exist_ok=True)
    times = [
        (float(t), f"frame-{t:03d}.png") for t in range(0, math.ceil(duration), 3) if t < duration
    ]
    times += [(t / 2, f"early-{t / 2:03.1f}.png") for t in range(1, 6) if t / 2 < duration]
    ending = max(0, math.ceil(duration - 10))
    times += [
        (t / 2, f"fine-{t / 2:04.1f}.png") for t in range(ending * 2, math.floor(duration * 2))
    ]
    frames = []
    try:
        for second, name in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
            ok, bgr = cap.read()
            if not ok:
                continue
            if not (output / name).exists():
                Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).save(output / name)
            frames.append({"seconds": second, "image": name})
    finally:
        cap.release()
    coarse = [f for f in frames if f["image"].startswith("frame-")]
    sheet = Image.new("RGB", (1280, math.ceil(len(coarse) / 3) * 204), "#222222")
    draw = ImageDraw.Draw(sheet)
    for index, frame in enumerate(coarse):
        with Image.open(output / frame["image"]) as im:
            im.thumbnail((426, 180))
            x = index % 3 * 426
            y = index // 3 * 204
            sheet.paste(im, (x, y + 23))
            draw.text((x + 5, y + 5), f"{frame['seconds']:g}s", fill="white")
    sheet.save(output / "storyboard.jpg")
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "source": str(source.resolve()),
                "sha256": digest,
                "fps": fps,
                "duration_seconds": duration,
                "frames": frames,
                "role": "development/calibration only",
            },
            indent=2,
        )
    )
    print(f"{len(frames)} frames extracted to {output.resolve()}")


if __name__ == "__main__":
    main()
