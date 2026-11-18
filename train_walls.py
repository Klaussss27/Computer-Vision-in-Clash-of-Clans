"""Train a sparse-supervision wall research model on saved screenshots."""

import argparse
import hashlib
import json
import random
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from dataset import Dataset
from wall_model import WallNet


def load_frames(public_negatives=False):
    root = Path("data/scouting-bases")
    annotations = json.loads(Path("wall_annotations.json").read_text())
    frames = {}
    references = json.loads(Path("templates/structures-v2/provenance.json").read_text())[
        "references"
    ]
    for record in Dataset(root).records():
        group = record["group"]
        if group not in annotations["training_strokes"]:
            continue
        if record["split"] != "train":
            raise ValueError("Wall training strokes must use training bases only")
        rgb = np.asarray(Image.open(root / "images" / (record["id"] + ".png")).convert("RGB"))
        labels = np.full(rgb.shape[:2], -1, np.int8)
        # Only interior points of reviewed collectors are negative examples.
        # All other unmarked pixels remain unknown, not background.
        for item in record["objects"]:
            if item["kind"] not in ("gold_mine", "elixir_collector", "dark_drill"):
                continue
            x1, y1, x2, y2 = item["box"]
            cv2.circle(labels, ((x1 + x2) // 2, (y1 + y2) // 2), 7, 0, -1)
        for x, y in [
            (100, 400),
            (1800, 400),
            (100, 650),
            (1800, 650),
            (300, 100),
            (1450, 50),
            (100, 750),
            (150, 730),
            (1800, 810),
            (110, 850),
            (1640, 45),
            (1700, 110),
            (1700, 170),
            (960, 50),
            (100, 40),
        ]:
            cv2.circle(labels, (x, y), 8, 0, -1)
        for stroke in annotations["training_strokes"][group]:
            cv2.polylines(labels, [np.asarray(stroke, np.int32)], False, 1, 3)
        for reference in references:
            if reference["base"] == group:
                x1, y1, x2, y2 = reference["box"]
                cv2.circle(labels, ((x1 + x2) // 2, (y1 + y2) // 2), 3, 1, -1)
        frames[group] = (rgb, labels)
    if public_negatives:
        public = Path("data/training/collector-baseline")
        with zipfile.ZipFile("Clash of clans villages.v28i.coco-segmentation.zip") as archive:
            coco = json.loads(archive.read("train/_annotations.coco.json"))
        by_name = {i["file_name"]: i["id"] for i in coco["images"]}
        boxes = {}
        for item in coco["annotations"]:
            boxes.setdefault(item["image_id"], []).append(item["bbox"])
        for entry in json.loads((public / "manifest.json").read_text()):
            if entry["split"] != "train":
                continue
            image_path = public / "images" / "train" / (entry["stem"] + entry["extension"])
            rgb = np.asarray(Image.open(image_path).convert("RGB"))
            labels = np.full(rgb.shape[:2], -1, np.int8)
            for x, y, w, h in boxes.get(by_name[Path(entry["member"]).name], []):
                radius = max(2, min(9, int(min(w, h) / 8)))
                cx, cy = round(x + w / 2), round(y + h / 2)
                if 0 <= cx < rgb.shape[1] and 0 <= cy < rgb.shape[0]:
                    cv2.circle(labels, (cx, cy), radius, 0, -1)
            frames["public-" + entry["stem"]] = (rgb, labels)
    return frames


def batch(frames, rng, count=24, size=96, locations=None):
    images, masks = [], []
    if locations is None:
        locations = {
            g: {k: np.argwhere(mask == k) for k in (0, 1)} for g, (_, mask) in frames.items()
        }
    for index in range(count):
        group = rng.choice([g for g in frames if len(locations[g][index % 2])])
        rgb, labels = frames[group]
        py, px = locations[group][index % 2][rng.randrange(len(locations[group][index % 2]))]
        x = max(0, min(rgb.shape[1] - size, int(px) - size // 2 + rng.randrange(-20, 21)))
        y = max(0, min(rgb.shape[0] - size, int(py) - size // 2 + rng.randrange(-20, 21)))
        image = rgb[y : y + size, x : x + size].copy()
        mask = labels[y : y + size, x : x + size].copy()
        if rng.random() < 0.5:
            image, mask = image[:, ::-1].copy(), mask[:, ::-1].copy()
        image = np.clip(image.astype(np.float32) * rng.uniform(0.8, 1.2), 0, 255) / 255
        images.append(image.transpose(2, 0, 1))
        masks.append(mask)
    return torch.from_numpy(np.stack(images)), torch.from_numpy(np.stack(masks)).float()


def sparse_loss(prediction, mask):
    losses = [
        F.binary_cross_entropy_with_logits(prediction[mask == k], mask[mask == k])
        for k in (0, 1)
        if (mask == k).any()
    ]
    if not losses:
        raise ValueError("No reviewed wall/background pixels in this batch")
    return torch.stack(losses).mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--output", default="runs/walls/sparse-01")
    parser.add_argument("--public-negatives", action="store_true")
    args = parser.parse_args()
    target = Path(args.output)
    if target.exists():
        parser.error("Choose a new run directory")
    target.mkdir(parents=True)
    rng = random.Random(42)
    torch.manual_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    annotation_snapshot = json.loads(Path("wall_annotations.json").read_text())
    (target / "annotation-snapshot.json").write_text(json.dumps(annotation_snapshot, indent=2))
    frames = load_frames(args.public_negatives)
    locations = {g: {k: np.argwhere(mask == k) for k in (0, 1)} for g, (_, mask) in frames.items()}
    model = WallNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    started = time.time()
    history = []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for _ in range(args.steps):
            image, mask = batch(frames, rng, locations=locations)
            image, mask = image.to(device), mask.to(device)
            pred = model(image)[:, 0]
            loss = sparse_loss(pred, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        loss = sum(losses) / len(losses)
        history.append({"epoch": epoch + 1, "loss": loss})
        progress = {
            "status": "training",
            "epoch": epoch + 1,
            "epochs": args.epochs,
            "elapsed_seconds": round(time.time() - started),
            "loss": loss,
        }
        (target / "progress.json").write_text(json.dumps(progress, indent=2))
        print(f"Epoch {epoch + 1}/{args.epochs}: sparse training loss {loss:.4f}", flush=True)
    torch.save(
        {"architecture": "wallnet-v1", "state_dict": model.cpu().state_dict()}, target / "last.pt"
    )
    report = {
        "status": "experimental",
        "training_groups": sorted(frames),
        "history": history,
        "annotation_sha256": hashlib.sha256(
            (target / "annotation-snapshot.json").read_bytes()
        ).hexdigest(),
        "note": "Sparse wall strokes and collector-interior negatives. Loss is not wall coverage or planning accuracy. Not authorized for live deployment.",
    }
    (target / "report.json").write_text(json.dumps(report, indent=2))
    progress.update(status="complete", checkpoint=str((target / "last.pt").resolve()))
    (target / "progress.json").write_text(json.dumps(progress, indent=2))


if __name__ == "__main__":
    main()
