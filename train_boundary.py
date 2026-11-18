"""Train an experimental boundary appearance model from sparse training references."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn

from boundary_model import features, network
from deployment_map import red_edge_mask


def main():
    torch.manual_seed(42)
    torch.set_num_threads(4)
    refs_path = Path("boundary_annotations.json")
    refs = json.loads(refs_path.read_text())
    samples = []
    labels = []
    sources = []
    for record in refs["frames"]:
        if record["split"] != "train":
            continue
        path = Path(record["image"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("Changed boundary reference image")
        rgb = np.asarray(Image.open(path).convert("RGB"))
        line = np.zeros(rgb.shape[:2], np.uint8)
        for points in record["lines"]:
            cv2.polylines(line, [np.asarray(points, np.int32)], False, 1, 1)
        nearby = cv2.dilate(line, np.ones((11, 11), np.uint8)) != 0
        # Only inspected stroke neighborhoods provide positive labels. Neither
        # unlabeled pixels nor collector boxes imply a boundary label.
        positive = nearby & (red_edge_mask(rgb, "ridge") != 0)
        negative = np.zeros(line.shape, np.uint8)
        for x, y in record["inside"] + record["outside"]:
            cv2.circle(negative, (x, y), 10, 1, -1)
        negative = (negative != 0) & ~nearby
        values = features(rgb)
        for mask, label in ((positive, 1), (negative, 0)):
            selected = values[mask]
            samples.append(selected)
            labels.append(np.full((len(selected), 1), label, np.float32))
        sources.append(
            {
                "base": record["group"],
                "sha256": record["sha256"],
                "positive_pixels": int(positive.sum()),
                "negative_pixels": int(negative.sum()),
            }
        )
    x = torch.from_numpy(np.concatenate(samples))
    y = torch.from_numpy(np.concatenate(labels))
    model = network()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    weight = (y == 0).sum() / (y == 1).sum().clamp(min=1)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=weight)
    for epoch in range(150):
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
    output = Path("models/boundary-pixels-01.pt")
    torch.save({"format": "boundary-pixels-v1", "weights": model.state_dict()}, output)
    report = {
        "role": "experimental appearance model; no full-mask or live validation",
        "references_sha256": hashlib.sha256(refs_path.read_bytes()).hexdigest(),
        "sources": sources,
        "samples": len(x),
        "epochs": 150,
        "final_training_loss": float(loss.detach()),
        "checkpoint_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
