"""Small, separately trained wall segmentation network; no gameplay controls."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def block(a, b):
    return nn.Sequential(
        nn.Conv2d(a, b, 3, padding=1), nn.ReLU(), nn.Conv2d(b, b, 3, padding=1), nn.ReLU()
    )


class WallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = block(3, 16)
        self.second = block(16, 32)
        self.middle = block(32, 64)
        self.up_second = block(96, 32)
        self.up_first = block(48, 16)
        self.output = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        a = self.first(x)
        b = self.second(F.max_pool2d(a, 2))
        c = self.middle(F.max_pool2d(b, 2))
        d = self.up_second(
            torch.cat(
                (F.interpolate(c, size=b.shape[-2:], mode="bilinear", align_corners=False), b), 1
            )
        )
        return self.output(
            self.up_first(
                torch.cat(
                    (F.interpolate(d, size=a.shape[-2:], mode="bilinear", align_corners=False), a),
                    1,
                )
            )
        )


class WallSegmenter:
    def __init__(self, checkpoint, device="cpu"):
        self.device = torch.device(device if device != "0" else "cuda:0")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state.get("architecture") != "wallnet-v1":
            raise ValueError("Unsupported wall checkpoint")
        self.model = WallNet().to(self.device).eval()
        self.model.load_state_dict(state["state_dict"])

    @torch.inference_mode()
    def probabilities(self, image, tile=384, margin=32):
        rgb = np.asarray(image.convert("RGB"))
        height, width = rgb.shape[:2]
        result = np.zeros((height, width), np.float32)
        for y in range(0, height, tile):
            for x in range(0, width, tile):
                ax, ay = max(0, x - margin), max(0, y - margin)
                bx, by = min(width, x + tile + margin), min(height, y + tile + margin)
                patch = (
                    torch.from_numpy(rgb[ay:by, ax:bx].copy())
                    .permute(2, 0, 1)
                    .float()
                    .div_(255)
                    .unsqueeze(0)
                    .to(self.device)
                )
                p = self.model(patch).sigmoid()[0, 0].cpu().numpy()
                ex, ey = min(width, x + tile), min(height, y + tile)
                result[y:ey, x:ex] = p[y - ay : ey - ay, x - ax : ex - ax]
        return result


def wall_cells(probability, field, threshold=0.5, cell=8, excluded=()):
    """Rasterize probabilities without filling the interiors of wall enclosures."""
    from vision import Detection

    x, y, width, height = field
    if probability.shape != (height, width):
        raise ValueError("Wall probability map must match the playfield")
    probability = probability.copy()
    for bx, by, bw, bh in excluded:
        left, top = max(0, bx - x), max(0, by - y)
        right, bottom = min(width, bx + bw - x), min(height, by + bh - y)
        if left < right and top < bottom:
            probability[top:bottom, left:right] = 0
    found = []
    for py in range(0, height, cell):
        row = []
        for px in range(0, width, cell):
            patch = probability[py : min(height, py + cell), px : min(width, px + cell)]
            row.append(float((patch >= threshold).mean()) >= 0.5)
        # Merge adjacent occupied cells along each row, never across a hole.
        begin = None
        for index, occupied in enumerate([*row, False]):
            if occupied and begin is None:
                begin = index
            elif not occupied and begin is not None:
                left, right = begin * cell, min(width, index * cell)
                found.append(
                    Detection(
                        "wall",
                        (x + left, y + py, right - left, min(cell, height - py)),
                        float(probability[py : py + cell, left:right].mean()),
                    )
                )
                begin = None
    return found
