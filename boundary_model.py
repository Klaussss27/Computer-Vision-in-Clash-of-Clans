"""Experimental pixel classifier for deployment-line appearance."""

from functools import lru_cache

import cv2
import numpy as np
import torch
from torch import nn


def features(rgb):
    pixels = rgb.astype(np.float32) / 255
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32) / np.array(
        [179, 255, 255], np.float32
    )
    chroma = pixels[:, :, 0] - 0.8 * pixels[:, :, 1] - 0.2 * pixels[:, :, 2]
    channels = [pixels, hsv, chroma[:, :, None]]
    for distance in (2, 4, 7):
        for direction in (-1, 1):
            before = np.roll(chroma, (distance, direction * max(1, round(distance * 0.75))), (0, 1))
            after = np.roll(
                chroma, (-distance, -direction * max(1, round(distance * 0.75))), (0, 1)
            )
            channels.append((chroma - np.maximum(before, after))[:, :, None])
    channels.append(cv2.GaussianBlur(chroma, (11, 11), 0)[:, :, None])
    return np.concatenate(channels, axis=2)


def network():
    return nn.Sequential(
        nn.Linear(14, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1)
    )


@lru_cache(maxsize=2)
def load_model(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != "boundary-pixels-v1":
        raise ValueError("Unsupported boundary checkpoint")
    model = network()
    model.load_state_dict(checkpoint["weights"])
    return model.eval()


def learned_edges(rgb, path, threshold=0.8):
    if not 0 < threshold < 1:
        raise ValueError("Boundary threshold must be between 0 and 1")
    model = load_model(str(path))
    values = features(rgb)
    flat = values.reshape(-1, values.shape[2])
    result = np.empty(len(flat), np.float32)
    with torch.inference_mode():
        for start in range(0, len(flat), 131072):
            chunk = torch.from_numpy(flat[start : start + 131072])
            result[start : start + len(chunk)] = model(chunk).sigmoid().numpy().ravel()
    mask = (result.reshape(rgb.shape[:2]) >= threshold).astype(np.uint8) * 255
    mask[:7] = 0
    mask[-7:] = 0
    mask[:, :7] = 0
    mask[:, -7:] = 0
    return mask
