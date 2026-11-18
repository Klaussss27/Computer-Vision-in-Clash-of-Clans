"""Recognize UI markers without treating dimmed modal backgrounds as active UI."""

import cv2
import numpy as np


def load_state_templates(config):
    found = {}
    for state, path in config["templates"].items():
        variants = []
        for filename in [path, *config.get("template_variants", {}).get(state, [])]:
            template = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            if template is None or min(template.shape) < 10 or float(template.std()) < 3:
                raise ValueError(f"Invalid state template: {filename}")
            variants.append(template)
        found[state] = variants
    return found


def marker_match(screen, template, confidence=0.9, max_mean_error=0.12):
    if isinstance(template, (list, tuple)):
        return any(marker_match(screen, item, confidence, max_mean_error) for item in template)
    gray = cv2.cvtColor(np.asarray(screen.convert("RGB")), cv2.COLOR_RGB2GRAY)
    height, width = template.shape
    if height > gray.shape[0] or width > gray.shape[1] or template.std() < 3:
        return False
    scores = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, (x, y) = cv2.minMaxLoc(scores)
    if score < confidence:
        return False
    patch = gray[y : y + height, x : x + width]
    # Normalized correlation alone is invariant to modal dimming.
    error = float(cv2.absdiff(patch, template).mean()) / 255
    return error <= max_mean_error
