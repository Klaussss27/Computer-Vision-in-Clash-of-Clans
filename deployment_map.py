"""Visible deployment regions, using observed red edges and exterior flood fill."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class DeploymentMap:
    inside: np.ndarray
    outside: np.ndarray
    edge: np.ndarray
    usable: np.ndarray
    field: tuple
    problem: str = ""


def combine_regions(regions, anchors):
    if not regions:
        raise ValueError("At least one supported boundary estimate is required")
    first = regions[0]
    if any(
        r.field != first.field
        or r.inside.shape != first.inside.shape
        or not np.array_equal(r.usable, first.usable)
        for r in regions
    ):
        raise ValueError("Boundary estimates must use the same field and exclusions")
    inside = np.logical_or.reduce([r.inside for r in regions]) & first.usable
    # A point is deployable only if every estimate considers it exterior.
    outside = np.logical_and.reduce([r.outside for r in regions]) & ~inside & first.usable
    edge = np.logical_or.reduce([r.edge for r in regions]) & first.usable
    x, y, w, h = first.field
    problem = ""
    if not anchors or any(
        not (0 <= px - x < w and 0 <= py - y < h) or not inside[py - y, px - x]
        for px, py in anchors
    ):
        problem = (
            "Combined boundary does not contain every visible collector; geometry is unresolved."
        )
    return DeploymentMap(inside, outside, edge, first.usable, first.field, problem)


def red_edge_mask(rgb, mode="legacy", model_path="models/boundary-pixels-01.pt", threshold=0.8):
    if mode == "learned":
        from boundary_model import learned_edges

        return learned_edges(rgb, model_path, threshold)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    if mode in ("legacy", "directional"):
        red = cv2.inRange(hsv, (0, 110, 35), (27, 255, 255))
        red |= cv2.inRange(hsv, (170, 110, 65), (179, 255, 235))
        if mode == "directional":
            original = red.copy()
            for rising in (False, True):
                kernel = np.zeros((19, 25), np.uint8)
                cv2.line(kernel, (0, 18 if rising else 0), (24, 0 if rising else 18), 1, 1)
                red |= cv2.morphologyEx(original, cv2.MORPH_CLOSE, kernel)
        return red
    if mode != "ridge":
        raise ValueError("Boundary edge mode must be legacy, ridge, directional, or learned")
    # A deployment line is a narrow red ridge between ground pixels. Roofs,
    # fires, and orange walls can be red too, but usually occupy a solid area.
    pixels = rgb.astype(np.float32)
    chroma = pixels[:, :, 0] - 0.8 * pixels[:, :, 1] - 0.2 * pixels[:, :, 2]
    ridge = np.zeros(chroma.shape, bool)
    for dx in (-2, 2):
        before = np.roll(chroma, (3, dx), (0, 1))
        after = np.roll(chroma, (-3, -dx), (0, 1))
        ridge |= (chroma - before > 15) & (chroma - after > 15)
    ridge &= (chroma > 12) & (hsv[:, :, 1] > 65) & ((hsv[:, :, 0] <= 30) | (hsv[:, :, 0] >= 170))
    ridge[:3] = False
    ridge[-3:] = False
    ridge[:, :2] = False
    ridge[:, -2:] = False
    return ridge.astype(np.uint8) * 255


def visible_deployment(
    image,
    field,
    exclusions=(),
    collector_centers=(),
    edge_mode="legacy",
    model_path="models/boundary-pixels-01.pt",
    threshold=0.8,
):
    if edge_mode == "ensemble":
        anchors = list(collector_centers)
        candidates = [
            visible_deployment(image, field, exclusions, anchors, mode, model_path, threshold)
            for mode in ("legacy", "learned")
        ]
        supported = [
            r
            for r in candidates
            if r.problem
            in (
                "",
                "Deployment region does not contain every detected collector; geometry is unresolved.",
            )
        ]
        return combine_regions(supported, anchors) if supported else candidates[0]
    x, y, width, height = field
    rgb = np.asarray(image.convert("RGB"))[y : y + height, x : x + width]
    if rgb.shape[:2] != (height, width):
        raise ValueError("Deployment field must fit the image")
    red = red_edge_mask(rgb, edge_mode, model_path, threshold)
    usable = np.ones((height, width), np.uint8)
    for bx, by, bw, bh in exclusions:
        left, top = max(0, bx - x), max(0, by - y)
        right, bottom = min(width, bx + bw - x), min(height, by + bh - y)
        if left < right and top < bottom:
            usable[top:bottom, left:right] = 0
    red[usable == 0] = 0
    # Preserve short staircase segments at wall/footprint corners as well as
    # long edges. Dilation repairs antialiasing holes, not occluded buildings.
    edge = cv2.dilate(red, np.ones((3, 3), np.uint8))
    edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    lines = cv2.HoughLinesP(red, 1, np.pi / 360, threshold=20, minLineLength=25, maxLineGap=18)
    if lines is not None:
        for ax, ay, bx, by in lines.reshape(-1, 4):
            dx, dy = abs(int(bx) - int(ax)), abs(int(by) - int(ay))
            if dx and 0.65 <= dy / dx <= 0.85:
                cv2.line(edge, (ax, ay), (bx, by), 255, 5)
    observed = edge != 0
    # Occluded UI is unknown and impassable, rather than a gap through which
    # flood fill or a planned route may pass.
    edge[usable == 0] = 255
    free = (edge == 0).astype(np.uint8)
    _, labels = cv2.connectedComponents(free)
    corner_roles = {
        int(labels[py, px])
        for px, py in ((0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1))
    }
    corner_roles.discard(0)
    for py in range(0, height, 40):
        for px in (0, width - 1):
            if usable[py, px] and labels[py, px]:
                corner_roles.add(int(labels[py, px]))
    outside = np.isin(labels, list(corner_roles)) & (free != 0)
    inside = (free != 0) & ~outside & (usable != 0)
    # Red/orange building pixels are barriers to flood fill, but are still part
    # of the forbidden region when enclosed by its perimeter. Keep the region
    # supported by enclosed free area, not isolated roofs. Spread-out bases can
    # have several separate forbidden regions; retaining only the largest one
    # discards the collectors in the other islands.
    _, regions = cv2.connectedComponents(((~outside) & (usable != 0)).astype(np.uint8))
    support = np.bincount(regions[inside], minlength=int(regions.max()) + 1)
    support[0] = 0
    supported_regions = np.flatnonzero(support)
    supported_area = int(support.sum())
    inside = np.isin(regions, supported_regions) & (usable != 0)
    problem = ""
    # Dense buildings consume enclosed free pixels. Assess the filled region
    # separately, while still requiring substantial enclosed free support so
    # isolated red roofs cannot establish a deployment region.
    if (
        np.count_nonzero(edge) < 100
        or supported_area < width * height * 0.02
        or np.count_nonzero(inside) < width * height * 0.08
    ):
        problem = "Visible red edges do not establish an enclosed deployment region."
    anchors = list(collector_centers)
    if not anchors:
        problem = "Collector anchors are required to check the experimental deployment region."
    elif any(
        not (0 <= px - x < width and 0 <= py - y < height) or not inside[py - y, px - x]
        for px, py in anchors
    ):
        problem = (
            "Deployment region does not contain every detected collector; geometry is unresolved."
        )
    return DeploymentMap(
        inside,
        outside & (usable != 0),
        observed & (usable != 0),
        usable != 0,
        tuple(field),
        problem,
    )
