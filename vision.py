"""Template detection and per-base route planning in captured-window pixels."""

import heapq
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import ImageDraw

from attack_policy import jumps_enabled, required_army

COLLECTORS = ("gold_mine", "elixir_collector", "dark_drill")
KINDS = (*COLLECTORS, "wall", "rubble")


@dataclass
class Detection:
    kind: str
    box: tuple
    score: float

    @property
    def center(self):
        x, y, w, h = self.box
        return (x + w // 2, y + h // 2)

    @property
    def ground_center(self):
        # Tall sprites extend upward from their ground footprint.
        x, y, w, h = self.box
        return (x + w // 2, y + round(h * 0.8))


@dataclass
class Target:
    collector: Detection
    spawn: tuple | None
    layers: int | None
    jump: tuple | None
    reason: str
    path: list | None = None

    @property
    def eligible(self):
        return self.spawn is not None and self.layers in (0, 1)


@dataclass
class Plan:
    detections: list
    targets: list
    boundary: list
    problems: list
    proposal_only: bool = False

    def summary(self):
        return {
            "collectors": len(self.targets),
            "eligible": sum(target.eligible for target in self.targets),
            "problems": self.problems,
            "proposal_only": self.proposal_only,
            "targets": [asdict(target) for target in self.targets],
        }


def overlap(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    area = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(0, min(ay + ah, by + bh) - max(ay, by))
    return area / max(1, min(aw * ah, bw * bh))


class Detector:
    def __init__(self, settings):
        self.settings = settings
        self.samples = []
        for kind in KINDS:
            for path in sorted(Path(settings["template_dir"], kind).glob("*.png")):
                image = cv2.imread(str(path))
                if image is None or min(image.shape[:2]) < 6:
                    raise ValueError(f"Invalid vision sample: {path}")
                if float(image.std()) < 3:
                    raise ValueError(f"Vision sample has too little detail: {path}")
                self.samples.append((kind, image))

    @property
    def kinds(self):
        return {kind for kind, _ in self.samples}

    def detect(self, image, kinds=None, region=None):
        rgb = np.array(image.convert("RGB"))
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        x, y, w, h = self.settings["playfield"]
        if region is not None:
            rx, ry, rw, rh = region
            right, bottom = min(x + w, rx + rw, image.width), min(y + h, ry + rh, image.height)
            x, y = max(x, rx, 0), max(y, ry, 0)
            w, h = right - x, bottom - y
            if min(w, h) <= 0:
                return []
        field = frame[y : y + h, x : x + w]
        candidates = []
        for kind, sample in self.samples:
            if kinds is not None and kind not in kinds:
                continue
            for scale in self.settings.get("scales", [0.9, 1.0, 1.1]):
                sw = max(6, round(sample.shape[1] * scale))
                sh = max(6, round(sample.shape[0] * scale))
                if sw > w or sh > h:
                    continue
                resized = cv2.resize(sample, (sw, sh))
                scores = cv2.matchTemplate(field, resized, cv2.TM_CCOEFF_NORMED)
                # Suppress each peak before searching for another instance.
                for _ in range(150):
                    _, score, _, point = cv2.minMaxLoc(scores)
                    if score < self.settings.get("confidence", 0.88):
                        break
                    px, py = point
                    candidates.append(Detection(kind, (x + px, y + py, sw, sh), float(score)))
                    scores[
                        max(0, py - sh // 2) : py + sh // 2 + 1,
                        max(0, px - sw // 2) : px + sw // 2 + 1,
                    ] = -1
        found = []
        for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
            if not any(overlap(candidate.box, previous.box) > 0.35 for previous in found):
                found.append(candidate)
        return found

    def detect_collectors(self, image):
        return self.detect(image, kinds=COLLECTORS)

    def detect_for_planning(self, image):
        return self.detect(image, kinds=set(COLLECTORS) | {"wall"})

    def detect_rubble(self, image, region):
        return self.detect(image, kinds={"rubble"}, region=region)


def deployment_boundary(image, field):
    """Require an observed closed red perimeter; never close it across screen edges."""
    x, y, w, h = field
    hsv = cv2.cvtColor(np.array(image.convert("RGB"))[y : y + h, x : x + w], cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (0, 110, 65), (12, 255, 235))
    mask |= cv2.inRange(hsv, (170, 110, 65), (179, 255, 235))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    # A thin closed perimeter must enclose a large hole. Red buildings do not.
    holes = [
        contour
        for index, contour in enumerate(contours)
        if hierarchy[0][index][3] >= 0 and cv2.contourArea(contour) > w * h * 0.08
    ]
    if not holes:
        return []
    contour = max(holes, key=cv2.contourArea)
    polygon = cv2.approxPolyDP(contour, 3, True).reshape(-1, 2)
    if any(px <= 5 or py <= 5 or px >= w - 5 or py >= h - 5 for px, py in polygon):
        return []
    return [(int(px + x), int(py + y)) for px, py in polygon]


def route_to_collector(walls, start_cells, goal, inside, max_layers=2, max_steps=None):
    """Dijkstra route: stepping into a wall band costs one wall layer."""
    height, width = walls.shape
    queue = []
    costs, previous = {}, {}
    for cell in start_cells:
        sx, sy = cell
        if not (0 <= sx < width and 0 <= sy < height) or not inside[sy, sx] or walls[sy, sx] != 0:
            continue
        costs[cell] = (0, 0)
        heapq.heappush(queue, (0, 0, cell))
    while queue:
        layers, length, cell = heapq.heappop(queue)
        if costs.get(cell) != (layers, length):
            continue
        if cell == goal:
            path = [cell]
            while cell in previous:
                cell = previous[cell]
                path.append(cell)
            return layers, path[::-1]
        cx, cy = cell
        if max_steps is not None and length >= max_steps:
            continue
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < width and 0 <= ny < height) or not inside[ny, nx]:
                continue
            # Non-wall buildings use value 2 and cannot be crossed.
            if walls[ny, nx] == 2:
                continue
            extra = int(walls[ny, nx] == 1 and walls[cy, cx] != 1)
            cost = (layers + extra, length + 1)
            neighbor = (nx, ny)
            if cost[0] <= max_layers and cost < costs.get(neighbor, (999, 999999)):
                costs[neighbor] = cost
                previous[neighbor] = cell
                heapq.heappush(queue, (*cost, neighbor))
    return None, []


def grass_terrain(image, field):
    x, y, w, h = field
    hsv = cv2.cvtColor(np.asarray(image.convert("RGB"))[y : y + h, x : x + w], cv2.COLOR_RGB2HSV)
    green = cv2.inRange(hsv, (28, 50, 40), (95, 255, 245))
    return cv2.morphologyEx(green, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) != 0


def collector_visibility_problem(collector, settings):
    x, y, width, height = settings["playfield"]
    bx, by, bw, bh = collector.box
    margin = settings.get("collector_edge_margin", 8)
    if (
        bx < x + margin
        or by < y + margin
        or bx + bw > x + width - margin
        or by + bh > y + height - margin
    ):
        return "collector clipped by viewport; skip"
    if any(overlap(collector.box, box) > 0 for box in settings.get("ui_exclusions", [])):
        return "collector occluded by interface; skip"
    return ""


def make_plan(detections, boundary, settings, available_kinds, deployment=None, terrain=None):
    problems = []
    if settings.get("geometry_validated") is False:
        problems.append("Wall/deployment geometry has not passed validation.")
    if settings.get("planning_preview_only"):
        problems.append("Experimental wall references: complete wall geometry is not verified.")
    collectors = [item for item in detections if item.kind in COLLECTORS]
    if not collectors:
        problems.append("No supported collectors detected. Add matching samples in Vision samples.")
    if "wall" not in available_kinds:
        problems.append("Wall samples missing; wall depth cannot be estimated.")
    elif not any(item.kind == "wall" for item in detections):
        problems.append(
            "No walls recognized; cannot distinguish a wall-free base from missed walls."
        )
    if deployment is not None and deployment.problem:
        problems.append(deployment.problem)
    elif deployment is None and not boundary:
        problems.append("Complete red deployment perimeter not visible. Zoom out and re-analyze.")
    if problems:
        return Plan(
            detections,
            [Target(c, None, None, None, "not enough visual evidence") for c in collectors],
            boundary,
            problems,
        )
    x, y, width, height = settings["playfield"]
    cell = settings.get("grid_size", 8)
    shape = (math.ceil(height / cell), math.ceil(width / cell))
    usable = np.ones(shape, bool)
    known_outside = None
    if deployment is None:
        polygon = np.array([((px - x) // cell, (py - y) // cell) for px, py in boundary], np.int32)
        inside = np.zeros(shape, np.uint8)
        cv2.fillPoly(inside, [polygon], 1)
    else:
        if tuple(settings["playfield"]) != deployment.field:
            raise ValueError("Deployment map must match the planner playfield")
        ys = np.minimum(np.arange(shape[0]) * cell + cell // 2, height - 1)
        xs = np.minimum(np.arange(shape[1]) * cell + cell // 2, width - 1)
        inside = deployment.inside[np.ix_(ys, xs)].astype(np.uint8)
        usable = deployment.usable[np.ix_(ys, xs)]
        known_outside = deployment.outside[np.ix_(ys, xs)]
    # Seed only points outside the observed perimeter and away from UI margins.
    outer = cv2.dilate(inside, np.ones((7, 7), np.uint8))
    rim = outer - cv2.dilate(inside, np.ones((5, 5), np.uint8))
    rim[~usable] = 0
    if known_outside is not None:
        rim[~known_outside] = 0
    sy, sx = np.where(rim > 0)
    starts = [
        (int(px), int(py))
        for px, py in zip(sx, sy)
        if 2 < px < shape[1] - 3 and 2 < py < shape[0] - 3
    ]
    if not starts:
        return Plan(detections, [], boundary, ["No visible deployment space outside perimeter."])
    # The route is allowed to enter the perimeter from the seed ring.
    traversable = outer.astype(bool) & usable
    if known_outside is not None:
        traversable &= inside.astype(bool) | known_outside
    wall_mask = np.zeros(shape, np.uint8)
    for item in detections:
        if item.kind != "wall":
            continue
        bx, by, bw, bh = item.box
        cv2.rectangle(
            wall_mask,
            ((bx - x) // cell, (by - y) // cell),
            ((bx + bw - x) // cell, (by + bh - y) // cell),
            1,
            -1,
        )
    # Close tiny cracks between neighboring matched wall tiles.
    wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    terrain_obstacles = np.zeros(shape, np.uint8)
    if terrain is not None:
        if terrain.shape != (height, width):
            raise ValueError("Terrain mask must match the playfield")
        padded = np.pad(
            terrain,
            ((0, shape[0] * cell - height), (0, shape[1] * cell - width)),
            constant_values=False,
        )
        fraction = padded.reshape(shape[0], cell, shape[1], cell).mean(axis=(1, 3))
        terrain_obstacles[fraction < settings.get("terrain_green_min_fraction", 0.75)] = 2
    terrain_obstacles[wall_mask != 0] = 1
    targets = []
    for collector in collectors:
        visibility_problem = collector_visibility_problem(collector, settings)
        if visibility_problem:
            targets.append(Target(collector, None, None, None, visibility_problem))
            continue
        cx, cy = collector.ground_center
        goal = ((cx - x) // cell, (cy - y) // cell)
        if (
            not (0 <= goal[0] < shape[1] and 0 <= goal[1] < shape[0])
            or not inside[goal[1], goal[0]]
        ):
            targets.append(Target(collector, None, None, None, "outside observed perimeter"))
            continue
        # Do not route through the other known collector footprints.
        obstacles = terrain_obstacles.copy()
        if terrain is not None:
            # The target's own artwork is not grass. Permit contact across its
            # detected box while preserving every recognized wall cell. Other
            # buildings are applied afterwards and remain impassable.
            bx, by, bw, bh = collector.box
            contact = np.zeros(shape, np.uint8)
            cv2.rectangle(
                contact,
                ((bx - x) // cell, (by - y) // cell),
                ((bx + bw - x) // cell, (by + bh - y) // cell),
                1,
                -1,
            )
            obstacles[(contact != 0) & (wall_mask == 0)] = 0
        for building in detections:
            if building.kind == "building" and overlap(building.box, collector.box) < 0.3:
                bx, by, bw, bh = building.box
                cv2.rectangle(
                    obstacles,
                    ((bx - x) // cell, (by - y) // cell),
                    ((bx + bw - x) // cell, (by + bh - y) // cell),
                    2,
                    -1,
                )
        for other in collectors:
            if other is collector:
                continue
            bx, by, bw, bh = other.box
            cv2.rectangle(
                obstacles,
                ((bx - x) // cell, (by - y) // cell),
                ((bx + bw - x) // cell, (by + bh - y) // cell),
                2,
                -1,
            )
        local_starts = starts
        maximum = settings.get("max_approach_pixels")
        if maximum is not None:
            distances = [math.dist(point, goal) * cell for point in starts]
            nearest = min(distances)
            if nearest > maximum:
                targets.append(
                    Target(collector, None, None, None, "no nearby visible deployment space")
                )
                continue
            local_starts = [
                point
                for point, distance in zip(starts, distances)
                if distance <= nearest + settings.get("approach_slack_pixels", 32)
            ]
        layers, path = route_to_collector(
            obstacles,
            local_starts,
            goal,
            traversable,
            max_steps=math.ceil(maximum / cell) if maximum else None,
        )
        if layers is None or layers > 1:
            targets.append(
                Target(collector, None, layers, None, "two or more wall layers or no route")
            )
            continue
        if layers == 1 and not jumps_enabled(settings):
            targets.append(
                Target(collector, None, 1, None, "wall crossing skipped: Goblins-only mode")
            )
            continue
        px, py = path[0]
        spawn = (x + px * cell + cell // 2, y + py * cell + cell // 2)
        jump = None
        if layers == 1:
            crossings = [(gx, gy) for gx, gy in path if obstacles[gy, gx] == 1]
            if len(crossings) * cell > settings.get("max_wall_band_pixels", 64):
                targets.append(
                    Target(collector, None, None, None, "wall band too thick for one jump")
                )
                continue
            gx, gy = crossings[len(crossings) // 2]
            jump = (x + gx * cell + cell // 2, y + gy * cell + cell // 2)
        route = [(x + gx * cell + cell // 2, y + gy * cell + cell // 2) for gx, gy in path]
        targets.append(
            Target(
                collector,
                spawn,
                layers,
                jump,
                "exposed" if layers == 0 else "one wall layer; jump needed",
                route,
            )
        )
    return Plan(detections, targets, boundary, problems)


def create_detector(settings):
    backend = settings.get("backend", "templates")
    if backend == "yolo":
        from neural_detector import NeuralDetector

        return NeuralDetector(settings)
    if backend == "hybrid":
        from neural_detector import HybridDetector

        return HybridDetector(settings)
    if backend != "templates":
        raise ValueError("Vision backend must be templates, yolo, or hybrid")
    return Detector(settings)


class VisualPlanner:
    def __init__(self, settings):
        self.settings = settings
        self.detector = create_detector(settings)
        self.reference_gray = None

    @property
    def visible_mode(self):
        return (
            self.settings.get(
                "boundary_mode", "visible" if not jumps_enabled(self.settings) else "closed"
            )
            == "visible"
        )

    def visible_region(self, image, detections=None):
        from deployment_map import visible_deployment

        if detections is None:
            detections = self.detector.detect_collectors(image)
        return visible_deployment(
            image,
            self.settings["playfield"],
            self.settings.get("ui_exclusions", []),
            [
                d.ground_center
                for d in detections
                if d.kind in COLLECTORS and not collector_visibility_problem(d, self.settings)
            ],
            edge_mode=self.settings.get("boundary_edge_mode", "legacy"),
            model_path=self.settings.get("boundary_model_path", "models/boundary-pixels-01.pt"),
            threshold=self.settings.get("boundary_threshold", 0.8),
        )

    def view_ready(self, image):
        if self.visible_mode:
            return not self.visible_region(image).problem
        return bool(deployment_boundary(image, self.settings["playfield"]))

    def frame_gray(self, image):
        x, y, w, h = self.settings["playfield"]
        gray = cv2.cvtColor(
            np.asarray(image.convert("RGB"))[y : y + h, x : x + w], cv2.COLOR_RGB2GRAY
        )
        return cv2.resize(gray, (480, 240)).astype(np.float32)

    def aligned(self, image):
        if self.reference_gray is None:
            return False
        current = self.frame_gray(image)
        shift, response = cv2.phaseCorrelate(self.reference_gray, current)
        _, _, width, height = self.settings["playfield"]
        return (
            response >= 0.1
            and abs(shift[0]) * width / 480 <= 4
            and abs(shift[1]) * height / 240 <= 4
        )

    def analyze(self, image):
        self.reference_gray = self.frame_gray(image)
        detect = getattr(self.detector, "detect_for_planning", self.detector.detect)
        detections = detect(image)
        if self.visible_mode:
            region = self.visible_region(image, detections)
            contours, _ = cv2.findContours(
                region.inside.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            contour = max(contours, key=cv2.contourArea) if contours else None
            x, y, _, _ = self.settings["playfield"]
            boundary = (
                [
                    (int(px) + x, int(py) + y)
                    for px, py in cv2.approxPolyDP(contour, 3, True).reshape(-1, 2)
                ]
                if contour is not None
                else []
            )
            terrain = (
                grass_terrain(image, self.settings["playfield"])
                if self.settings.get("terrain_guard", not jumps_enabled(self.settings))
                else None
            )
            return make_plan(
                detections, boundary, self.settings, self.detector.kinds, region, terrain
            )
        boundary = deployment_boundary(image, self.settings["playfield"])
        return make_plan(detections, boundary, self.settings, self.detector.kinds)

    def analyze_offline(self, image):
        """Show saved-image proposals without producing an executable plan."""
        preview = VisualPlanner.__new__(VisualPlanner)
        preview.settings = dict(self.settings, geometry_validated=True, planning_preview_only=False)
        preview.detector = self.detector
        preview.reference_gray = None
        plan = preview.analyze(image)
        plan.proposal_only = True
        return plan

    def destroyed(self, image, collector):
        """Disappearance alone is not evidence of destruction."""
        if not self.aligned(image):
            return False
        found = (
            self.detector.detect_collectors(image)
            if hasattr(self.detector, "detect_collectors")
            else self.detector.detect(image)
        )
        still_present = any(
            d.kind in COLLECTORS and overlap(d.box, collector.box) > 0.3 for d in found
        )
        x, y, w, h = collector.box
        ruins = (
            self.detector.detect_rubble(image, (x - 24, y - 24, w + 48, h + 48))
            if hasattr(self.detector, "detect_rubble")
            else found
        )
        rubble = any(d.kind == "rubble" and overlap(d.box, collector.box) > 0.3 for d in ruins)
        return rubble and not still_present


def draw_plan(image, plan):
    image = image.copy()
    draw = ImageDraw.Draw(image)
    if plan.boundary:
        draw.line(plan.boundary + [plan.boundary[0]], fill="cyan", width=3)
    for item in plan.detections:
        if item.kind == "building":
            x, y, w, h = item.box
            draw.rectangle((x, y, x + w, y + h), outline="steelblue", width=1)
        if item.kind == "wall":
            x, y, w, h = item.box
            draw.rectangle((x, y, x + w, y + h), outline="orange", width=2)
    for target in plan.targets:
        x, y, w, h = target.collector.box
        color = "lime" if target.eligible else "red"
        draw.rectangle((x, y, x + w, y + h), outline=color, width=3)
        draw.text(
            (x, max(0, y - 15)),
            f"{target.collector.kind}: {target.reason}",
            fill=color,
            stroke_width=1,
            stroke_fill="black",
        )
        if target.spawn:
            sx, sy = target.spawn
            draw.ellipse((sx - 7, sy - 7, sx + 7, sy + 7), outline="lime", width=3)
            draw.line(
                target.path or (target.spawn, target.collector.ground_center), fill=color, width=2
            )
        if target.jump:
            jx, jy = target.jump
            draw.ellipse((jx - 12, jy - 12, jx + 12, jy + 12), outline="magenta", width=3)
    return image


def validate_vision(config):
    settings = config["vision"]
    if type(settings.get("use_jump_spells", True)) is not bool:
        raise ValueError("use_jump_spells must be true or false")
    if settings.get("planning_preview_only"):
        raise ValueError("An experimental preview profile cannot be used for live attacks.")
    if settings.get("geometry_validated") is not True:
        raise ValueError("Wall/deployment geometry must pass validation before live attacks.")
    if not config.get("army_detection"):
        raise ValueError("Army icon recognition must be calibrated before live attacks.")
    detector = create_detector(settings)
    for kind in KINDS:
        if kind not in detector.kinds:
            raise ValueError(f"Add {kind} samples in Vision samples / analyze")
    for name in ("surrender",):
        if name not in config["templates"]:
            raise ValueError(f"Missing template: {name}")
    for name in ("end_battle", "confirm_surrender"):
        if name not in config["buttons"]:
            raise ValueError(f"Missing button: {name}")
    width, height = config["capture_size"]
    x, y, w, h = settings["playfield"]
    if min(x, y) < 0 or min(w, h) <= 0 or x + w > width or y + h > height:
        raise ValueError("Vision playfield must fit inside capture_size")
    if not 0.7 <= settings["confidence"] <= 0.99:
        raise ValueError("Vision confidence must be between 0.7 and 0.99")
    if not 4 <= settings["grid_size"] <= 24:
        raise ValueError("Vision grid_size must be between 4 and 24")
    scales = settings.get("scales", [0.9, 1.0, 1.1])
    if (
        not scales
        or len(scales) > 10
        or any(not isinstance(s, (int, float)) or not 0.25 <= s <= 2 for s in scales)
    ):
        raise ValueError("Use 1 to 10 vision scales between 0.25 and 2")
    for name, minimum, maximum in (
        ("goblin_budget", 1, 300),
        ("jump_budget", 0, 20),
        ("max_targets", 1, 30),
        ("goblins_per_batch", 1, 20),
        ("max_batches_per_target", 1, 5),
        ("target_wait_seconds", 2, 30),
        ("max_zoom_out_ticks", 0, 12),
    ):
        if name == "jump_budget" and not settings.get("use_jump_spells", True):
            continue
        value = settings[name]
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError(f"Vision {name} must be an integer from {minimum} to {maximum}")
    for name in required_army(settings):
        point = config["army_slots"][name]
        if len(point) != 2 or not (0 <= point[0] < width and 0 <= point[1] < height):
            raise ValueError(f"Invalid army slot: {name}")
