"""Inference adapter for a collector detector trained on local annotations."""

from pathlib import Path

from vision import COLLECTORS, KINDS, Detection, Detector


def confidence_settings(settings, kinds):
    default = settings.get("model_confidence", 0.6)
    thresholds = settings.get("model_class_confidence", {})
    if not isinstance(thresholds, dict) or any(kind not in kinds for kind in thresholds):
        raise ValueError("Per-class confidence keys must match model class names")
    for value in [default, *thresholds.values()]:
        if type(value) not in (int, float) or not 0 < value <= 1:
            raise ValueError("Confidence thresholds must be numbers between 0 and 1")
    return default, thresholds, min(thresholds.get(kind, default) for kind in kinds)


class NeuralDetector:
    def __init__(self, settings, required_classes=KINDS):
        checkpoint = Path(settings["model_path"])
        if not checkpoint.is_file():
            raise ValueError(
                "Trained detector missing. Label bases and run train_detector.py first."
            )
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise ValueError(
                "Install model dependencies: python -m pip install -r requirements-ml.txt"
            ) from error
        self.model = YOLO(str(checkpoint))
        self.settings = settings
        names = self.model.names
        self.kinds = set(names.values() if isinstance(names, dict) else names)
        if self.kinds != set(required_classes):
            raise ValueError(
                "Checkpoint classes do not match the collector dataset. A general pretrained model cannot be used here."
            )

    def detect(self, image):
        x, y, width, height = self.settings["playfield"]
        field = image.crop((x, y, x + width, y + height)).convert("RGB")
        default, thresholds, confidence = confidence_settings(
            self.settings, getattr(self, "kinds", COLLECTORS)
        )
        result = self.model.predict(
            field,
            conf=confidence,
            imgsz=self.settings.get("model_image_size", 1280),
            device=self.settings.get("model_device", "cpu"),
            max_det=1000,
            verbose=False,
        )[0]
        found = []
        if result.boxes is None:
            return found
        boxes = result.boxes.xyxy.cpu().tolist()
        classes = result.boxes.cls.cpu().tolist()
        scores = result.boxes.conf.cpu().tolist()
        for box, kind, score in zip(boxes, classes, scores):
            name = result.names[int(kind)]
            if score < thresholds.get(name, default):
                continue
            x1, y1, x2, y2 = box
            margin = self.settings.get("model_edge_margin", 0)
            if (
                name in COLLECTORS
                and margin > 0
                and (x1 <= margin or y1 <= margin or x2 >= width - margin or y2 >= height - margin)
            ):
                continue
            left, top = max(0, round(x1)), max(0, round(y1))
            right, bottom = min(width, round(x2)), min(height, round(y2))
            if right > left and bottom > top:
                found.append(
                    Detection(name, (x + left, y + top, right - left, bottom - top), float(score))
                )
        return found


class HybridDetector:
    """Collector model plus local wall/rubble samples, without inventing missing classes."""

    def __init__(self, settings):
        if settings.get("wall_model_path") and not settings.get("planning_preview_only"):
            raise ValueError(
                "The experimental wall network is available for offline previews only."
            )
        self.settings = settings
        self.collectors = NeuralDetector(settings, required_classes=COLLECTORS)
        self.rubble = Detector(settings)
        structure_settings = dict(settings)
        if settings.get("structure_template_dir"):
            structure_settings["template_dir"] = settings["structure_template_dir"]
        structure_settings["confidence"] = settings.get(
            "structure_confidence", settings.get("confidence", 0.88)
        )
        self.structures = Detector(structure_settings)
        self.structures.samples = [
            (kind, sample) for kind, sample in self.structures.samples if kind in ("wall", "rubble")
        ]
        if settings.get("structure_template_dir"):
            # Wall-reference packs need not duplicate the calibrated rubble
            # examples. Keep destruction recognition when changing wall packs.
            original = Detector(settings)
            self.structures.samples = [
                (kind, sample) for kind, sample in self.structures.samples if kind == "wall"
            ]
            self.structures.samples.extend(
                (kind, sample) for kind, sample in original.samples if kind == "rubble"
            )
        self.kinds = self.collectors.kinds | self.structures.kinds
        self.walls = self.buildings = None
        if settings.get("wall_model_path"):
            from wall_model import WallSegmenter

            self.walls = WallSegmenter(
                settings["wall_model_path"], settings.get("model_device", "cpu")
            )
            self.structures.samples = [
                (kind, sample) for kind, sample in self.structures.samples if kind == "rubble"
            ]
            self.kinds.add("wall")
            if settings.get("building_model_path"):
                building_settings = dict(
                    settings,
                    model_path=settings["building_model_path"],
                    model_confidence=0.5,
                    model_class_confidence={},
                )
                self.buildings = NeuralDetector(building_settings, required_classes=("building",))
                self.kinds.add("building")

    def detect(self, image, include_rubble=True):
        collectors = self.collectors.detect(image)
        structures = self.structures.detect(image, kinds=None if include_rubble else {"wall"})
        if self.walls is None:
            return collectors + structures
        from wall_model import wall_cells

        x, y, width, height = self.settings["playfield"]
        buildings = self.buildings.detect(image) if self.buildings else []
        probability = self.walls.probabilities(image.crop((x, y, x + width, y + height)))
        excluded = [d.box for d in buildings] + self.settings.get("ui_exclusions", [])
        walls = wall_cells(probability, (x, y, width, height), excluded=excluded)
        return collectors + structures + walls + buildings

    def detect_for_planning(self, image):
        return self.detect(image, include_rubble=False)

    def detect_collectors(self, image):
        return self.collectors.detect(image)

    def detect_rubble(self, image, region):
        return self.rubble.detect_rubble(image, region)
