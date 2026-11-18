"""Locate army cards and read counts from a supplied image; never captures input."""

import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


@dataclass
class ArmyCard:
    kind: str
    point: tuple
    count: int
    score: float


class ArmyReader:
    def __init__(self, settings, ocr=None):
        self.settings = settings
        self.samples = {}
        self.empty_samples = {}
        self.variant_samples = {}
        self.zero_samples = {}
        if ocr is None:
            import pytesseract

            ocr = pytesseract
        self.ocr = ocr
        for kind, thresholds in settings.get("count_thresholds", {}).items():
            if (
                len(thresholds) != 2
                or len(set(thresholds)) != 2
                or any(type(t) is not int or not 0 < t < 255 for t in thresholds)
            ):
                raise ValueError("Army counts require two distinct valid thresholds")
        for kind, path in settings["templates"].items():
            sample = cv2.imread(str(path))
            if sample is None or min(sample.shape[:2]) < 10 or sample.std() < 3:
                raise ValueError("Missing or invalid army icon: " + str(path))
            self.samples[kind] = sample
        for kind, path in settings.get("empty_templates", {}).items():
            sample = cv2.imread(str(path))
            if (
                kind not in self.samples
                or sample is None
                or min(sample.shape[:2]) < 10
                or sample.std() < 3
            ):
                raise ValueError("Missing or invalid empty army icon: " + str(path))
            self.empty_samples[kind] = sample
        for kind, paths in settings.get("variant_templates", {}).items():
            self.variant_samples[kind] = []
            for reference in paths:
                path = reference["path"] if isinstance(reference, dict) else reference
                offset = reference.get("count_offset") if isinstance(reference, dict) else None
                sample = cv2.imread(str(path))
                if (
                    kind not in self.samples
                    or sample is None
                    or min(sample.shape[:2]) < 10
                    or sample.std() < 3
                ):
                    raise ValueError("Missing or invalid selected army icon: " + str(path))
                self.variant_samples[kind].append((sample, offset))
        for kind, path in settings.get("empty_count_templates", {}).items():
            sample = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if kind not in self.samples or sample is None or sample.std() < 3:
                raise ValueError("Missing or invalid zero count marker: " + str(path))
            self.zero_samples[kind] = sample

    def read(self, image, kind):
        if kind not in self.samples:
            raise ValueError("Army icon not configured: " + kind)
        x, y, width, height = self.settings["region"]
        if (
            min(x, y) < 0
            or min(width, height) <= 0
            or x + width > image.width
            or y + height > image.height
        ):
            raise RuntimeError("Army region does not fit the image; no selection made")
        field = cv2.cvtColor(
            np.asarray(image.convert("RGB"))[y : y + height, x : x + width], cv2.COLOR_RGB2BGR
        )
        candidates = []
        artwork = (
            [(False, self.samples[kind], None)]
            + [(False, s, offset) for s, offset in self.variant_samples.get(kind, [])]
            + ([(True, self.empty_samples[kind], None)] if kind in self.empty_samples else [])
        )
        for empty, sample, offset in artwork:
            if sample.shape[0] > height or sample.shape[1] > width:
                raise RuntimeError("Army reference is larger than its search region")
            scores = cv2.matchTemplate(field, sample, cv2.TM_CCOEFF_NORMED)
            _, score, _, point = cv2.minMaxLoc(scores)
            candidates.append((score, point, sample, scores, empty, offset))
        score, (px, py), sample, _, is_empty, offset = max(candidates, key=lambda item: item[0])
        if score < self.settings.get("confidence", 0.9):
            raise RuntimeError(f"{kind} card is not confidently visible; no selection made")
        sh, sw = sample.shape[:2]
        for _, _, _, scores, _, _ in candidates:
            scores[
                max(0, py - sh // 2) : py + sh // 2 + 1, max(0, px - sw // 2) : px + sw // 2 + 1
            ] = -1
            if float(scores.max()) > score - self.settings.get("ambiguity_margin", 0.03):
                raise RuntimeError("Ambiguous army icon; no selection made")
        dx, dy, cw, ch = self.settings.get("count_offsets", {}).get(
            kind, self.settings.get("count_offset", [18, -42, 66, 28])
        )
        if offset is not None:
            dx, dy, cw, ch = offset
        if is_empty and kind in self.settings.get("empty_count_offsets", {}):
            dx, dy, cw, ch = self.settings["empty_count_offsets"][kind]
        box = (x + px + dx, y + py + dy, x + px + dx + cw, y + py + dy + ch)
        if min(box) < 0 or box[2] > image.width or box[3] > image.height:
            raise RuntimeError("Army count region is outside the image")
        header = np.asarray(image.crop(box).convert("L"))
        if is_empty and kind in self.zero_samples:
            zero = self.zero_samples[kind]
            if zero.shape[0] <= ch and zero.shape[1] <= cw:
                if cv2.matchTemplate(header, zero, cv2.TM_CCOEFF_NORMED).max() >= 0.95:
                    return ArmyCard(kind, (x + px + sw // 2, y + py + sh // 2), 0, score)
            raise RuntimeError(
                "Empty army card has no matching zero-count marker; no selection made"
            )
        crop = np.asarray(image.crop(box).convert("L").resize((cw * 4, ch * 4)))
        values = []
        for cutoff in self.settings.get("count_thresholds", {}).get(kind, (175, 220)):
            mask = np.where(crop >= cutoff, 255, 0).astype(np.uint8)
            raw = self.ocr.image_to_string(
                Image.fromarray(mask), config="--psm 7 -c tessedit_char_whitelist=0123456789"
            )
            digits = re.sub(r"\s", "", raw)
            if not re.fullmatch(r"\d{1,3}", digits):
                raise RuntimeError(f"Cannot read {kind} count; no selection made")
            values.append(int(digits))
        if values[0] != values[1] or not 0 <= values[0] <= 300:
            raise RuntimeError("Army count readings disagree; no selection made")
        if is_empty and values[0] != 0:
            raise RuntimeError("Empty card artwork disagrees with its count; no selection made")
        return ArmyCard(kind, (x + px + sw // 2, y + py + sh // 2), values[0], score)


def saved_profile(root=Path(".")):
    return {
        "region": [0, 910, 1920, 170],
        "confidence": 0.9,
        "ambiguity_margin": 0.03,
        "count_offset": [18, -42, 66, 28],
        "templates": {
            kind: str(root / "templates" / "army" / (kind + ".png"))
            for kind in ("sneaky_goblins", "jump_spell")
        },
    }
