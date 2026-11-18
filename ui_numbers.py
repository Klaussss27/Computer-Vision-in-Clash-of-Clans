"""Read a numeric game UI field only when two image preparations agree."""

import re

import cv2
import numpy as np
from PIL import Image


def read_number(image, region, ocr, modes=("gray", 175)):
    if len(modes) != 2:
        raise ValueError("Numeric recognition requires two preparations")
    if any(mode != "gray" and (type(mode) is not int or not 0 < mode < 255) for mode in modes):
        raise ValueError("Invalid numeric image preparation")
    x, y, width, height = region
    if (
        min(x, y) < 0
        or min(width, height) <= 0
        or x + width > image.width
        or y + height > image.height
    ):
        raise ValueError("Numeric region must fit the image")
    gray = np.asarray(
        image.crop((x, y, x + width, y + height)).convert("L").resize((width * 4, height * 4))
    )
    readings = []
    for mode in modes:
        prepared = 255 - gray if mode == "gray" else np.where(gray >= mode, 0, 255).astype(np.uint8)
        prepared = cv2.copyMakeBorder(prepared, 16, 16, 16, 16, cv2.BORDER_CONSTANT, value=255)
        raw = ocr.image_to_string(
            Image.fromarray(prepared), config="--psm 7 -c tessedit_char_whitelist=0123456789"
        )
        digits = re.sub(r"\s", "", raw)
        if not re.fullmatch(r"\d{1,12}", digits):
            raise RuntimeError("Numeric field is not clearly readable")
        readings.append(int(digits))
    if readings[0] != readings[1]:
        raise RuntimeError("Numeric readings disagree")
    return readings[0]
