from PIL import Image, ImageEnhance
from typing import List
import logging

logger = logging.getLogger(__name__)


def preprocess_image(image_path: str) -> List[Image.Image]:
    try:
        img = Image.open(image_path)
    except Exception as e:
        logger.error(f"Failed to open image {image_path}: {e}")
        return []

    original_mode = img.mode
    if img.mode != "RGB":
        img = img.convert("RGB")
        logger.debug(f"Converted image from {original_mode} to RGB")

    width, height = img.size
    if width < 900:
        scale_factor = 900 / width
        new_width = 900
        new_height = int(height * scale_factor)
        img = img.resize((new_width, new_height), Image.LANCZOS)
        width, height = img.size

    img = ImageEnhance.Contrast(img).enhance(1.25)
    img = ImageEnhance.Sharpness(img).enhance(1.4)

    width, height = img.size
    aspect_ratio = height / width

    if aspect_ratio > 2.5:
        crops = _tile_vertical(img, width, height)
        logger.info(f"Tiled tall image (h/w={aspect_ratio:.1f}) into {len(crops)} crops")
        return crops
    return [img]


def _tile_vertical(img: Image.Image, width: int, height: int) -> List[Image.Image]:
    crop_height = int(width * 1.5)
    overlap = int(crop_height * 0.2)
    stride = crop_height - overlap

    crops = []
    y = 0
    while y < height:
        y_end = min(y + crop_height, height)
        crops.append(img.crop((0, y, width, y_end)))
        if y_end >= height:
            break
        y += stride

    return crops if crops else [img]
