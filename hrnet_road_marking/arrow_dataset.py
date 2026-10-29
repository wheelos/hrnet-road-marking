"""Complete annotated arrow crops mixed with ordinary road crops."""

import json
import math

import numpy as np
from PIL import Image, ImageOps
import torch

from .dataset import Vistas, flip_labels

ARROWS = tuple(range(39, 45))
FLIPPED = {39: 41, 41: 39, 42: 43, 43: 42}


def complete_window(box, size, crop, rng):
    """Choose a context window containing the entire annotated bounding box.

    Only visible annotated pixels are guaranteed; an occluded arrow is not
    completed. Padding is allowed at image edges and ignored in the loss.
    """
    x0, y0, x1, y1 = box
    width, height = size
    crop_h, crop_w = crop
    occupancy = float(rng.uniform(0.35, 0.65))
    scale = min(2.0, occupancy * crop_w / (x1 - x0), occupancy * crop_h / (y1 - y0))
    window_w = math.ceil(crop_w / scale)
    window_h = math.ceil(crop_h / scale)

    def origin(low, high, window, extent):
        start_min = max(high - window, min(0, extent - window))
        start_max = min(low, max(0, extent - window))
        if start_min > start_max:
            raise RuntimeError("Invalid containment interval")
        margin = int(min(window * 0.1, (window - (high - low)) * 0.25))
        contextual_min = max(start_min, high - window + margin)
        contextual_max = min(start_max, low - margin)
        if contextual_min <= contextual_max:
            start_min, start_max = contextual_min, contextual_max
        return int(rng.integers(start_min, start_max + 1))

    left = origin(x0, x1, window_w, width)
    top = origin(y0, y1, window_h, height)
    result = (left, top, left + window_w, top + window_h)
    if not (left <= x0 < x1 <= result[2] and top <= y0 < y1 <= result[3]):
        raise RuntimeError("Annotated arrow would be clipped")
    return result


def padded_crop(image, window, fill):
    left, top, right, bottom = window
    padding = (max(0, -left), max(0, -top), max(0, right - image.width), max(0, bottom - image.height))
    expanded = ImageOps.expand(image, padding, fill=fill)
    return expanded.crop((left + padding[0], top + padding[1], right + padding[0], bottom + padding[1]))


class ArrowVistas(Vistas):
    def __init__(self, root, index_path, crop=(512, 768), seed=20261026):
        super().__init__(root, "training", crop=crop, seed=seed)
        index = json.loads(index_path.read_text())
        self.records = [r for r in index["training"] if r["eligible"]]
        image_indices = {p.stem: i for i, p in enumerate(self.images)}
        self.record_image_indices = [image_indices[r["image"]] for r in self.records]
        self.class_records = {
            c: [i for i, r in enumerate(self.records) if r["class_id"] == c] for c in ARROWS
        }
        if not self.records:
            raise RuntimeError("No eligible arrow instances; check the training index")
        if {r["image"] for r in index["validation"]} & set(image_indices):
            raise RuntimeError("Validation image present in arrow training index")

    def crop_target(self, record_index, seed):
        record = self.records[record_index]
        path = self.images[self.record_image_indices[record_index]]
        with Image.open(path) as source:
            image = source.convert("RGB")
        with Image.open(self.root / "training/v2.0/labels" / (path.stem + ".png")) as source:
            if source.mode not in ("P", "L"):
                raise RuntimeError("Expected original v2 palette indices")
            mask = source.copy()
        if list(image.size) != record["size"] or mask.size != image.size:
            raise RuntimeError("Image dimensions changed since instance indexing")
        rng = np.random.default_rng(seed)
        window = complete_window(record["bbox"], image.size, self.crop, rng)
        output_size = (self.crop[1], self.crop[0])
        image = padded_crop(image, window, (124, 116, 104)).resize(output_size, Image.Resampling.BILINEAR)
        mask = padded_crop(mask, window, 255).resize(output_size, Image.Resampling.NEAREST)
        array = np.asarray(mask)
        flipped = bool(rng.random() < 0.5)
        if flipped:
            image = ImageOps.mirror(image)
            array = flip_labels(array)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            image = self.jitter(image)
        info = dict(
            record,
            source_window=list(window),
            flipped=flipped,
            output_class=FLIPPED.get(record["class_id"], record["class_id"])
            if flipped
            else record["class_id"],
        )
        return image, array, info

    def __getitem__(self, index):
        if not isinstance(index, tuple) or len(index) == 2:
            return super().__getitem__(index)
        image_index, seed, record_index = index
        if image_index != self.record_image_indices[record_index]:
            raise RuntimeError("Sampler instance/image mismatch")
        image, mask, _ = self.crop_target(record_index, seed)
        tensor = torch.from_numpy(np.array(image).transpose(2, 0, 1).copy()).float() / 255
        target = torch.from_numpy(self.lookup[mask].astype(np.int64))
        return (tensor - self.mean) / self.std, target
