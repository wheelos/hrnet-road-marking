"""Palette indices are labels; RGB palette colors are not unique in Vistas."""

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter

MARKINGS = tuple(range(35, 59))
FOCUS = (35, 36, 39, 40, 41, 42, 43, 44, 45, 48, 49, 51)


def flip_labels(mask):
    result = np.fliplr(mask).copy()
    original = result.copy()
    for left, right in ((39, 41), (42, 43)):
        result[original == left] = right
        result[original == right] = left
    return result


class Vistas(Dataset):
    def __init__(self, root, split, crop=(512, 768), seed=20261026, samples=None):
        self.root = Path(root)
        self.split = split
        self.crop = crop
        self.seed = seed
        self.labels = json.loads((self.root / "config_v2.0.json").read_text())["labels"]
        if len(self.labels) != 124:
            raise ValueError("Expected Mapillary Vistas v2 with 124 original labels")
        self.lookup = np.arange(256, dtype=np.uint8)
        self.lookup[len(self.labels) :] = 255
        for i, label in enumerate(self.labels):
            if not label["evaluate"]:
                self.lookup[i] = 255
        self.images = sorted((self.root / split / "images").glob("*.jpg"))
        if samples is not None:
            indices = np.random.default_rng(seed).permutation(len(self.images))[:samples]
            self.images = [self.images[i] for i in indices]
        if not self.images:
            raise RuntimeError(f"No images: {self.root}/{split}")
        self.jitter = ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.025)
        self.mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        self.std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # A (image_index, augmentation_seed) pair makes resume reproducible
        # regardless of worker prefetching.
        if isinstance(index, tuple):
            index, augmentation_seed = index
        else:
            augmentation_seed = self.seed + index
        rng = np.random.default_rng(augmentation_seed)
        path = self.images[index]
        label_path = self.root / self.split / "v2.0/labels" / (path.stem + ".png")
        with Image.open(path) as source:
            image = source.convert("RGB")
        with Image.open(label_path) as source:
            if source.mode not in ("P", "L"):
                raise ValueError(f"Expected palette label IDs, got {source.mode}: {label_path}")
            mask = source.copy()
        if image.size != mask.size:
            raise ValueError(f"Image/label size mismatch: {path.name}")
        if self.split == "training":
            scale = float(rng.uniform(768, 1280)) / min(image.size)
            size = tuple(max(1, round(v * scale)) for v in image.size)
        else:
            scale = min(1.0, 1536 / max(image.size))
            size = tuple(max(1, round(v * scale)) for v in image.size)
        image = image.resize(size, Image.Resampling.BILINEAR)
        mask = mask.resize(size, Image.Resampling.NEAREST)
        if self.split == "training":
            height, width = self.crop
            image = ImageOps.expand(image, (0, 0, max(0, width - size[0]), max(0, height - size[1])))
            mask = ImageOps.expand(mask, (0, 0, max(0, width - size[0]), max(0, height - size[1])), fill=255)
            array = np.asarray(mask)
            # Half the crops are centered on a randomly selected marking class
            # present in this training image; validation is never oversampled.
            present = [c for c in FOCUS if np.any(array[::4, ::4] == c)]
            if present and rng.random() < 0.5:
                chosen = int(rng.choice(present))
                points = np.argwhere(array[::4, ::4] == chosen)
                y, x = points[rng.integers(len(points))] * 4
                x = int(np.clip(x - rng.integers(width // 4, 3 * width // 4), 0, image.width - width))
                y = int(np.clip(y - rng.integers(height // 4, 3 * height // 4), 0, image.height - height))
            else:
                x = int(rng.integers(image.width - width + 1))
                y = int(rng.integers(image.height - height + 1))
            # Void regions are not supervised. Retry an all-ignored crop rather
            # than letting cross entropy produce NaN hours into a long run.
            for attempt in range(8):
                if np.any(self.lookup[array[y : y + height, x : x + width]] != 255):
                    break
                x = int(rng.integers(image.width - width + 1))
                y = int(rng.integers(image.height - height + 1))
            else:
                valid = np.argwhere(self.lookup[array] != 255)
                if not len(valid):
                    raise RuntimeError(f"Entire training image has ignored labels: {path}")
                cy, cx = valid[rng.integers(len(valid))]
                x = int(np.clip(cx - width // 2, 0, image.width - width))
                y = int(np.clip(cy - height // 2, 0, image.height - height))
            box = (x, y, x + width, y + height)
            image, mask = image.crop(box), mask.crop(box)
            array = np.asarray(mask)
            if rng.random() < 0.5:
                image = ImageOps.mirror(image)
                array = flip_labels(array)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(augmentation_seed))
                image = self.jitter(image)
        else:
            array = np.asarray(mask)
        image = torch.from_numpy(np.array(image).transpose(2, 0, 1).copy()).float() / 255
        return (image - self.mean) / self.std, torch.from_numpy(self.lookup[array].astype(np.int64))


class TrainingOrder:
    def __init__(self, images, microsteps, start=0, seed=20261026):
        self.images = images
        self.microsteps = microsteps
        self.start = start
        self.seed = seed

    def __iter__(self):
        epoch = -1
        for step in range(self.start, self.microsteps):
            new_epoch, offset = divmod(step, self.images)
            if new_epoch != epoch:
                epoch = new_epoch
                permutation = np.random.default_rng(self.seed + epoch).permutation(self.images)
            yield int(permutation[offset]), self.seed + step

    def __len__(self):
        return self.microsteps - self.start
