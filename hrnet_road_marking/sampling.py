"""Deterministic 75% ordinary / 25% complete-arrow sampling."""

import json
from pathlib import Path
import numpy as np
import torch
from .arrow_dataset import ArrowVistas
from .dataset import TrainingOrder, Vistas
from .taxonomy import LOOKUP


class CoarseTraining:
    def __init__(self, data, arrow_index, seed=20261028, crop=(512, 768)):
        self.base = ArrowVistas(data, arrow_index, seed=seed, crop=crop)
        self.records = self.base.records
        self.record_image_indices = self.base.record_image_indices
        self.images = self.base.images
        self.mean, self.std = self.base.mean, self.base.std

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image, original = self.base[index]
        target = torch.from_numpy(LOOKUP[original.numpy()].astype(np.int64))
        return image, target


class CoarseOrder:
    """75% ordinary road crops, 25% complete arrows; no direction balancing."""

    def __init__(self, dataset, microsteps, start=0, seed=20261028):
        self.dataset, self.microsteps, self.start, self.seed = dataset, microsteps, start, seed

    def __len__(self):
        return self.microsteps - self.start

    def __iter__(self):
        ordinary = iter(
            TrainingOrder(
                len(self.dataset),
                self.microsteps - self.microsteps // 4,
                self.start - self.start // 4,
                self.seed,
            )
        )
        for step in range(self.start, self.microsteps):
            seed = self.seed + step
            if step % 4 != 3:
                image_index, _ = next(ordinary)
                yield image_index, seed
            else:
                rng = np.random.default_rng(self.seed + 10000000 + step)
                r = int(rng.integers(len(self.dataset.records)))
                yield self.dataset.record_image_indices[r], seed, r


def validation(data, manifest=None, samples=None):
    result = Vistas(data, "validation", seed=20261026, samples=samples)
    if manifest:
        ids = json.loads(Path(manifest).read_text())
        assert len(ids) == len(set(ids))
        result.images = [p for p in result.images if p.stem in set(ids)]
        assert len(result.images) == len(ids)
    # Keep original fine labels for category-specific exclusion recall auditing.
    return result
