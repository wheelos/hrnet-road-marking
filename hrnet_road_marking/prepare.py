"""Validate Vistas v2 splits and index complete arrow instances locally."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image

from .runtime import write_json


def inspect_instance(payload, label_path, image):
    with Image.open(io.BytesIO(payload)) as source:
        array = np.array(source)
    if array.dtype.kind not in "ui" or array.ndim != 2 or array.max() > 65535:
        raise ValueError(f"Expected a 16-bit instance-ID image: {image}")
    with Image.open(label_path) as source:
        if source.mode not in ("P", "L"):
            raise ValueError(f"Expected original palette label IDs: {label_path}")
        labels = np.array(source)
    if not np.array_equal(array // 256, labels):
        raise ValueError(f"Instance/semantic class mismatch: {image}")
    y, x = np.nonzero((array >= 39 * 256) & (array < 45 * 256))
    ids = array[y, x]
    height, width = array.shape
    rows = []
    for instance_id in np.unique(ids):
        keep = ids == instance_id
        xs, ys = x[keep], y[keep]
        box = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        edge = box[0] == 0 or box[1] == 0 or box[2] == width or box[3] == height
        tiny = int(keep.sum()) < 64 or max(box[2] - box[0], box[3] - box[1]) < 16
        rows.append(
            dict(
                image=image,
                class_id=int(instance_id // 256),
                instance_id=int(instance_id),
                bbox=box,
                size=[width, height],
                pixels=int(keep.sum()),
                touches_edge=edge,
                tiny=tiny,
                eligible=not edge and not tiny,
            )
        )
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True, help="Extracted images, labels and config_v2.0.json")
    p.add_argument("--instances", type=Path, help="Directory of instance PNGs or a local ZIP archive")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument(
        "--allow-subset", action="store_true", help="Tests only; accept fewer than 18000/2000 images"
    )
    a = p.parse_args()
    labels = json.loads((a.data / "config_v2.0.json").read_text())["labels"]
    if len(labels) != 124:
        p.error("Expected Vistas v2, 124 original labels")
    stems = {}
    for split, expected in [("training", 18000), ("validation", 2000)]:
        images = sorted((a.data / split / "images").glob("*.jpg"))
        stems[split] = {f.stem for f in images}
        masks = {f.stem for f in (a.data / split / "v2.0/labels").glob("*.png")}
        if not images or stems[split] != masks:
            p.error(f"{split}: missing/unpaired images and labels")
        if not a.allow_subset and len(images) != expected:
            p.error(f"{split}: expected {expected} images, got {len(images)}")
    if stems["training"] & stems["validation"]:
        p.error("Training/validation overlap")
    write_json(
        a.output / "dataset_audit.json",
        dict(images={k: len(v) for k, v in stems.items()}, subset=a.allow_subset, classes=124),
    )
    if not a.instances:
        print("Image/label pairs and split separation verified; no instance index requested")
        return
    archive = zipfile.ZipFile(a.instances) if a.instances.is_file() else None
    try:
        members = (
            [n for n in archive.namelist() if "/instances/" in "/" + n and n.endswith(".png")]
            if archive
            else sorted(a.instances.rglob("*.png"))
        )
        by_stem = {}
        for member in members:
            stem = Path(member).stem
            if stem in by_stem:
                raise ValueError(f"Duplicate instance stem: {stem}")
            by_stem[stem] = member
        inventory = {}
        for split, image_ids in stems.items():
            if image_ids - by_stem.keys():
                raise ValueError(f"{split}: missing {len(image_ids - by_stem.keys())} instance masks")

            def inspect(stem):
                member = by_stem[stem]
                payload = archive.read(member) if archive else member.read_bytes()
                return inspect_instance(payload, a.data / split / "v2.0/labels" / (stem + ".png"), stem)

            rows = []
            with ThreadPoolExecutor(a.workers) as pool:
                for i, found in enumerate(pool.map(inspect, sorted(image_ids))):
                    rows.extend(found)
                    if (i + 1) % 1000 == 0:
                        print(split, i + 1, flush=True)
            inventory[split] = rows
        write_json(a.output / "arrow_instances.json", inventory)
        summary = {
            k: dict(
                images=len({r["image"] for r in rows}),
                eligible=dict(Counter(r["class_id"] for r in rows if r["eligible"])),
            )
            for k, rows in inventory.items()
        }
        write_json(a.output / "arrow_index_summary.json", summary)
        images = sorted(stems["validation"])
        indices = np.random.default_rng(20261026).permutation(len(images))[:200]
        selected = {images[i] for i in indices} | {r["image"] for r in inventory["validation"]}
        write_json(a.output / "validation_ids.json", sorted(selected))
        print(json.dumps(summary), "validation_panel=", len(selected))
    finally:
        if archive:
            archive.close()


if __name__ == "__main__":
    main()
